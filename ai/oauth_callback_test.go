package ai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

type oauthCallbackFixture struct {
	UpstreamCommit string
	Pages          []struct {
		Message        string
		Detail         *string
		Success, Error string
	}
	Cases []struct {
		Input struct {
			State              *string
			Completion, Action string
			RedirectHost       *string
			Requests           []struct{ Method, Path string }
		}
		Start, Responses, Codes, Output, AfterClose json.RawMessage
		URI                                         string
	}
	Races []struct {
		Action                             string
		Fail                               bool
		Duplicate, Early, Response, Output json.RawMessage
	}
	Manual []struct {
		CallbackMode, ManualMode string
		Output                   json.RawMessage
		Cancelled                int
		Aborted                  bool
	}
	Browser []struct {
		Mode               string
		Descriptor, Output json.RawMessage
		Aborted            bool
	}
}

func readOAuthCallbackFixture(t *testing.T) oauthCallbackFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-oauth-callback.json")
	if err != nil {
		t.Fatal(err)
	}
	var f oauthCallbackFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Pages) != 9 || len(f.Cases) != 28 || len(f.Races) != 8 || len(f.Manual) != 20 || len(f.Browser) != 4 {
		t.Fatal("unexpected callback fixture coverage")
	}
	return f
}
func callbackCapture(value any, err error) map[string]any {
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	if jsonjs.IsUndefined(value) {
		return map[string]any{"absent": true}
	}
	return map[string]any{"value": value}
}
func callbackPage(client *http.Client, method, uri string) (any, error) {
	request, err := http.NewRequest(method, uri, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	hash := sha256.Sum256(body)
	return map[string]any{"status": response.StatusCode, "contentType": response.Header.Get("Content-Type"), "cache": response.Header.Get("Cache-Control"), "sha256": hex.EncodeToString(hash[:])}, nil
}
func TestPiOAuthPages(t *testing.T) {
	for i, tc := range readOAuthCallbackFixture(t).Pages {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if actual := OAuthSuccessHTML(tc.Message); actual != tc.Success {
				t.Fatal("success HTML differs")
			}
			var actual string
			if tc.Detail == nil {
				actual = OAuthErrorHTML(tc.Message)
			} else {
				actual = OAuthErrorHTML(tc.Message, *tc.Detail)
			}
			if actual != tc.Error {
				t.Fatal("error HTML differs")
			}
		})
	}
}
func TestPiOAuthCallbackHTTP(t *testing.T) {
	client := &http.Client{Timeout: 3 * time.Second}
	defer client.CloseIdleConnections()
	for i, tc := range readOAuthCallbackFixture(t).Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.Input.Action == "preabort" {
				cancel()
			}
			value := NewObject(Property{Name: "token", Value: "shared"})
			failure := errors.New("exchange <failed>")
			codes := NewArray()
			options := &OAuthCallbackServerOptions{ProviderName: "Example <&>", Host: "127.0.0.1", Path: "/callback", State: tc.Input.State, RedirectHost: tc.Input.RedirectHost, Context: ctx, Complete: func(code string) (any, error) {
				codes.Append(code)
				switch tc.Input.Completion {
				case "failure":
					return nil, failure
				case "null":
					return nil, nil
				case "undefined":
					return Undefined, nil
				}
				return value, nil
			}}
			if tc.Input.Action == "timeout" {
				ms := float64(2)
				options.TimeoutMS = &ms
			}
			server, err := StartOAuthCallbackServer(options)
			if err != nil {
				if len(tc.Start) == 0 {
					t.Fatal(err)
				}
				catalogCompare(t, callbackCapture(nil, err), tc.Start)
				return
			}
			defer server.Close()
			if len(tc.Start) > 0 {
				t.Fatal("expected start failure")
			}
			normalized := regexp.MustCompile(`:\d+/`).ReplaceAllString(server.RedirectURI, ":PORT/")
			if normalized != tc.URI {
				t.Fatalf("redirect URI: %s", normalized)
			}
			switch tc.Input.Action {
			case "cancel":
				server.Cancel()
			case "abort":
				cancel()
				server.Wait()
			case "timeout":
				server.Wait()
			case "close":
				server.Close()
			}
			uri, err := url.Parse(server.RedirectURI)
			if err != nil {
				t.Fatal(err)
			}
			origin := uri.Scheme + "://" + uri.Host
			responses := NewArray()
			for _, request := range tc.Input.Requests {
				response, err := callbackPage(client, request.Method, origin+request.Path)
				if err != nil {
					t.Fatal(err)
				}
				responses.Append(response)
			}
			server.Cancel()
			result, err := server.Wait()
			output := callbackCapture(result, err)
			if err != nil {
				output["same"] = err == failure
			} else {
				output["same"] = result == value
			}
			catalogCompare(t, responses, tc.Responses)
			catalogCompare(t, codes, tc.Codes)
			catalogCompare(t, output, tc.Output)
			server.Close()
			result, err = server.Wait()
			catalogCompare(t, callbackCapture(result, err), tc.AfterClose)
		})
	}
}
func TestPiOAuthCallbackClaimedRaces(t *testing.T) {
	client := &http.Client{Timeout: 4 * time.Second}
	defer client.CloseIdleConnections()
	for _, tc := range readOAuthCallbackFixture(t).Races {
		t.Run(tc.Action+"/"+strconv.FormatBool(tc.Fail), func(t *testing.T) {
			started, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(release) })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			value := NewObject(Property{Name: "token", Value: "shared"})
			failure := errors.New("exchange failed")
			options := &OAuthCallbackServerOptions{ProviderName: "Example", Host: "127.0.0.1", Path: "/callback", Context: ctx, Complete: func(string) (any, error) {
				close(started)
				<-release
				if tc.Fail {
					return nil, failure
				}
				return value, nil
			}}
			if tc.Action == "timeout" {
				ms := float64(200)
				options.TimeoutMS = &ms
			}
			server, err := StartOAuthCallbackServer(options)
			if err != nil {
				t.Fatal(err)
			}
			defer server.Close()
			first := make(chan struct {
				page any
				err  error
			}, 1)
			go func() {
				p, e := callbackPage(client, "GET", server.RedirectURI+"?code=x")
				first <- struct {
					page any
					err  error
				}{p, e}
			}()
			select {
			case <-started:
			case <-time.After(3 * time.Second):
				t.Fatal("callback was not claimed")
			}
			duplicate, err := callbackPage(client, "GET", server.RedirectURI+"?code=y")
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, duplicate, tc.Duplicate)
			switch tc.Action {
			case "cancel":
				server.Cancel()
			case "abort":
				cancel()
			case "close":
				server.Close()
			case "timeout":
				server.Wait()
			}
			var early any
			if tc.Action != "cancel" {
				v, e := server.Wait()
				early = callbackCapture(v, e)
			}
			catalogCompare(t, early, tc.Early)
			once.Do(func() { close(release) })
			response := <-first
			if response.err != nil {
				t.Fatal(response.err)
			}
			catalogCompare(t, response.page, tc.Response)
			result, err := server.Wait()
			output := callbackCapture(result, err)
			if err != nil {
				output["same"] = err == failure
			} else {
				output["same"] = result == value
			}
			catalogCompare(t, output, tc.Output)
		})
	}
}
func TestOAuthCallbackPortInUse(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	server, err := StartOAuthCallbackServer(&OAuthCallbackServerOptions{Host: "127.0.0.1", Port: listener.Addr().(*net.TCPAddr).Port, Path: "/callback"})
	if server != nil {
		server.Close()
		t.Fatal("silently selected another port")
	}
	var op *net.OpError
	if !errors.As(err, &op) {
		t.Fatalf("missing original bind error: %v", err)
	}
}
func TestPiOAuthManualInputPriority(t *testing.T) {
	for _, tc := range readOAuthCallbackFixture(t).Manual {
		t.Run(tc.CallbackMode+"/"+tc.ManualMode, func(t *testing.T) {
			entered, releaseManual, releaseCallback, manualFinished := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
			var promptContext context.Context
			manualFailure, callbackFailure := errors.New("manual failed"), errors.New("callback failed")
			cancelled := 0
			var callback *OAuthCallbackServer
			if tc.CallbackMode != "none" {
				callback = &OAuthCallbackServer{Cancel: func() { cancelled++; close(manualFinished) }, Wait: func() (any, error) {
					<-releaseCallback
					switch tc.CallbackMode {
					case "error":
						return nil, callbackFailure
					case "undefined":
						return Undefined, nil
					case "null":
						return nil, nil
					}
					return NewObject(Property{Name: "token", Value: "browser"}), nil
				}}
			}
			parent, stop := context.WithCancel(context.Background())
			stop()
			interaction := ProviderAuthInteraction{Context: parent, Prompt: func(ctx context.Context, prompt *Object) (any, error) {
				promptContext = ctx
				if ctx.Err() != nil {
					t.Error("manual prompt inherited parent abort")
				}
				if prompt.Get("type") != "manual_code" || prompt.Get("message") != "paste" || prompt.Get("placeholder") != "url" {
					t.Error("prompt descriptor differs")
				}
				close(entered)
				<-releaseManual
				if callback == nil {
					close(manualFinished)
				}
				switch tc.ManualMode {
				case "error":
					return nil, manualFailure
				case "null":
					return nil, nil
				case "undefined":
					return Undefined, nil
				}
				return "pasted", nil
			}}
			completed := make(chan struct {
				value *Object
				err   error
			}, 1)
			go func() {
				v, e := WaitForCallbackOrManualInput(interaction, callback, OAuthManualPrompt{Message: "paste", Placeholder: "url"})
				completed <- struct {
					value *Object
					err   error
				}{v, e}
			}()
			<-entered
			close(releaseManual)
			<-manualFinished
			close(releaseCallback)
			result := <-completed
			output := callbackCapture(result.value, result.err)
			if result.err != nil {
				expected := manualFailure
				if tc.CallbackMode == "error" {
					expected = callbackFailure
				}
				output["same"] = result.err == expected
			}
			catalogCompare(t, output, tc.Output)
			if cancelled != tc.Cancelled || (promptContext.Err() != nil) != tc.Aborted {
				t.Fatalf("prompt cleanup: %d, %v", cancelled, promptContext.Err())
			}
		})
	}
}

func TestPiOAuthBrowserManualIntegration(t *testing.T) {
	client := &http.Client{Timeout: 3 * time.Second}
	defer client.CloseIdleConnections()
	for _, tc := range readOAuthCallbackFixture(t).Browser {
		t.Run(tc.Mode, func(t *testing.T) {
			entered, releaseManual, promptDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var promptContext context.Context
			var descriptor any
			failure := errors.New("exchange failed")
			server, err := StartOAuthCallbackServer(&OAuthCallbackServerOptions{ProviderName: "Example", Host: "127.0.0.1", Path: "/callback", Complete: func(string) (any, error) {
				switch tc.Mode {
				case "failure":
					return nil, failure
				case "null":
					return nil, nil
				case "undefined":
					return Undefined, nil
				}
				return "browser", nil
			}})
			if err != nil {
				t.Fatal(err)
			}
			defer server.Close()
			parent, stop := context.WithCancel(context.Background())
			stop()
			interaction := ProviderAuthInteraction{Context: parent, Prompt: func(ctx context.Context, prompt *Object) (any, error) {
				defer close(promptDone)
				promptContext = ctx
				descriptor = map[string]any{"type": prompt.Get("type"), "message": prompt.Get("message"), "placeholder": prompt.Get("placeholder"), "initiallyAborted": ctx.Err() != nil}
				close(entered)
				select {
				case <-ctx.Done():
					return nil, errors.New("manual aborted")
				case <-releaseManual:
					return "fallback", nil
				}
			}}
			completed := make(chan struct {
				value *Object
				err   error
			}, 1)
			go func() {
				v, e := WaitForCallbackOrManualInput(interaction, server, OAuthManualPrompt{Message: "paste", Placeholder: "url"})
				completed <- struct {
					value *Object
					err   error
				}{v, e}
			}()
			<-entered
			if _, err = callbackPage(client, "GET", server.RedirectURI+"?code=x"); err != nil {
				t.Fatal(err)
			}
			if tc.Mode == "undefined" {
				close(releaseManual)
			}
			result := <-completed
			select {
			case <-promptDone:
			case <-time.After(time.Second):
				t.Fatal("manual prompt not cancelled")
			}
			output := callbackCapture(result.value, result.err)
			if result.err != nil {
				output["same"] = result.err == failure
			}
			catalogCompare(t, output, tc.Output)
			catalogCompare(t, descriptor, tc.Descriptor)
			if (promptContext.Err() != nil) != tc.Aborted {
				t.Fatal("manual abort differs")
			}
		})
	}
}
