package ai

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/2found/2ai/ai/protocol"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

func piMessagesTestValue(t *testing.T, value any) any {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return catalogDecode(t, raw)
}

func TestPiMessagesHTTPProtocol(t *testing.T) {
	f := readPiMessagesProtocolFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) {
			model := catalogDecode(t, f.Model).(*Object)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			now := float64(1000000)
			data := tc.Input.Bytes
			if data == nil {
				data = []byte(string(jsonjs.StringCodePoints(tc.Input.Wire)))
			}
			closed := false
			rawEvents, snapshots := NewArray(), NewArray()
			retained := []AssistantMessageEvent{}
			var result *Message
			adapter := newPiMessagesTypedAdapter()
			options := PiMessagesStreamOptions{Values: NewObject(Property{Name: "apiKey", Value: "key"}), Now: func() float64 { return now }, Client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: &piMessagesTestBody{Reader: &piMessagesFixtureReader{data: data, size: tc.Input.ChunkSize, failure: tc.Input.ReadError}, close: func() { closed = true }}}, nil
			})}, OnProviderStreamEvent: func(_ context.Context, event any, _ *Object) error {
				if tc.Input.Action == "advance-clock" {
					now += 10
				}
				if tc.Input.Action == "mutate" && catalogProperty(event, "type") == "text_delta" {
					event.(*Object).Set("delta", "changed")
				}
				var clone jsonjs.ValueCloner
				rawEvents.Append(clone.Clone(event))
				if tc.Input.Action == "callback-abort" {
					cancel()
				}
				if tc.Input.Action == "callback-error" || tc.Input.Action == "callback-abort" {
					return errors.New("callback failed")
				}
				return nil
			}}
			producePiMessages(ctx, model, NewObject(Property{Name: "messages", Value: NewArray()}), options, newPiMessagesEventConverter(model, options.Now), nil, func(event *Object) error {
				typed, err := adapter.event(event)
				if err != nil {
					return err
				}
				snapshots.Append(piMessagesTestValue(t, typed))
				retained = append(retained, typed)
				if typed.Type == "done" {
					result = typed.Message
				}
				if typed.Type == "error" {
					result = typed.Error
				}
				return nil
			})
			catalogCompare(t, rawEvents, tc.RawEvents)
			catalogCompare(t, snapshots, tc.Snapshots)
			catalogCompare(t, piMessagesTestValue(t, retained), tc.Retained)
			catalogCompare(t, piMessagesTestValue(t, result), tc.Result)
			if !closed {
				t.Fatal("HTTP response body not closed")
			}
		})
	}
}

type piMessagesTestBody struct {
	io.Reader
	close func()
}

func (b *piMessagesTestBody) Close() error {
	if b.close != nil {
		b.close()
	}
	return nil
}

type piMessagesHTTPFixture struct {
	UpstreamCommit string
	Model          json.RawMessage
	Cases          []struct{ Input, Trace, Requests, Events, Result json.RawMessage }
}

func TestPiMessagesHTTP(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-messages-http.json")
	if err != nil {
		t.Fatal(err)
	}
	var f piMessagesHTTPFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 100 {
		t.Fatal("unexpected HTTP fixture coverage")
	}
	for i, tc := range f.Cases {
		input := catalogDecode(t, tc.Input).(*Object)
		t.Run(strconv.Itoa(i)+"-"+input.Get("name").(string), func(t *testing.T) {
			model := catalogDecode(t, f.Model).(*Object)
			authSpreadInto(model, catalogProperty(input, "model"))
			controls := NewObject(Property{Name: "apiKey", Value: "key"})
			authSpreadInto(controls, catalogProperty(input, "values"))
			if catalogEntryTruthy(input.Get("missingKey")) {
				controls.Delete("apiKey")
			}
			env, _ := input.Get("env").(string)
			t.Setenv("PI_CACHE_RETENTION", env)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			action, _ := input.Get("action").(string)
			now := float64(1000000)
			trace, requests := NewArray(), NewArray()
			snapshot := func(value any) any { var clone jsonjs.ValueCloner; return clone.Clone(value) }
			options := PiMessagesStreamOptions{Values: controls, Now: func() float64 { return now }, ErrorStack: func(error) string { return "fixture stack" }}
			options.OnPayload = func(_ context.Context, payload any, received *Object) (any, error) {
				trace.Append(NewObject(Property{Name: "type", Value: "payload"}, Property{Name: "payload", Value: snapshot(payload)}, Property{Name: "model", Value: snapshot(received)}))
				switch action {
				case "payload-error":
					return nil, errors.New("payload failed")
				case "payload-null":
					return nil, nil
				case "payload-string":
					return "replacement", nil
				case "payload-array":
					return NewArray(1, nil), nil
				case "payload-mutate":
					payload.(*Object).Get("options").(*Object).Set("extra", "changed")
				case "payload-change-model":
					received.Set("id", "changed")
					received.Set("provider", "changed")
				case "payload-change-controls":
					controls.Set("headers", NewObject(Property{Name: "authorization", Value: "changed"}))
					controls.Set("apiKey", "new-key")
				case "clock":
					now += 10
				}
				return Undefined, nil
			}
			options.OnResponse = func(_ context.Context, response CompletionsResponse, received *Object) error {
				trace.Append(NewObject(Property{Name: "type", Value: "response"}, Property{Name: "response", Value: piMessagesTestValue(t, response)}, Property{Name: "model", Value: snapshot(received)}))
				if action == "response-abort" {
					cancel()
				}
				if action == "response-error" || action == "response-abort" {
					return errors.New("response failed")
				}
				if action == "clock" {
					now += 10
				}
				return nil
			}
			options.OnProviderStreamEvent = func(_ context.Context, event any, received *Object) error {
				trace.Append(NewObject(Property{Name: "type", Value: "event"}, Property{Name: "event", Value: snapshot(event)}, Property{Name: "model", Value: snapshot(received)}))
				if action == "stream-abort" {
					cancel()
				}
				if action == "stream-error" || action == "stream-abort" {
					return errors.New("stream failed")
				}
				if action == "clock" {
					now += 10
				}
				return nil
			}
			options.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				body := []byte{}
				if request.Body != nil {
					body, _ = io.ReadAll(request.Body)
				}
				headers := NewObject()
				for name, values := range request.Header {
					headers.Set(strings.ToLower(name), strings.Join(values, ", "))
				}
				requests.Append(NewObject(Property{Name: "url", Value: request.URL.String()}, Property{Name: "method", Value: request.Method}, Property{Name: "headers", Value: headers}, Property{Name: "body", Value: string(body)}))
				if action == "fetch-abort" {
					cancel()
				}
				if action == "fetch-error" || action == "fetch-abort" {
					return nil, errors.New("fetch failed")
				}
				status := 200
				if value, ok := input.Get("status").(float64); ok {
					status = int(value)
				}
				statusText, _ := input.Get("statusText").(string)
				wire := "data: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{\"input\":1,\"output\":2,\"cacheRead\":0,\"cacheWrite\":0,\"totalTokens\":3,\"cost\":{\"input\":0,\"output\":0,\"cacheRead\":0,\"cacheWrite\":0,\"total\":0}}}\n\n"
				if value, ok := input.Get("body").(string); ok {
					wire = value
				}
				if action == "http-read-error" {
					return &http.Response{StatusCode: status, Header: http.Header{"X-Result": {"yes"}, "X-List": {"a, b"}}, Body: io.NopCloser(&piMessagesFixtureReader{failure: true})}, nil
				}
				return &http.Response{StatusCode: status, Status: strconv.Itoa(status) + " " + statusText, Header: http.Header{"X-Result": {"yes"}, "X-List": {"a, b"}}, Body: io.NopCloser(strings.NewReader(wire))}, nil
			})}
			adapter := newPiMessagesTypedAdapter()
			events := []AssistantMessageEvent{}
			var result *Message
			producePiMessages(ctx, model, NewObject(Property{Name: "messages", Value: NewArray()}), options, newPiMessagesEventConverter(model, options.Now), nil, func(event *Object) error {
				typed, err := adapter.event(event)
				if err != nil {
					return err
				}
				events = append(events, typed)
				if typed.Type == "done" {
					result = typed.Message
				}
				if typed.Type == "error" {
					result = typed.Error
				}
				return nil
			})
			catalogCompare(t, trace, tc.Trace)
			catalogCompare(t, requests, tc.Requests)
			catalogCompare(t, piMessagesTestValue(t, events), tc.Events)
			catalogCompare(t, piMessagesTestValue(t, result), tc.Result)
		})
	}
}

func TestRadiusNativePiMessagesHTTP(t *testing.T) {
	requests := make(chan string, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests <- string(body)
		if r.URL.Path != "/v1/messages" || r.Header.Get("Authorization") != "Bearer request-key" || r.Header.Get("X-Call") != "callback" {
			t.Errorf("unexpected request %s %v", r.URL.Path, r.Header)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"text_start\",\"contentIndex\":0}\n\ndata: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"hello\"}\n\ndata: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{\"input\":1,\"output\":2,\"totalTokens\":3,\"cost\":{\"total\":0}}}\n\n")
	}))
	defer server.Close()
	gateway := server.URL
	provider, err := RadiusProvider(RadiusProviderOptions{Gateway: &gateway}, RadiusProviderDependencies{Client: server.Client(), Now: func() float64 { return 1000000 }})
	if err != nil {
		t.Fatal(err)
	}
	registry := NewModels(ModelsOptions{AuthContext: &AuthContext{Env: func(context.Context, string) (any, error) { return Undefined, nil }}})
	registry.SetProvider(provider)
	model := catalogDecode(t, readPiMessagesProtocolFixture(t).Model).(*Object)
	model.Set("provider", "radius")
	model.Set("baseUrl", gateway+"/v1")
	for _, simple := range []bool{false, true} {
		options := NewObject(Property{Name: "apiKey", Value: "request-key"})
		called := false
		options.Set("onPayload", func(_ context.Context, payload any, _ *Object) (any, error) {
			called = true
			options.Set("headers", NewObject(Property{Name: "X-Call", Value: "callback"}))
			// Models captures request options; mutate the actual payload independently.
			payload.(*Object).Get("options").(*Object).Set("custom", true)
			return Undefined, nil
		})
		options.Set("headers", NewObject(Property{Name: "X-Call", Value: "callback"}))
		var stream *AssistantMessageEventStream
		if simple {
			stream = registry.StreamSimple(context.Background(), model, Context{}, &ModelsRequestOptions{Values: options})
		} else {
			stream = registry.Stream(context.Background(), model, Context{}, &ModelsRequestOptions{Values: options})
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		result, err := stream.SnapshotResult(ctx)
		if err == nil {
			err = stream.WaitForEnd(ctx)
		}
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		if !called || result.StopReason != "stop" || result.Content.Blocks.Get(0).Text != "hello" || result.Timestamp != 1000000 {
			t.Fatalf("unexpected native result %#v", result)
		}
		body := publicationAwait(t, requests)
		if !strings.Contains(body, `"custom":true`) {
			t.Fatalf("callback payload lost: %s", body)
		}
	}
}

func TestPiMessagesHTTPStreamingAndCancellation(t *testing.T) {
	for _, abort := range []bool{false, true} {
		t.Run(strconv.FormatBool(abort), func(t *testing.T) {
			release := make(chan struct{})
			var once sync.Once
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"type\":\"text_start\",\"contentIndex\":0}\n\ndata: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"first\"}\n\n")
				w.(http.Flusher).Flush()
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				for i := 0; i < 50; i++ {
					_, _ = io.WriteString(w, "data: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\".\"}\n\n")
					w.(http.Flusher).Flush()
				}
				_, _ = io.WriteString(w, "data: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{\"input\":1,\"output\":2,\"totalTokens\":3}}\n\n")
			}))
			defer func() { once.Do(func() { close(release) }); server.Close() }()
			model := catalogDecode(t, readPiMessagesProtocolFixture(t).Model).(*Object)
			model.Set("baseUrl", server.URL)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
			defer stop()
			stream := StreamPiMessages(ctx, model, NormalizeContext(Context{}), PiMessagesStreamOptions{Values: NewObject(Property{Name: "apiKey", Value: "key"}), Client: server.Client()})
			var partial *Message
			for {
				event, ok, err := stream.Next(wait)
				if err != nil || !ok {
					t.Fatalf("missing progressive event: %v", err)
				}
				if event.Type == "text_delta" {
					partial = event.Partial
					break
				}
			}
			first, err := stream.SnapshotEvent(AssistantMessageEvent{Partial: partial})
			if err != nil {
				t.Fatal(err)
			}
			if first.Partial.Content.Blocks.Get(0).Text != "first" {
				t.Fatal("provider buffered progressive output")
			}
			if abort {
				cancel()
			} else {
				once.Do(func() { close(release) })
			}
			// Observe the live object concurrently with subsequent producer updates.
			observed := make(chan struct{})
			go func() {
				defer close(observed)
				for i := 0; i < 100; i++ {
					_, _ = stream.SnapshotEvent(AssistantMessageEvent{Partial: partial})
				}
			}()
			result, err := stream.SnapshotResult(wait)
			if err != nil {
				t.Fatal(err)
			}
			if err = stream.WaitForEnd(wait); err != nil {
				t.Fatal(err)
			}
			publicationAwait(t, observed)
			if abort {
				if result.StopReason != "aborted" || result.Content.Blocks.Len() != 0 {
					t.Fatalf("aborted result retained partial content: %#v", result)
				}
				retained, _ := stream.SnapshotEvent(AssistantMessageEvent{Partial: partial})
				if retained.Partial.Content.Blocks.Get(0).Text != "first" {
					t.Fatal("partial message mutated into failure")
				}
			} else {
				native, err := stream.Result(wait)
				if err != nil {
					t.Fatal(err)
				}
				if native != partial || result.Content.Blocks.Get(0).Text != "first"+strings.Repeat(".", 50) {
					t.Fatal("terminal lost partial identity or text")
				}
			}
			if first.Partial.Content.Blocks.Get(0).Text != "first" {
				t.Fatal("snapshot changed after producer updates")
			}
		})
	}
}

func TestPiMessagesHTTPFailureMetadata(t *testing.T) {
	for _, callback := range []bool{false, true} {
		t.Run(strconv.FormatBool(callback), func(t *testing.T) {
			closed := false
			ctx, capture := WithNativeProviderFailure(context.Background())
			options := PiMessagesStreamOptions{Values: NewObject(Property{Name: "apiKey", Value: "key"}), Client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 429, Status: "429 Too Many Requests", Header: http.Header{"Retry-After": {"7"}}, Body: &piMessagesTestBody{Reader: strings.NewReader(`{"error":{"message":"busy","code":"limit"}}`), close: func() { closed = true }}}, nil
			})}}
			if callback {
				options.OnResponse = func(context.Context, CompletionsResponse, *Object) error {
					return &PiMessagesResponseError{Message: "host failed", Status: 401}
				}
			}
			model := catalogDecode(t, readPiMessagesProtocolFixture(t).Model).(*Object)
			stream := StreamPiMessages(ctx, model, NormalizeContext(Context{}), options)
			wait, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			result, err := stream.Result(wait)
			if err != nil {
				t.Fatal(err)
			}
			if err = stream.WaitForEnd(wait); err != nil {
				t.Fatal(err)
			}
			if !closed || result.StopReason != "error" {
				t.Fatal("error did not settle and close body")
			}
			failure, ok := capture.Failure().(*protocol.ProviderError)
			if callback {
				if ok || !capture.HostFailure() {
					t.Fatal("callback failure classified as provider failure")
				}
			} else if !ok || failure.Status != 429 {
				t.Fatalf("lost provider metadata: %#v", capture.Failure())
			}
		})
	}
}
