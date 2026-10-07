package ai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

type OAuthCallbackServerOptions struct {
	ProviderName string
	Host         string
	Port         int
	Path         string
	RedirectHost *string
	State        *string
	Complete     func(string) (any, error)
	Context      context.Context
	TimeoutMS    *float64
}

// OAuthCallbackServer is a concrete callback boundary. Wait shares one result,
// retaining its identity; cancellation without a claimed code returns Undefined.
// Close stops accepting connections and lets active completion callbacks finish.
type OAuthCallbackServer struct {
	RedirectURI string
	Wait        func() (any, error)
	Cancel      func()
	Close       func()
}

// StartOAuthCallbackServer binds the requested TCP port without fallback. Path,
// state and Complete remain live; callers synchronize any option mutations.
func StartOAuthCallbackServer(options *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
	name, signal := options.ProviderName, options.Context
	if signal != nil && signal.Err() != nil {
		return nil, errors.New(deviceCodeCancelMessage)
	}
	listener, err := net.Listen("tcp", net.JoinHostPort(options.Host, strconv.Itoa(options.Port)))
	if err != nil {
		return nil, err
	}
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		_ = listener.Close()
		return nil, errors.New("OAuth callback server did not bind to TCP")
	}
	var mu sync.Mutex
	claimed, settled, closed := false, false, false
	var value any
	var failure error
	var timer *time.Timer
	var stopAbort func() bool
	ready := make(chan struct{})
	// finish runs with mu held, keeping claim/settlement transitions atomic.
	finish := func(result any, err error) {
		if settled {
			return
		}
		settled, value, failure = true, result, err
		if timer != nil {
			timer.Stop()
		}
		if stopAbort != nil {
			stopAbort()
		}
		close(ready)
	}
	page := func(w http.ResponseWriter, status int, html string) {
		w.Header().Set("content-type", "text/html; charset=utf-8")
		w.Header().Set("cache-control", "no-store")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(html))
	}
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		base, _ := whatwg.Parse("http://localhost")
		url, err := base.Parse(request.RequestURI)
		if err != nil {
			// Invalid HTTP targets are outside Pi's typed callback contract.
			page(w, http.StatusBadRequest, OAuthErrorHTML("Callback route not found."))
			return
		}
		if request.Method != "GET" || url.Pathname() != options.Path {
			page(w, 404, OAuthErrorHTML("Callback route not found."))
			return
		}
		params := urlQueryValues(url.Query())
		state, hasState := params["state"]
		if options.State != nil && (!hasState || state != *options.State) {
			page(w, 400, OAuthErrorHTML("State mismatch."))
			return
		}
		mu.Lock()
		if claimed || settled {
			mu.Unlock()
			page(w, 409, OAuthErrorHTML("This sign-in has already been handled."))
			return
		}
		if code := params["error"]; code != "" {
			description := code
			if detail, exists := params["error_description"]; exists {
				description = detail
			}
			page(w, 400, OAuthErrorHTML(name+" authorization failed.", description))
			finish(nil, errors.New(name+" authorization failed: "+description))
			mu.Unlock()
			return
		}
		code := params["code"]
		if code == "" {
			mu.Unlock()
			page(w, 400, OAuthErrorHTML("Missing authorization code."))
			return
		}
		claimed = true
		mu.Unlock()
		result, err := invokeAuth(func() (any, error) { return options.Complete(code) })
		if err != nil {
			page(w, 502, OAuthErrorHTML(name+" sign-in failed.", err.Error()))
		} else {
			page(w, 200, OAuthSuccessHTML("Signed in to "+name+". You may now close this page."))
		}
		mu.Lock()
		finish(result, err)
		mu.Unlock()
	})}
	go func() {
		err := server.Serve(listener)
		mu.Lock()
		if !closed && err != nil {
			finish(nil, err)
		}
		mu.Unlock()
	}()
	mu.Lock()
	if signal != nil {
		stopAbort = context.AfterFunc(signal, func() {
			mu.Lock()
			finish(nil, errors.New(deviceCodeCancelMessage))
			mu.Unlock()
		})
	}
	if options.TimeoutMS != nil {
		ms := *options.TimeoutMS
		if math.IsNaN(ms) || ms < 1 || ms > math.MaxInt32 {
			ms = 1
		}
		timer = time.AfterFunc(time.Duration(math.Floor(ms))*time.Millisecond, func() {
			mu.Lock()
			finish(nil, errors.New(name+" sign-in timed out"))
			mu.Unlock()
		})
	}
	mu.Unlock()
	host := options.Host
	if options.RedirectHost != nil {
		host = *options.RedirectHost
	}
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	return &OAuthCallbackServer{
		RedirectURI: fmt.Sprintf("http://%s:%d%s", host, address.Port, options.Path),
		Wait: func() (any, error) {
			<-ready
			return value, failure
		},
		Cancel: func() {
			mu.Lock()
			// AbortSignal listeners run synchronously in Pi. Observe an abort
			// before cancelling even if Go's AfterFunc has not run yet.
			if signal != nil && signal.Err() != nil {
				finish(nil, errors.New(deviceCodeCancelMessage))
			} else if !claimed {
				finish(Undefined, nil)
			}
			mu.Unlock()
		},
		Close: func() {
			mu.Lock()
			finish(nil, errors.New("OAuth callback server closed"))
			alreadyClosed := closed
			closed = true
			mu.Unlock()
			if !alreadyClosed {
				_ = listener.Close()
				go func() { _ = server.Shutdown(context.Background()) }()
			}
		},
	}, nil
}

type OAuthManualPrompt struct{ Message, Placeholder string }

// WaitForCallbackOrManualInput cancels the manual prompt on every exit. A
// claimed browser callback keeps priority over successful manual input; manual
// failure is checked after a successful callback wait, as in Pi.
func WaitForCallbackOrManualInput(interaction ProviderAuthInteraction, callback *OAuthCallbackServer, prompt OAuthManualPrompt) (*Object, error) {
	manualContext, abortManual := context.WithCancel(context.Background())
	defer abortManual()
	var mu sync.Mutex
	var manualValue any
	var manualError error
	manualDone := make(chan struct{})
	go func() {
		value, err := invokeAuth(func() (any, error) {
			return interaction.Prompt(manualContext, NewObject(Property{Name: "type", Value: "manual_code"}, Property{Name: "message", Value: prompt.Message}, Property{Name: "placeholder", Value: prompt.Placeholder}))
		})
		mu.Lock()
		manualValue, manualError = value, err
		mu.Unlock()
		if callback != nil {
			callback.Cancel()
		}
		close(manualDone)
	}()
	value := any(Undefined)
	var err error
	if callback != nil {
		value, err = invokeAuth(callback.Wait)
		if err != nil {
			return nil, err
		}
	}
	mu.Lock()
	err = manualError
	mu.Unlock()
	if err != nil {
		return nil, err
	}
	if !jsonjs.IsUndefined(value) {
		return NewObject(Property{Name: "type", Value: "callback"}, Property{Name: "value", Value: value}), nil
	}
	<-manualDone
	mu.Lock()
	value, err = manualValue, manualError
	mu.Unlock()
	if err != nil {
		return nil, err
	}
	if jsonjs.IsNullish(value) {
		value = ""
	}
	return NewObject(Property{Name: "type", Value: "manual"}, Property{Name: "input", Value: value}), nil
}
