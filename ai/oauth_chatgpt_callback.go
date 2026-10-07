package ai

import (
	"errors"
	"net"
	"net/http"
	"strconv"
	"sync"

	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

type ChatGPTOAuthAuthorization struct{ Code, ClientID string }
type ChatGPTOAuthCallbackOptions struct {
	Host, State string
	Port        int
}

// Ready is shared by all observers. Result is read after Ready closes. Closing
// an unused listener does not settle the authorization result, matching Pi.
type ChatGPTOAuthCallback struct {
	URL    string
	Ready  <-chan struct{}
	Result func() (ChatGPTOAuthAuthorization, error)
	Close  func()
}

func StartChatGPTOAuthCallback(options ChatGPTOAuthCallbackOptions) (*ChatGPTOAuthCallback, error) {
	listener, err := net.Listen("tcp", net.JoinHostPort(options.Host, strconv.Itoa(options.Port)))
	if err != nil {
		return nil, err
	}
	ready := make(chan struct{})
	var once sync.Once
	var result ChatGPTOAuthAuthorization
	var failure error
	finish := func(value ChatGPTOAuthAuthorization, err error) {
		once.Do(func() { result, failure = value, err; close(ready) })
	}
	page := func(w http.ResponseWriter, status int, body string) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		// Close forcefully drops sockets; send a complete, length-delimited page
		// before waking the flow that may immediately close this listener.
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		base, _ := whatwg.Parse(chatGPTOAuthRedirect)
		parsed, err := base.Parse(r.RequestURI)
		if err != nil {
			page(w, 500, OAuthErrorHTML("Internal error while processing the callback."))
			return
		}
		if parsed.Pathname() != "/auth/callback" {
			page(w, 404, OAuthErrorHTML("Callback route not found."))
			return
		}
		query := urlQueryValues(parsed.Query())
		if problem := query["error"]; problem != "" {
			page(w, 400, OAuthErrorHTML("ChatGPT was not connected.", "Error: "+problem))
			finish(ChatGPTOAuthAuthorization{}, errors.New("ChatGPT authorization failed: "+problem))
			return
		}
		value, err := chatGPTOAuthCallbackAuthorization(query, options.State)
		if err != nil {
			page(w, 400, OAuthErrorHTML(err.Error()))
			return
		}
		page(w, 200, OAuthSuccessHTML("ChatGPT authentication completed. You can close this window."))
		finish(value, nil)
	})}
	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			finish(ChatGPTOAuthAuthorization{}, err)
		}
	}()
	return &ChatGPTOAuthCallback{URL: "http://" + listener.Addr().String() + "/auth/callback", Ready: ready, Result: func() (ChatGPTOAuthAuthorization, error) { <-ready; return result, failure }, Close: func() { _ = server.Close() }}, nil
}
