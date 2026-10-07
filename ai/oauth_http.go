package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"golang.org/x/text/encoding/unicode"
)

func oauthFetch(ctx context.Context, client *http.Client, endpoint, method string, headers http.Header, body string) (*http.Response, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}
	if _, err := whatwg.Parse(endpoint); err != nil {
		return nil, errors.New("fetch() URL is invalid")
	}
	request, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, errors.New("fetch() URL is invalid")
	}
	request.Header = headers.Clone()
	response, err := client.Do(request)
	if wrapped, ok := err.(*url.Error); ok {
		err = wrapped.Err
	}
	if (err == context.Canceled || err == context.DeadlineExceeded) && ctx.Err() != nil {
		err = oauthCancelCause(ctx)
	}
	return response, err
}
func oauthRead(response *http.Response) ([]byte, error) {
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
	return raw, nil
}
func oauthReadJSON(response *http.Response) (any, error) {
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	// Bun fetch resolves an empty HTTP body as null. A byte-backed Response
	// constructed in a test throws instead; transport fixtures use real fetch
	// for this boundary. A BOM-only body is nonempty and still fails parsing.
	if len(raw) == 0 {
		return nil, nil
	}
	raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
	if err = jsonjs.ValidateJSON(raw); err != nil {
		return nil, errors.New("Failed to parse JSON")
	}
	return jsonjs.DecodeValue(raw)
}

// The derived signal stays linked after fetch returns, as AbortSignal.any does.
func oauthRequestTimeout(ctx context.Context, timeout time.Duration) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	name := "TimeoutError"
	derived, cancel := context.WithTimeoutCause(ctx, timeout, &OAuthDiagnosticError{Name: &name, Message: "The operation timed out.", Code: 23})
	context.AfterFunc(derived, cancel)
	return derived
}

// OAuth object readers accept arrays and swallow body/JSON failures, matching
// the device flows' typeof object guard. Other flows use oauthReadJSON directly.
func oauthReadObjectJSON(response *http.Response) any {
	value, err := oauthReadJSON(response)
	if err == nil {
		switch value.(type) {
		case *Object, *Array:
			return value
		}
	}
	return nil
}
