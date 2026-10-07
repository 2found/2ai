package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"golang.org/x/text/encoding/unicode"
)

const VendorPiMessages = "pi-messages"

// PiMessagesStreamOptions retains JSON-shaped values and serial callbacks.
// Undefined from OnPayload preserves its input; nil replaces it with JSON null.
// Model, Values and callback values are live. Callers synchronize edits while a
// request is active. ErrorStack supplies host stack metadata; Go cannot produce
// the JavaScript source locations of the reference runtime.
type PiMessagesStreamOptions struct {
	// The serialized host callback may replace the entire JSON event.
	transformEvent        func(context.Context, any, *Object) (any, error)
	Values                *Object
	Client                *http.Client
	OnPayload             func(context.Context, any, *Object) (any, error)
	OnResponse            func(context.Context, CompletionsResponse, *Object) error
	OnProviderStreamEvent func(context.Context, any, *Object) error
	Now                   func() float64
	ErrorStack            func(error) string
}

type PiMessagesResponseError struct {
	Message           string
	Code              *string
	DiagnosticDetails *Object
	Status            int
	Headers           http.Header
}

func (e *PiMessagesResponseError) Error() string { return e.Message }

// StreamPiMessages implements the gateway protocol entirely in Go. A terminal
// provider event wins even if the request context has since been canceled.
func StreamPiMessages(ctx context.Context, model *Object, transcript TranscriptContext, options PiMessagesStreamOptions) *AssistantMessageEventStream {
	if ctx == nil {
		ctx = context.Background()
	}
	stream := NewAssistantMessageEventStreamFor(ctx)
	converter := newPiMessagesEventConverter(model, options.Now)
	adapter := newPiMessagesTypedAdapter()
	// Transcript serialization occurs before returning to the caller, matching
	// other native providers' ownership of the supplied native slice.
	raw, prepareErr := transcript.MarshalJSON()
	input, decodeErr := jsonjs.DecodeValue(raw)
	if prepareErr == nil {
		prepareErr = decodeErr
	}
	go func() {
		defer stream.End()
		producePiMessages(ctx, model, input, options, converter, prepareErr, func(event *Object) error {
			var err error
			stream.Synchronize(func() {
				var typed AssistantMessageEvent
				typed, err = adapter.event(event)
				if err == nil {
					stream.Push(typed)
				}
			})
			return err
		})
	}()
	return stream
}

// Both upstream entry points use the same transport and controls.
func StreamSimplePiMessages(ctx context.Context, model *Object, transcript TranscriptContext, options PiMessagesStreamOptions) *AssistantMessageEventStream {
	return StreamPiMessages(ctx, model, transcript, options)
}

// PiMessagesAPI is the concrete provider factory used by Radius and custom
// providers. Request callbacks use the same signatures as the option fields.
func PiMessagesAPI(defaults PiMessagesStreamOptions) *ProviderStreams {
	call := func(ctx context.Context, model any, transcript TranscriptContext, values *Object) (*ProviderEventSource, error) {
		object, ok := model.(*Object)
		if !ok {
			return nil, errors.New("pi-messages requires a model object")
		}
		options := defaults
		options.Values = values
		if callback, ok := catalogProperty(values, "onPayload").(func(context.Context, any, *Object) (any, error)); ok {
			options.OnPayload = callback
		}
		if callback, ok := catalogProperty(values, "onResponse").(func(context.Context, CompletionsResponse, *Object) error); ok {
			options.OnResponse = callback
		}
		if callback, ok := catalogProperty(values, "onProviderStreamEvent").(func(context.Context, any, *Object) error); ok {
			options.OnProviderStreamEvent = callback
		}
		return SourceFromAssistantStream(StreamPiMessages(ctx, object, transcript, options)), nil
	}
	implementation := &ProviderStreams{Stream: call, StreamSimple: call}
	var now func() int64
	if defaults.Now != nil {
		now = func() int64 { return int64(defaults.Now()) }
	}
	return LazyAPI(func(context.Context) (*ProviderStreams, error) { return implementation, nil }, LazyAPICapabilities{}, now)
}

func producePiMessages(ctx context.Context, model *Object, transcript any, options PiMessagesStreamOptions, converter *piMessagesEventConverter, prepareErr error, yield func(*Object) error) {
	callbackFailure := false
	provider, _ := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
	recordNativeFailure(ctx, provider, nil, false)
	_, err := invokeAuth(func() (any, error) {
		if prepareErr != nil {
			return nil, prepareErr
		}
		return nil, runPiMessagesHTTP(ctx, model, transcript, options, converter, &callbackFailure, yield)
	})
	if err == nil {
		return
	}
	recordNativeFailure(ctx, provider, err, callbackFailure)
	failure := newPiMessagesEventConverter(model, converter.now).partial
	reason := "error"
	if ctx.Err() != nil {
		reason = "aborted"
	}
	failure.Set("stopReason", reason)
	failure.Set("errorMessage", err.Error())
	var responseError *PiMessagesResponseError
	if reason != "aborted" && errors.As(err, &responseError) {
		info := NewObject(Property{Name: "name", Value: "PiMessagesResponseError"}, Property{Name: "message", Value: err.Error()})
		if options.ErrorStack != nil {
			info.Set("stack", options.ErrorStack(err))
		}
		if responseError.Code != nil {
			info.Set("code", *responseError.Code)
		}
		failure.Set("diagnostics", NewArray(NewObject(Property{Name: "type", Value: "pi_messages_response_failure"}, Property{Name: "timestamp", Value: converter.now()}, Property{Name: "error", Value: info}, Property{Name: "details", Value: responseError.DiagnosticDetails})))
	}
	// Native adaptation accepts every JSON-shaped protocol field, including
	// malformed typed values. Its only failures are serialization failures.
	_ = yield(NewObject(Property{Name: "type", Value: "error"}, Property{Name: "reason", Value: reason}, Property{Name: "error", Value: failure}))
}

func runPiMessagesHTTP(ctx context.Context, model *Object, transcript any, options PiMessagesStreamOptions, converter *piMessagesEventConverter, callbackFailure *bool, yield func(*Object) error) error {
	controls := options.Values
	key := catalogProperty(controls, "apiKey")
	provider, err := catalogKey(catalogProperty(model, "provider"), make(map[*Array]bool))
	if err != nil {
		return err
	}
	if !catalogEntryTruthy(key) {
		return errors.New("No API key provided for provider \"" + provider + "\"")
	}
	base := catalogProperty(model, "baseUrl")
	baseURL, ok := base.(string)
	if !ok {
		if jsonjs.IsNullish(base) {
			return jsonjs.PropertyReadError(!jsonjs.IsUndefined(base), "model.baseUrl.replace")
		}
		return errors.New("model.baseUrl.replace is not a function. (In 'model.baseUrl.replace(/\\/+$/u, \"\")', 'model.baseUrl.replace' is undefined)")
	}
	endpoint, err := whatwg.Parse(string(jsonjs.StringCodePoints(strings.TrimRight(baseURL, "/") + "/messages")))
	if err != nil {
		return errors.New("\"" + strings.TrimRight(baseURL, "/") + "/messages\" cannot be parsed as a URL.")
	}
	if catalogEntryTruthy(catalogProperty(controls, "debug")) {
		fields := []Property{}
		found := false
		for _, pair := range strings.Split(strings.TrimPrefix(endpoint.Search(), "?"), "&") {
			if pair == "" {
				continue
			}
			name, value, _ := strings.Cut(pair, "=")
			name, value = urlQueryComponent(name), urlQueryComponent(value)
			if name == "debug" {
				if found {
					continue
				}
				value = "1"
				found = true
			}
			fields = append(fields, Property{Name: name, Value: value})
		}
		if !found {
			fields = append(fields, Property{Name: "debug", Value: "1"})
		}
		query, err := oauthFormEncode(fields...)
		if err != nil {
			return err
		}
		endpoint.SetSearch(query)
	}
	cache := catalogProperty(controls, "cacheRetention")
	if !catalogEntryTruthy(cache) {
		env := catalogProperty(catalogProperty(controls, "env"), "PI_CACHE_RETENTION")
		if !catalogEntryTruthy(env) {
			env = os.Getenv("PI_CACHE_RETENTION")
		}
		cache = Undefined
		if env == "long" {
			cache = "long"
		}
	}
	projected := NewObject()
	for _, name := range []string{"temperature", "maxTokens", "reasoning", "cacheRetention", "sessionId", "toolChoice"} {
		value := catalogProperty(controls, name)
		if name == "cacheRetention" {
			value = cache
		}
		projected.Set(name, value)
	}
	var payload any = NewObject(Property{Name: "model", Value: catalogProperty(model, "id")}, Property{Name: "context", Value: transcript}, Property{Name: "options", Value: projected})
	if options.OnPayload != nil {
		*callbackFailure = true
		replacement, err := options.OnPayload(ctx, payload, model)
		if err != nil {
			return err
		}
		*callbackFailure = false
		if !jsonjs.IsUndefined(replacement) {
			payload = replacement
		}
	}
	bearer, err := catalogKey(key, make(map[*Array]bool))
	if err != nil {
		return err
	}
	headers, err := piMessagesHeaders(bearer, catalogProperty(controls, "headers"))
	if err != nil {
		return err
	}
	body, err := jsonjs.StringifyValue(payload)
	if err != nil {
		return err
	}
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	response, err := oauthFetch(ctx, client, endpoint.String(), http.MethodPost, headers, string(body))
	if err != nil {
		return err
	}
	if response.Body != nil {
		defer response.Body.Close()
	}
	if options.OnResponse != nil {
		values := make(map[string]string, len(response.Header))
		for name, entries := range response.Header {
			values[strings.ToLower(name)] = strings.Join(entries, ", ")
		}
		*callbackFailure = true
		err = options.OnResponse(ctx, CompletionsResponse{Status: response.StatusCode, Headers: values}, model)
		if err != nil {
			return err
		}
		*callbackFailure = false
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		var raw []byte
		if response.Body != nil {
			raw, err = io.ReadAll(response.Body)
		}
		if err != nil {
			return err
		}
		raw, _ = unicode.UTF8BOM.NewDecoder().Bytes(raw)
		return piMessagesResponseFailure(model, endpoint.String(), response, string(raw), converter.now)
	}
	if response.Body == nil || response.StatusCode == 204 || response.StatusCode == 205 {
		return errors.New(provider + " response has no body")
	}
	terminal := false
	err = readPiMessagesEvents(response.Body, func(event any) (bool, error) {
		if options.OnProviderStreamEvent != nil {
			*callbackFailure = true
			err := options.OnProviderStreamEvent(ctx, event, model)
			if err != nil {
				return false, err
			}
			*callbackFailure = false
		}
		if options.transformEvent != nil {
			*callbackFailure = true
			var err error
			event, err = options.transformEvent(ctx, event, model)
			if err != nil {
				return false, err
			}
			*callbackFailure = false
		}
		converted, err := converter.convert(event)
		if err != nil {
			return false, err
		}
		if err = yield(converted); err != nil {
			return false, err
		}
		terminal = converted.Get("type") == "done" || converted.Get("type") == "error"
		return !terminal, nil
	})
	if err != nil {
		return err
	}
	if !terminal {
		return errors.New(provider + " stream ended without a terminal event")
	}
	return nil
}

func piMessagesHeaders(key string, input any) (http.Header, error) {
	// Source deduplication is case-insensitive but preserves the last spelling;
	// object spread then overrides defaults only when that spelling is identical.
	entries := []Property{}
	for _, entry := range authSpread(input).Entries() {
		for i := 0; i < len(entries); i++ {
			if authHeaderName(entries[i].Name) == authHeaderName(entry.Name) {
				entries = append(entries[:i], entries[i+1:]...)
				break
			}
		}
		if entry.Value != nil && !jsonjs.IsNull(entry.Value) {
			entries = append(entries, entry)
		}
	}
	fields := NewObject(Property{Name: "authorization", Value: "Bearer " + key}, Property{Name: "accept", Value: "text/event-stream"}, Property{Name: "content-type", Value: "application/json"})
	for _, entry := range entries {
		fields.Set(entry.Name, entry.Value)
	}
	headers := make(http.Header)
	for _, entry := range fields.Entries() {
		value, err := catalogKey(entry.Value, make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		name := strings.ToLower(entry.Name)
		value = strings.Trim(value, " \t\r\n")
		if previous := headers.Get(name); previous != "" {
			value = previous + ", " + value
		}
		headers.Set(name, value)
	}
	return headers, nil
}

func piMessagesResponseFailure(model *Object, endpoint string, response *http.Response, body string, now func() float64) *PiMessagesResponseError {
	statusText := strings.TrimPrefix(response.Status, strconv.Itoa(response.StatusCode))
	statusText = strings.TrimPrefix(statusText, " ")
	if response.Status == "" {
		statusText = http.StatusText(response.StatusCode)
	}
	var errorBody *Object
	if parsed, err := jsonjs.DecodeValue([]byte(body)); err == nil {
		errorBody, _ = catalogProperty(parsed, "error").(*Object)
	}
	message := body
	var code *string
	if errorBody != nil {
		if value, ok := errorBody.Get("message").(string); ok {
			message = value
		}
		if value, ok := errorBody.Get("code").(string); ok {
			code = &value
		}
	}
	message = fmt.Sprintf("%d %s: %s", response.StatusCode, statusText, message)
	if code != nil && *code != "" {
		message += " (" + *code + ")"
	}
	details := NewObject(Property{Name: "version", Value: 1}, Property{Name: "provider", Value: catalogProperty(model, "provider")}, Property{Name: "model", Value: catalogProperty(model, "id")}, Property{Name: "url", Value: endpoint}, Property{Name: "status", Value: response.StatusCode}, Property{Name: "statusText", Value: statusText})
	if errorBody != nil {
		details.Set("error", errorBody)
	} else {
		units := []uint16{}
		for _, point := range jsonjs.StringCodePoints(body) {
			if point > 0xffff {
				units = append(units, uint16(0xd800+(point-0x10000)>>10), uint16(0xdc00+(point-0x10000)&0x3ff))
			} else {
				units = append(units, uint16(point))
			}
		}
		if len(units) > 8192 {
			var text strings.Builder
			for _, unit := range units[:8192] {
				text.WriteString(catalogCodeUnit(rune(unit)))
			}
			body = text.String() + "…"
		}
		details.Set("body", body)
	}
	details.Set("timestampMs", now())
	return &PiMessagesResponseError{Message: message, Code: code, DiagnosticDetails: details, Status: response.StatusCode, Headers: response.Header.Clone()}
}
