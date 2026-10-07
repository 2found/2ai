package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// OpenAICompletionsStreamOptions carries serializable Pi provider controls in
// Options and concrete Go dependencies/callbacks separately. A nil OnPayload
// result preserves the payload. OnProviderStreamEvent may replace the decoded
// JSON through its pointer before accumulation, just as Pi permits mutation.
// Callbacks run serially on the producer and must honor their context.
type OpenAICompletionsStreamOptions struct {
	Options               json.RawMessage
	Client                *http.Client
	OnPayload             func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error)
	OnResponse            func(context.Context, CompletionsResponse, json.RawMessage) error
	OnProviderStreamEvent func(context.Context, *json.RawMessage, json.RawMessage) error
	Now                   func() int64
}

type CompletionsResponse struct {
	Status  int               `json:"status"`
	Headers map[string]string `json:"headers"`
}

// StreamOpenAICompletions executes the native provider against its HTTP endpoint.
// It returns the live stream immediately; preparation, callback and HTTP errors
// settle it with an error event. It does not invoke a JavaScript runtime or the
// legacy agentcore transcript adapter.
func StreamOpenAICompletions(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStreamFor(ctx)
	// Freeze caller-owned JSON/transcript before launching the producer. Go
	// callers may reuse their inputs as soon as this function returns.
	rawModel = append(json.RawMessage(nil), rawModel...)
	options.Options = append(json.RawMessage(nil), options.Options...)
	var model completionsModel
	prepareErr := json.Unmarshal(rawModel, &model)
	encoded, err := json.Marshal(transcript)
	if prepareErr == nil {
		prepareErr = err
	}
	var frozen Context
	if err = json.Unmarshal(encoded, &frozen); prepareErr == nil {
		prepareErr = err
	}
	transcript = NormalizeContext(frozen)
	compat, err := ResolveOpenAICompletionsCompat(rawModel)
	if prepareErr == nil {
		prepareErr = err
	}
	transcript = ResolveTranscript(transcript, compat.SupportsMidConvoSystemMessages)
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	options, callbackFailure := nativeFailureCallbacks(options)
	recordNativeFailure(ctx, model.Provider, nil, false)
	acc := newCompletionsAccumulator(model, compat, nil, stream, now)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				recordNativeFailure(ctx, model.Provider, fmt.Errorf("%v", recovered), true)
				acc.fail(fmt.Sprint(recovered), ctx.Err() != nil)
			}
		}()
		err := prepareErr
		if err == nil {
			err = runCompletionsHTTP(ctx, acc, rawModel, transcript, options)
		}
		if err != nil {
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			acc.fail(formatCompletionsError(err), ctx.Err() != nil)
		}
	}()
	return stream
}

func completionsAPIKey(provider string, controls map[string]json.RawMessage) (string, error) {
	key := samplingString(controls["apiKey"])
	if key == "" {
		headers, _ := samplingObject(controls["headers"])
		for name, value := range headers {
			if (strings.EqualFold(name, "authorization") || strings.EqualFold(name, "cf-aig-authorization")) && strings.TrimFunc(samplingString(value), jsWhitespace) != "" {
				key = "unused"
				break
			}
		}
		if key == "" {
			return "", fmt.Errorf("No API key for provider: %s", provider)
		}
	}
	return key, nil
}

func runCompletionsHTTP(ctx context.Context, acc *completionsAccumulator, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions) error {
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return err
		}
	}
	key, err := completionsAPIKey(acc.model.Provider, controls)
	if err != nil {
		return err
	}
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(transcript.Messages()), acc.compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return err
	}
	acc.grammar = grammar
	// createClient captures headers before onPayload; SDK header validation
	// happens afterwards, when building the request.
	headers, headerErr := completionsHeaders(rawModel, transcript, controls, acc.compat, key)
	params, err := BuildOpenAICompletionsParams(rawModel, transcript, options.Options)
	if err != nil {
		return err
	}
	return runOpenAIHTTP(ctx, rawModel, options, controls, headers, headerErr, params, openAIHTTPStream{
		path: "/chat/completions", start: acc.start, chunk: acc.chunk, finish: acc.finish,
		nonIterableError: "undefined is not a function (near '...chunk of openaiStream...')",
	})
}

// Transport is shared by the native OpenAI-compatible APIs. Provider-specific accumulation
// stays in concrete callbacks, with no adapter to the legacy agent runtime.
type openAIHTTPStream struct {
	nonstreamObjectRequired bool
	path                    string
	endpoint                string
	manualRedirect          bool
	start                   func()
	chunk                   func(json.RawMessage) error
	finish                  func(context.Context)
	nonIterableError        string
}

func runOpenAIHTTP(ctx context.Context, rawModel json.RawMessage, options OpenAICompletionsStreamOptions, controls map[string]json.RawMessage, headers http.Header, headerErr error, params json.RawMessage, execution openAIHTTPStream) error {
	if options.OnPayload != nil {
		next, err := options.OnPayload(ctx, params, rawModel)
		if err != nil {
			return err
		}
		if next != nil {
			params = next
		}
	}
	if !json.Valid(params) {
		return errors.New("onPayload returned invalid JSON")
	}
	paramsFields, _ := samplingObject(params)
	if bytes.Equal(bytes.TrimSpace(params), []byte("null")) {
		return errors.New("Cannot read properties of null (reading 'stream')")
	}
	timeout := 600000.0
	if value, exists := controls["timeoutMs"]; exists {
		var parsed float64
		if json.Unmarshal(value, &parsed) != nil || !samplingNonNull(value) || math.Trunc(parsed) != parsed {
			return errors.New("timeout must be an integer")
		}
		if parsed < 0 {
			return errors.New("timeout must be a positive integer")
		}
		timeout = parsed
	}
	if headerErr != nil {
		return headerErr
	}
	if samplingTruthy(params) {
		headers.Set("Content-Type", "application/json")
	}
	// Each attempt is a new SDK request, so the retry count header stays zero.
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	model, _ := samplingObject(rawModel)
	endpoint := samplingString(model["baseUrl"])
	if endpoint == "" {
		endpoint = "https://api.openai.com/v1"
	}
	endpoint = strings.TrimSuffix(endpoint, "/") + execution.path
	if execution.endpoint != "" {
		endpoint = execution.endpoint
	}
	if execution.manualRedirect {
		copy := *client
		copy.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		client = &copy
	}
	retries := 0.0
	_ = json.Unmarshal(controls["maxRetries"], &retries)
	var maxDelay *float64
	_ = json.Unmarshal(controls["maxRetryDelayMs"], &maxDelay)
	body := []byte(params)
	if !samplingTruthy(params) {
		body = nil
	}
	response, err := retryCompletionsRequest(ctx, func() (*http.Response, error) {
		requestCtx, cancel := context.WithCancel(ctx)
		request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			cancel()
			return nil, err
		}
		request.Header = headers.Clone()
		delay := timeout
		if delay > 2147483647 {
			delay = 1
		}
		timer := time.AfterFunc(time.Duration(delay)*time.Millisecond, cancel)
		response, err := client.Do(request)
		timer.Stop()
		if err != nil {
			message := "Connection error."
			if requestCtx.Err() != nil || isTimeout(err) {
				message = "Request timed out."
			}
			cancel()
			return nil, &completionsRequestError{message: message}
		}
		response.Body = &completionsHTTPBody{ReadCloser: response.Body, cancel: cancel, ctx: requestCtx}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			data, readErr := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if readErr != nil {
				data = []byte(readErr.Error())
			}
			fallback := string(data)
			var errorBody json.RawMessage
			if json.Valid(data) && samplingTruthy(data) {
				fields, _ := samplingObject(data)
				errorBody = fields["error"]
				if !samplingNonNull(errorBody) && (bytes.TrimSpace(data)[0] == '{' || bytes.TrimSpace(data)[0] == '[') {
					errorBody = data
				}
				fallback = ""
			}
			return nil, newCompletionsRequestError(response.StatusCode, errorBody, fallback, response.Header)
		}
		return response, nil
	}, retries, maxDelay, defaultCompletionsRetryTiming())
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var nonstream json.RawMessage
	if !samplingTruthy(paramsFields["stream"]) {
		body, err := io.ReadAll(response.Body)
		if err != nil {
			return err
		}
		mediaType := strings.ToLower(strings.TrimSpace(strings.Split(response.Header.Get("Content-Type"), ";")[0]))
		if strings.Contains(mediaType, "application/json") || strings.HasSuffix(mediaType, "+json") {
			if len(body) > 0 && !json.Valid(body) {
				return errors.New("Invalid JSON response")
			}
			nonstream = body
		} else {
			nonstream = json.RawMessage(marshalSamplingString(string(body)))
		}
	}

	if execution.nonstreamObjectRequired && !samplingTruthy(paramsFields["stream"]) {
		trimmed := bytes.TrimSpace(nonstream)
		if len(trimmed) == 0 || (trimmed[0] != '{' && trimmed[0] != '[') {
			return errors.New("rsp is not an Object. (evaluating '\"object\" in rsp')")
		}
	}
	if options.OnResponse != nil {
		info := CompletionsResponse{Status: response.StatusCode, Headers: map[string]string{}}
		for name, values := range response.Header {
			info.Headers[strings.ToLower(name)] = strings.Join(values, ", ")
		}
		if err := options.OnResponse(ctx, info, rawModel); err != nil {
			return err
		}
	}
	execution.start()
	if response.StatusCode == http.StatusNoContent && samplingTruthy(paramsFields["stream"]) {
		return errors.New("Attempted to iterate over a response with no body")
	}
	consume := func(raw json.RawMessage) error {
		if options.OnProviderStreamEvent != nil {
			if err := options.OnProviderStreamEvent(ctx, &raw, rawModel); err != nil {
				return err
			}
		}
		return execution.chunk(raw)
	}
	if samplingTruthy(paramsFields["stream"]) {
		readCtx := ctx
		if body, ok := response.Body.(*completionsHTTPBody); ok {
			readCtx = body.ctx
		}
		err = readCompletionsSSE(readCtx, response.Body, response.Header, consume)
	} else {
		var chunks []json.RawMessage
		if len(nonstream) > 0 && nonstream[0] == '"' {
			for _, r := range samplingString(nonstream) {
				chunks = append(chunks, json.RawMessage(marshalSamplingString(string(r))))
			}
		} else if json.Unmarshal(nonstream, &chunks) != nil || chunks == nil {
			// Diagnostic from the pinned Bun oracle for the provider wrapper's
			// for-await applied to a non-iterable response.
			return errors.New(execution.nonIterableError)
		}
		for _, chunk := range chunks {
			if err = consume(chunk); err != nil {
				break
			}
		}
	}
	if err != nil {
		return err
	}
	execution.finish(ctx)
	return nil
}

type completionsHTTPBody struct {
	io.ReadCloser
	cancel context.CancelFunc
	ctx    context.Context
}

func (b *completionsHTTPBody) Close() error { defer b.cancel(); return b.ReadCloser.Close() }

var completionsHeaderName = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

func completionsHeaders(rawModel json.RawMessage, transcript TranscriptContext, options map[string]json.RawMessage, compat OpenAICompletionsCompat, key string) (http.Header, error) {
	return openAIHeaders(rawModel, transcript, options, compat.SessionAffinityFormat, compat.SendSessionAffinityHeaders, true, key)
}

func openAIHeaders(rawModel json.RawMessage, transcript TranscriptContext, options map[string]json.RawMessage, affinityFormat string, sendAffinity, includeSessionAffinity bool, key string) (http.Header, error) {
	return openAIHeadersFor(rawModel, transcript, options, affinityFormat, sendAffinity, includeSessionAffinity, key, false)
}

func openAIHeadersFor(rawModel json.RawMessage, transcript TranscriptContext, options map[string]json.RawMessage, affinityFormat string, sendAffinity, includeSessionAffinity bool, key string, azure bool) (http.Header, error) {
	model, _ := samplingObject(rawModel)
	// Object.assign merges by exact spelling first. Converting to HTTP headers
	// only afterwards preserves the source's case-collision precedence.
	fields := map[string]json.RawMessage{}
	keys := []string{}
	set := func(name string, value json.RawMessage) {
		if _, exists := fields[name]; !exists {
			keys = append(keys, name)
		}
		fields[name] = value
	}
	text := func(name, value string) { set(name, json.RawMessage(marshalSamplingString(value))) }
	merge := func(raw json.RawMessage) {
		values, _ := samplingObject(raw)
		for _, name := range samplingObjectKeys(raw) {
			set(name, values[name])
		}
	}
	text("User-Agent", completionsUserAgent())
	merge(model["headers"])
	if !azure && samplingString(model["provider"]) == "github-copilot" {
		messages := transcript.Messages()
		initiator := "user"
		if len(messages) > 0 && messages[len(messages)-1].Role != "user" {
			initiator = "agent"
		}
		text("X-Initiator", initiator)
		text("Openai-Intent", "conversation-edits")
		for _, message := range messages {
			if message.Role == "user" || message.Role == "toolResult" {
				for _, block := range message.Content.Blocks.Values() {
					if block.Type == "image" {
						text("Copilot-Vision-Request", "true")
					}
				}
			}
		}
	}
	if session := samplingString(options["sessionId"]); session != "" && samplingString(options["cacheRetention"]) != "none" && sendAffinity {
		if affinityFormat == "openrouter" {
			text("x-session-id", session)
		} else {
			if affinityFormat == "openai" {
				text("session_id", session)
			}
			text("x-client-request-id", session)
			if includeSessionAffinity {
				text("x-session-affinity", session)
			}
		}
	}
	merge(options["headers"])
	headers := http.Header{"Accept": []string{"application/json"}, "X-Stainless-Retry-Count": []string{"0"}}
	if azure {
		headers.Set("Api-Key", strings.Trim(key, " \t\r\n"))
	} else {
		headers.Set("Authorization", strings.Trim("Bearer "+key, " \t\r\n"))
	}
	if value := options["timeoutMs"]; samplingTruthy(value) {
		var ms float64
		_ = json.Unmarshal(value, &ms)
		headers.Set("X-Stainless-Timeout", strconv.FormatFloat(math.Trunc(ms/1000), 'f', -1, 64))
	}
	for _, item := range []struct{ env, header string }{{"OPENAI_ORG_ID", "OpenAI-Organization"}, {"OPENAI_PROJECT_ID", "OpenAI-Project"}} {
		if value := strings.TrimFunc(os.Getenv(item.env), jsWhitespace); value != "" {
			headers.Set(item.header, value)
		}
	}
	nulls := map[string]bool{}
	for _, name := range keys {
		if !completionsHeaderName.MatchString(name) {
			return nil, fmt.Errorf("Header name must be a valid HTTP token [\"%s\"]", name)
		}
		value := fields[name]
		if !samplingNonNull(value) {
			headers.Del(name)
			nulls[strings.ToLower(name)] = true
		} else {
			headers.Set(name, strings.Trim(samplingString(value), " \t\r\n"))
			delete(nulls, strings.ToLower(name))
		}
	}
	if headers.Get("Authorization") == "" && headers.Get("Api-Key") == "" && !nulls["authorization"] && !nulls["api-key"] {
		return nil, errors.New("Could not resolve authentication method. Expected either apiKey or adminAPIKey to be set. Or for one of the \"Authorization\" or \"api-key\" headers to be explicitly omitted")
	}
	// An explicitly omitted UA must not be replaced by net/http's Go default.
	if _, exists := headers["User-Agent"]; !exists {
		headers["User-Agent"] = nil
	}
	return headers, nil
}
