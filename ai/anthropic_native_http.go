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
	"strconv"
	"strings"
	"time"
	"unicode/utf16"
)

// AnthropicStreamOptions separates concrete Go transport/callback dependencies
// from serializable provider controls. Client replaces fetch, not SDK auth.
type AnthropicStreamOptions = OpenAICompletionsStreamOptions

func anthropicSpreadObject(raw json.RawMessage) map[string]json.RawMessage {
	raw = bytes.TrimSpace(raw)
	if fields, ok := samplingObject(raw); ok {
		return fields
	}
	fields := map[string]json.RawMessage{}
	var array []json.RawMessage
	if json.Unmarshal(raw, &array) == nil {
		for i, value := range array {
			fields[strconv.Itoa(i)] = value
		}
		return fields
	}
	if len(raw) > 0 && raw[0] == '"' {
		for i, unit := range utf16.Encode([]rune(samplingString(raw))) {
			fields[strconv.Itoa(i)] = json.RawMessage(fmt.Sprintf(`"\u%04x"`, unit))
		}
	}
	return fields
}

// StreamAnthropic executes the native Messages provider without a worker.
func StreamAnthropic(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions) *AssistantMessageEventStream {
	return streamAnthropic(ctx, rawModel, transcript, options, nil, nil)
}

func streamAnthropic(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions, numeric *ThinkingTokenBudget, client *AnthropicMessageClient) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStreamFor(ctx)
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
	compat, err := ResolveAnthropicCompat(rawModel)
	if prepareErr == nil {
		prepareErr = err
	}
	transcript = ResolveTranscript(NormalizeContext(frozen), compat.SupportsMidConvoSystemMessages)
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err = json.Unmarshal(options.Options, &controls); prepareErr == nil {
			prepareErr = err
		}
	}
	now := options.Now
	if now == nil {
		now = func() int64 { return time.Now().UnixMilli() }
	}
	key := samplingString(controls["apiKey"])
	oauth := client == nil && model.Provider != "github-copilot" && (model.Provider == VendorClaudeCode || strings.Contains(key, "sk-ant-oat"))
	options, callbackFailure := nativeFailureCallbacks(options)
	recordNativeFailure(ctx, model.Provider, nil, false)
	acc := newAnthropicAccumulator(model, oauth, GetCurrentTools(transcript.Messages()), controls, stream, now)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				recordNativeFailure(ctx, model.Provider, fmt.Errorf("%v", recovered), true)
				acc.fail(fmt.Sprint(recovered), ctx.Err() != nil)
			}
		}()
		err := prepareErr
		if err == nil {
			if client != nil {
				err = runAnthropicClient(ctx, acc, rawModel, transcript, controls, options, client)
			} else {
				err = runAnthropicHTTP(ctx, acc, rawModel, transcript, controls, options, numeric)
			}
		}
		if err != nil {
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			acc.fail(err.Error(), ctx.Err() != nil)
		}
	}()
	return stream
}

func anthropicRequestAuth(provider string, controls map[string]json.RawMessage) error {
	if samplingTruthy(controls["apiKey"]) {
		return nil
	}
	headers, _ := samplingObject(controls["headers"])
	for name, value := range headers {
		if (strings.EqualFold(name, "authorization") || strings.EqualFold(name, "x-api-key") || strings.EqualFold(name, "cf-aig-authorization")) && strings.TrimFunc(samplingString(value), jsWhitespace) != "" {
			return nil
		}
	}
	return fmt.Errorf("No API key for provider: %s", provider)
}

func anthropicHeaders(rawModel json.RawMessage, transcript TranscriptContext, controls map[string]json.RawMessage, oauth bool, timeout float64, federated bool) (http.Header, error) {
	model, _ := samplingObject(rawModel)
	fields := map[string]json.RawMessage{}
	keys := []string{}
	set := func(name string, value json.RawMessage) {
		if _, ok := fields[name]; !ok {
			keys = append(keys, name)
		}
		fields[name] = value
	}
	put := func(name, value string) { set(name, json.RawMessage(marshalSamplingString(value))) }
	merge := func(raw json.RawMessage) {
		values, _ := samplingObject(raw)
		for _, name := range samplingObjectKeys(raw) {
			set(name, values[name])
		}
	}
	for _, line := range strings.Split(os.Getenv("ANTHROPIC_CUSTOM_HEADERS"), "\n") {
		if name, value, ok := strings.Cut(line, ":"); ok {
			put(strings.TrimFunc(name, jsWhitespace), strings.TrimFunc(value, jsWhitespace))
		}
	}
	put("User-Agent", completionsUserAgent())
	put("accept", "application/json")
	put("anthropic-dangerous-direct-browser-access", "true")
	copilot := samplingString(model["provider"]) == "github-copilot"
	if oauth {
		put("user-agent", "claude-cli/2.1.280")
		put("x-app", "cli")
	}
	if !oauth && !copilot {
		compat, _ := ResolveAnthropicCompat(rawModel)
		if session := samplingString(controls["sessionId"]); session != "" && samplingString(controls["cacheRetention"]) != "none" && compat.SendSessionAffinityHeaders {
			name := "x-session-affinity"
			if compat.SessionAffinityFormat != nil && *compat.SessionAffinityFormat == "openrouter" {
				name = "x-session-id"
			}
			put(name, session)
		}
	}
	merge(model["headers"])
	if copilot {
		messages := transcript.Messages()
		initiator := "user"
		if len(messages) > 0 && messages[len(messages)-1].Role != "user" {
			initiator = "agent"
		}
		put("X-Initiator", initiator)
		put("Openai-Intent", "conversation-edits")
		for _, message := range messages {
			if message.Role == "user" || message.Role == "toolResult" {
				for _, block := range message.Content.Blocks.Values() {
					if block.Type == "image" {
						put("Copilot-Vision-Request", "true")
					}
				}
			}
		}
	}
	merge(controls["headers"])
	headers := http.Header{}
	headers.Set("Accept", "application/json")
	headers.Set("Anthropic-Version", "2023-06-01")
	headers.Set("X-Stainless-Retry-Count", "0")
	headers.Set("X-Stainless-Timeout", strconv.FormatFloat(math.Trunc(timeout/1000), 'f', -1, 64))
	if key := samplingString(controls["apiKey"]); key != "" {
		if oauth || copilot {
			headers.Set("Authorization", strings.Trim("Bearer "+key, " \t\r\n"))
		} else {
			headers.Set("X-Api-Key", strings.Trim(key, " \t\r\n"))
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
	if !federated && headers.Get("X-Api-Key") == "" && headers.Get("Authorization") == "" && !nulls["x-api-key"] && !nulls["authorization"] {
		return nil, errors.New(`Could not resolve authentication method. Expected one of apiKey, authToken, credentials, config, or profile to be set. Or for one of the "X-Api-Key" or "Authorization" headers to be explicitly omitted`)
	}
	if _, ok := headers["User-Agent"]; !ok {
		headers["User-Agent"] = nil
	}
	return headers, nil
}

func anthropicHTTPTimeout(controls map[string]json.RawMessage) (float64, error) {
	timeout := 600000.0
	if value, exists := controls["timeoutMs"]; exists {
		var parsed float64
		if json.Unmarshal(value, &parsed) != nil || !samplingNonNull(value) || math.Trunc(parsed) != parsed {
			return 0, errors.New("timeout must be an integer")
		}
		if parsed < 0 {
			return 0, errors.New("timeout must be a positive integer")
		}
		timeout = parsed
	}
	return timeout, nil
}

func runAnthropicHTTP(ctx context.Context, acc *anthropicAccumulator, rawModel json.RawMessage, transcript TranscriptContext, controls map[string]json.RawMessage, options AnthropicStreamOptions, numeric *ThinkingTokenBudget) error {
	federation := anthropicFederation(acc.model.Provider, controls)
	if err := anthropicRequestAuth(acc.model.Provider, controls); err != nil && federation == nil {
		return err
	}
	endpoint := acc.model.BaseURL
	if endpoint == "" {
		endpoint = "https://api.anthropic.com"
	}
	var tokenCache *anthropicTokenCache
	if federation != nil {
		tokenCache = anthropicFederationCache(*federation, endpoint, options.Client, acc.now)
	}
	timeout, timeoutErr := anthropicHTTPTimeout(controls)
	headers, headerErr := anthropicHeaders(rawModel, transcript, controls, acc.oauth, timeout, federation != nil)
	overrideBearer := false
	modelFields, _ := samplingObject(rawModel)
	for _, source := range []json.RawMessage{modelFields["headers"], controls["headers"]} {
		for _, key := range samplingObjectKeys(source) {
			if strings.EqualFold(key, "authorization") {
				overrideBearer = true
			}
		}
	}
	for _, line := range strings.Split(os.Getenv("ANTHROPIC_CUSTOM_HEADERS"), "\n") {
		if key, _, ok := strings.Cut(line, ":"); ok && strings.EqualFold(strings.TrimFunc(key, jsWhitespace), "authorization") {
			overrideBearer = true
		}
	}
	params, err := prepareAnthropicPayload(ctx, acc, rawModel, transcript, options, numeric)
	if err != nil {
		return err
	}
	fields, _ := samplingObject(params)
	if samplingTruthy(fields["output_format"]) {
		config := anthropicSpreadObject(fields["output_config"])
		if samplingTruthy(config["format"]) {
			return errors.New("Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).")
		}
		config["format"] = fields["output_format"]
		delete(fields, "output_format")
		fields["output_config"], _ = json.Marshal(config)
	}
	if timeoutErr != nil {
		return timeoutErr
	}
	if headerErr != nil {
		return headerErr
	}
	for _, field := range []struct{ key, header string }{{"betas", "anthropic-beta"}, {"user_profile_id", "anthropic-user-profile-id"}, {"workspace_id", "anthropic-workspace-id"}} {
		if value := fields[field.key]; samplingNonNull(value) {
			headers.Set(field.header, completionsErrorString(value))
		}
		delete(fields, field.key)
	}
	headers.Set("Content-Type", "application/json")
	body, err := json.Marshal(fields)
	if err != nil {
		return err
	}
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	endpoint = strings.TrimSuffix(endpoint, "/") + "/v1/messages?beta=true"
	retries := 0.0
	_ = json.Unmarshal(controls["maxRetries"], &retries)
	var maxDelay *float64
	_ = json.Unmarshal(controls["maxRetryDelayMs"], &maxDelay)
	response, err := retryCompletionsRequest(ctx, func() (*http.Response, error) {
		token := ""
		if tokenCache != nil {
			var err error
			token, err = tokenCache.getToken()
			if err != nil {
				return nil, err
			}
		}
		requestCtx, cancel := context.WithCancel(ctx)
		request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			cancel()
			return nil, err
		}
		request.Header = headers.Clone()
		if tokenCache != nil {
			if !overrideBearer {
				request.Header.Set("Authorization", strings.Trim("Bearer "+token, " \t\r\n"))
			}
			parts := []string{}
			if values, exists := request.Header["Anthropic-Beta"]; exists {
				for _, part := range strings.Split(strings.Join(values, ", "), ",") {
					parts = append(parts, strings.TrimSpace(part))
				}
			}
			found := false
			for _, part := range parts {
				if part == "oauth-2025-04-20" {
					found = true
				}
			}
			if !found {
				request.Header.Set("Anthropic-Beta", strings.Join(append(parts, "oauth-2025-04-20"), ","))
			}
		}
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
		if response.StatusCode == 401 && tokenCache != nil {
			tokenCache.invalidate()
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			data, readErr := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if readErr != nil {
				data = []byte(readErr.Error())
			}
			fallback := string(data)
			var errorBody json.RawMessage
			if json.Valid(data) && samplingTruthy(data) {
				errorBody = data
				fallback = ""
			}
			return nil, newCompletionsRequestError(response.StatusCode, errorBody, fallback, response.Header)
		}
		return response, nil
	}, retries, maxDelay, defaultCompletionsRetryTiming())
	if err != nil {
		return err
	}
	return consumeAnthropicResponse(ctx, acc, rawModel, options, response)
}

func prepareAnthropicPayload(ctx context.Context, acc *anthropicAccumulator, rawModel json.RawMessage, transcript TranscriptContext, options AnthropicStreamOptions, numeric *ThinkingTokenBudget) (json.RawMessage, error) {
	params, err := BuildAnthropicParams(rawModel, transcript, acc.oauth, options.Options)
	if err != nil {
		return nil, err
	}
	if numeric != nil {
		fields, _ := samplingObject(params)
		fields["max_tokens"] = anthropicJSONNumber(numeric.MaxTokens)
		// buildParams uses `thinkingBudgetTokens || 1024`, so its NaN
		// budget fallback is already preserved by the serialized null.
		params, _ = json.Marshal(fields)
	}
	if options.OnPayload != nil {
		next, err := options.OnPayload(ctx, params, rawModel)
		if err != nil {
			return nil, err
		}
		if next != nil {
			if !json.Valid(next) {
				return nil, errors.New("onPayload returned invalid JSON")
			}
			fields := anthropicSpreadObject(next)
			fields["stream"] = json.RawMessage(`true`)
			params, err = json.Marshal(fields)
			if err != nil {
				return nil, err
			}
		}
	}
	return params, nil
}

func consumeAnthropicResponse(ctx context.Context, acc *anthropicAccumulator, rawModel json.RawMessage, options AnthropicStreamOptions, response *http.Response) error {
	if response == nil {
		return errors.New("Anthropic client returned no response")
	}
	if response.Body != nil {
		defer response.Body.Close()
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
	acc.start()
	var reader io.Reader = response.Body
	if response.StatusCode == http.StatusNoContent {
		reader = nil
	}
	err := readAnthropicSSE(ctx, reader, func(raw json.RawMessage) error {
		if options.OnProviderStreamEvent != nil {
			if err := options.OnProviderStreamEvent(ctx, &raw, rawModel); err != nil {
				return err
			}
		}
		return acc.chunk(raw)
	})
	if err != nil {
		return err
	}
	acc.finish(ctx)
	return nil
}
