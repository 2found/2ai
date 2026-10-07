package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
	"golang.org/x/text/encoding/unicode"
)

// ProxyStreamOptions configures Pi's compact HTTP streaming protocol. Options
// holds provider controls; only the protocol's explicit allowlist is sent.
// Client and Now are optional Go transport/clock dependencies.
type ProxyStreamOptions struct {
	ProxyURL  string
	AuthToken string
	Options   map[string]any
	Client    *http.Client
	Now       func() int64
}

// StreamProxy reconstructs assistant messages in Go, without a JS worker.
// It returns immediately; request/protocol failures settle an error event.
// Partial pointers stay live. Use the stream's snapshot methods or Synchronize
// when inspecting payloads while the producer is still running.
func StreamProxy(ctx context.Context, model json.RawMessage, transcript ai.TranscriptContext, options ProxyStreamOptions) *ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStreamFor(ctx)
	// Pi captures identity before the clock, then timestamps the partial before
	// JSON.stringify invokes user serializers. Those callbacks can mutate inputs.
	partial, identityErr := proxyPartial(model)
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	// Freeze the request before returning so the goroutine does not retain a
	// caller-owned map/transcript. JSON failures follow the asynchronous path.
	controls := proxyRequestOptions{}
	for _, key := range []string{"temperature", "samplingParams", "maxTokens", "reasoning", "cacheRetention", "sessionId", "headers", "metadata", "transport", "thinkingBudgets", "maxRetryDelayMs"} {
		if value, ok := options.Options[key]; ok {
			controls = append(controls, proxyRequestOption{key, value})
		}
	}
	body, requestErr := marshalProxyRequest(struct {
		Model   json.RawMessage      `json:"model"`
		Context ai.TranscriptContext `json:"context"`
		Options proxyRequestOptions  `json:"options"`
	}{model, transcript, controls})
	if requestErr == nil {
		requestErr = identityErr
	}
	partial.Timestamp = now
	go func() {
		defer stream.End()
		err := requestErr
		if err == nil {
			err = readProxy(ctx, stream, partial, body, options)
		}
		if err != nil {
			stream.Synchronize(func() {
				reason, message := "error", err.Error()
				if ctx.Err() != nil {
					reason = "aborted"
				}
				partial.StopReason, partial.ErrorMessage = reason, &message
				stream.Push(ai.AssistantMessageEvent{Type: "error", Reason: reason, Error: partial})
			})
		}
	}()
	return stream
}

// Copy only the exact model properties Pi reads, before the clock or request
// serializers can mutate the source. Message's wire decoder retains omission,
// null, non-string values and UTF-16 without inventing empty identity fields.
func proxyPartial(model json.RawMessage) (*ai.Message, error) {
	partial := &ai.Message{Role: "assistant", StopReason: "pending", Content: ai.BlockContent(), Usage: &ai.Usage{}}
	properties, err := jsonjs.DecodeObjectProperties(model)
	if err != nil {
		return partial, err
	}
	fields := map[string]any{"role": partial.Role, "stopReason": partial.StopReason,
		"content": partial.Content, "usage": partial.Usage, "timestamp": 0}
	for _, property := range properties {
		switch property.Name {
		case "api", "provider":
			fields[property.Name] = property.Value
		case "id":
			fields["model"] = property.Value
		}
	}
	raw, err := json.Marshal(fields)
	if err == nil {
		err = json.Unmarshal(raw, partial)
	}
	return partial, err
}

// JSON.stringify runs inside Pi's producer try/catch. Go MarshalJSON callbacks
// may return errors or panic; both must settle the stream instead of escaping
// synchronously. encoding/json's wrapper is not part of the callback's error.
func marshalProxyRequest(value any) (body []byte, err error) {
	defer func() {
		if failure := recover(); failure != nil {
			body, err = nil, failureError(failure)
		}
	}()
	body, err = marshalJSScalar(value)
	if err == nil {
		return ai.StringifyJSON(body)
	}
	for {
		wrapped, ok := err.(*json.MarshalerError)
		if !ok {
			return body, err
		}
		err = wrapped.Err
	}
}

func readProxy(ctx context.Context, stream *ai.AssistantMessageEventStream, partial *ai.Message, body []byte, options ProxyStreamOptions) (err error) {
	// Pi catches both fetch rejection and reader rejection inside the producer.
	// Go transport/reader callbacks may panic instead of returning an error;
	// convert those at this boundary so the stream still publishes its result.
	defer func() {
		if value := recover(); value != nil {
			err = failureError(value)
		}
	}()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, options.ProxyURL+"/api/stream", bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+options.AuthToken)
	request.Header.Set("Content-Type", "application/json")
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	var closeOnce sync.Once
	closeBody := func() { closeOnce.Do(func() { _ = response.Body.Close() }) }
	defer closeBody()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return errors.New(proxyHTTPErrorMessage(response.Body, "Proxy error: "+response.Status))
	}
	// Pi cancels the active reader independently of fetch's abort handling.
	// An injected Go transport may not wire its response body to request.Context.
	// Close exactly once to release such reads, including an already-canceled
	// context; a cleanup panic must not escape the abort callback goroutine.
	stopAbort := context.AfterFunc(ctx, func() {
		defer func() { _ = recover() }()
		closeBody()
	})
	defer stopAbort()
	// Unlike Scanner, ReadString imposes no 64 KiB token limit. The decoder
	// handles UTF-8 sequences split across network reads and a leading BOM.
	reader := bufio.NewReader(unicode.UTF8BOM.NewDecoder().Reader(response.Body))
	terminal := false
	accumulator := proxyAccumulator{message: partial}
	for {
		line, readErr := reader.ReadString('\n')
		if ctx.Err() != nil {
			return errors.New("Request aborted by user")
		}
		if len(line) > 0 && (readErr == nil || readErr == io.EOF) {
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimFunc(line[6:], func(r rune) bool {
					return r >= '\u2000' && r <= '\u200a' || strings.ContainsRune("\t\n\v\f\r \u00a0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff", r)
				})
				if data != "" {
					frame, err := decodeProxyEvent([]byte(data))
					if err != nil {
						return err
					}
					var event ai.AssistantMessageEvent
					stream.Synchronize(func() {
						event, err = accumulator.process(frame)
						if err == nil && event.Type != "" {
							stream.Push(event)
						}
					})
					if err != nil {
						return err
					}
					if event.Type == "done" || event.Type == "error" {
						terminal = true
					}
				}
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if !terminal {
		return errors.New("Connection closed by proxy server before the response completed")
	}
	return nil
}

// Response.json decodes UTF-8 (including its BOM), then JSON.parse retains
// binary64 overflow and UTF-16 code units. Pi ignores body/read/coercion failures
// here and keeps the HTTP status message. Property lookup is case-sensitive.
func proxyHTTPErrorMessage(body io.Reader, fallback string) (message string) {
	message = fallback
	defer func() { _ = recover() }()
	raw, err := io.ReadAll(unicode.UTF8BOM.NewDecoder().Reader(body))
	if err != nil {
		return
	}
	value, err := jsonjs.DecodeJSON(raw)
	if err != nil {
		return
	}
	object, ok := value.(map[string]any)
	if !ok || !proxyTruthy(object["error"]) {
		return
	}
	if detail, ok := proxyHTTPErrorString(object["error"]); ok {
		message = "Proxy error: " + detail
	}
	return
}

func proxyHTTPErrorString(value any) (string, bool) {
	switch value := value.(type) {
	case map[string]any:
		// JSON can shadow Object.prototype.toString with a non-callable own
		// property. valueOf cannot supply a primitive either, so coercion fails.
		if _, shadowed := value["toString"]; shadowed {
			return "", false
		}
	case []any:
		parts := make([]string, len(value))
		for i, item := range value {
			if item == nil {
				continue
			}
			part, ok := proxyHTTPErrorString(item)
			if !ok {
				return "", false
			}
			parts[i] = part
		}
		return strings.Join(parts, ","), true
	}
	return jsValueString(value), true
}

type proxyEvent struct {
	Type                  string                     `json:"type"`
	ContentIndex          int                        `json:"contentIndex"`
	Delta                 string                     `json:"delta"`
	ContentSignature      json.RawMessage            `json:"contentSignature"`
	ID                    string                     `json:"id"`
	ToolName              string                     `json:"toolName"`
	ToolCall              map[string]json.RawMessage `json:"toolCall"`
	Reason                string                     `json:"reason"`
	Usage                 json.RawMessage            `json:"usage"`
	ErrorMessage          json.RawMessage            `json:"errorMessage"`
	ProviderThinkingLevel json.RawMessage            `json:"providerThinkingLevel"`
}

// Pi parses JSON before switching on type. Unknown types and JSON primitives
// are ignored; each known branch reads only its own fields, so unrelated
// extension fields must not trigger typed Go decoding failures.
func decodeProxyEvent(raw []byte) (proxyEvent, error) {
	if err := jsonjs.ValidateJSON(raw); err != nil {
		return proxyEvent{}, err
	}
	raw = bytes.TrimSpace(raw)
	if bytes.Equal(raw, []byte("null")) {
		return proxyEvent{}, jsonjs.PropertyReadError(true, "proxyEvent.type")
	}
	if raw[0] != '{' {
		return proxyEvent{}, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return proxyEvent{}, err
	}
	var kind string
	if err := json.Unmarshal(fields["type"], &kind); err != nil {
		return proxyEvent{}, nil
	}
	var names []string
	switch kind {
	case "start":
	case "text_start", "thinking_start":
		names = []string{"contentIndex"}
	case "toolcall_start":
		names = []string{"contentIndex", "id", "toolName"}
	case "text_delta", "thinking_delta", "toolcall_delta":
		names = []string{"contentIndex", "delta"}
	case "text_end", "thinking_end":
		names = []string{"contentIndex", "contentSignature"}
	case "toolcall_end":
		names = []string{"contentIndex", "toolCall"}
	case "done":
		names = []string{"reason", "usage", "providerThinkingLevel"}
	case "error":
		names = []string{"reason", "usage", "providerThinkingLevel", "errorMessage"}
	default:
		return proxyEvent{}, nil
	}
	frame := proxyEvent{Type: kind}
	for _, name := range names {
		value, ok := fields[name]
		if !ok {
			continue
		}
		var target any
		switch name {
		case "contentIndex":
			target = &frame.ContentIndex
		case "id":
			target = &frame.ID
		case "toolName":
			target = &frame.ToolName
		case "delta":
			target = &frame.Delta
		case "reason":
			target = &frame.Reason
		case "toolCall":
			target = &frame.ToolCall
		case "contentSignature":
			frame.ContentSignature = value
		case "usage":
			frame.Usage = value
		case "providerThinkingLevel":
			frame.ProviderThinkingLevel = value
		case "errorMessage":
			frame.ErrorMessage = value
		}
		if target != nil {
			if err := json.Unmarshal(value, target); err != nil {
				return proxyEvent{}, err
			}
			if text, ok := target.(*string); ok {
				decoded, err := jsonjs.DecodeJSON(value)
				if err != nil {
					return proxyEvent{}, err
				}
				if decoded, ok := decoded.(string); ok {
					*text = decoded
				}
			}
		}
	}
	return frame, nil
}

// Content entries and toolcall_end events share live block objects and the
// message's content list, including across sparse writes and terminal events.
type proxyAccumulator struct {
	message *ai.Message
}

func (p *proxyAccumulator) process(frame proxyEvent) (ai.AssistantMessageEvent, error) {
	partial := p.message
	event := ai.AssistantMessageEvent{Type: frame.Type, ContentIndex: frame.ContentIndex, Partial: partial}
	var content *ai.ContentBlock
	if frame.ContentIndex >= 0 && frame.ContentIndex < partial.Content.Blocks.Len() {
		content = partial.Content.Blocks.Get(frame.ContentIndex)
	}
	require := func(kind string) error {
		if content == nil || content.Type != kind {
			return fmt.Errorf("Received %s for non-%s content", frame.Type, kind)
		}
		return nil
	}
	switch frame.Type {
	case "start":
	case "text_start", "thinking_start", "toolcall_start":
		if frame.ContentIndex < 0 {
			return ai.AssistantMessageEvent{}, errors.New("Invalid negative contentIndex")
		}
		content = &ai.ContentBlock{}
		partial.Content.Blocks.Set(frame.ContentIndex, content)
		switch frame.Type {
		case "text_start":
			*content = ai.ContentBlock{Type: "text"}
		case "thinking_start":
			*content = ai.ContentBlock{Type: "thinking"}
		case "toolcall_start":
			*content = ai.ContentBlock{Type: "toolCall", ID: frame.ID, Name: frame.ToolName, Arguments: json.RawMessage(`{}`), Extra: map[string]json.RawMessage{"partialJson": json.RawMessage(`""`)}}
		}
	case "text_delta", "text_end":
		if err := require("text"); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		if frame.Type == "text_delta" {
			content.Text += frame.Delta
			event.Delta = frame.Delta
		} else {
			var err error
			content.TextSignature, err = proxySignature(content, "textSignature", frame.ContentSignature)
			if err != nil {
				return ai.AssistantMessageEvent{}, err
			}
			event.Content = content.Text
		}
	case "thinking_delta", "thinking_end":
		if err := require("thinking"); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		if frame.Type == "thinking_delta" {
			content.Thinking += frame.Delta
			event.Delta = frame.Delta
		} else {
			var err error
			content.ThinkingSignature, err = proxySignature(content, "thinkingSignature", frame.ContentSignature)
			if err != nil {
				return ai.AssistantMessageEvent{}, err
			}
			event.Content = content.Thinking
		}
	case "toolcall_delta":
		if err := require("toolCall"); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		var input string
		if data, ok := content.Extra["partialJson"]; ok {
			decoded, _ := jsonjs.DecodeJSON(data)
			input, _ = decoded.(string)
		} else {
			input = "undefined"
		}
		input += frame.Delta
		if content.Extra == nil {
			content.Extra = map[string]json.RawMessage{}
		}
		content.Extra["partialJson"] = jsonjs.QuoteString(input)
		content.Arguments = ai.ParseStreamingJSON(input)
		var parsed any
		_ = json.Unmarshal(content.Arguments, &parsed)
		if !proxyTruthy(parsed) {
			content.Arguments = json.RawMessage(`{}`)
		}
		event.Delta = frame.Delta
		// Pi changes the old object, then installs a shallow copy for reactivity.
		copied := *content
		copied.Extra = make(map[string]json.RawMessage, len(content.Extra))
		for key, value := range content.Extra {
			copied.Extra[key] = value
		}
		partial.Content.Blocks.Set(frame.ContentIndex, &copied)
	case "toolcall_end":
		if content == nil || content.Type != "toolCall" {
			return ai.AssistantMessageEvent{}, nil
		}
		// Object.assign preserves fields absent from the terminal frame,
		// including metadata the Go union does not know about.
		data, err := json.Marshal(content)
		if err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		for key, value := range frame.ToolCall {
			fields[key] = value
		}
		delete(fields, "partialJson")
		data, err = json.Marshal(fields)
		if err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		if err := json.Unmarshal(data, content); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		event.ToolCall = content
	case "done", "error":
		if err := p.terminal(frame); err != nil {
			return ai.AssistantMessageEvent{}, err
		}
		event.Partial, event.Reason = nil, frame.Reason
		if frame.Type == "done" {
			event.Message = partial
		} else {
			event.Error = partial
		}
	default:
		return ai.AssistantMessageEvent{}, nil
	}
	return event, nil
}

func proxySignature(content *ai.ContentBlock, name string, raw json.RawMessage) (*string, error) {
	var signature *string
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &signature); err != nil {
			return nil, err
		}
		if signature != nil {
			value, err := jsonjs.DecodeJSON(raw)
			if err != nil {
				return nil, err
			}
			*signature = value.(string)
		}
	}
	delete(content.Extra, name)
	if len(raw) > 0 && signature == nil {
		if content.Extra == nil {
			content.Extra = map[string]json.RawMessage{}
		}
		content.Extra[name] = json.RawMessage(`null`)
	}
	return signature, nil
}

// Pi assigns terminal fields directly, preserving null versus omission and
// provider usage extensions. Decode the updated wire record through Message's
// lossless encoding without replacing the published partial pointer. An absent
// thinking level retains the old value; absent usage/errorMessage removes it.
func (p *proxyAccumulator) terminal(frame proxyEvent) error {
	raw, err := json.Marshal(p.message)
	if err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	assign := func(name string, value json.RawMessage) {
		if len(value) == 0 {
			delete(fields, name)
		} else {
			fields[name] = value
		}
	}
	fields["stopReason"] = jsonjs.QuoteString(frame.Reason)
	assign("usage", frame.Usage)
	if len(frame.ProviderThinkingLevel) > 0 {
		assign("providerThinkingLevel", frame.ProviderThinkingLevel)
	}
	if frame.Type == "error" {
		assign("errorMessage", frame.ErrorMessage)
	}
	raw, err = json.Marshal(fields)
	if err != nil {
		return err
	}
	var updated ai.Message
	if err := json.Unmarshal(raw, &updated); err != nil {
		return err
	}
	// Terminal assignment does not touch content. Decoding it would break list
	// and block identity and turn sparse holes into explicit null entries.
	updated.Content = p.message.Content
	*p.message = updated
	return nil
}

func proxyTruthy(value any) bool {
	switch value := value.(type) {
	case nil:
		return false
	case bool:
		return value
	case string:
		return value != ""
	case float64:
		return value != 0
	default:
		return true
	}
}
