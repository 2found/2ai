package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

// StreamAntigravityPooled keeps account credentials out of the transcript and
// carries the selected account's project through each native request attempt.
func StreamAntigravityPooled(ctx context.Context, model json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, source TokenSource) (*AssistantMessageEventStream, error) {
	var identity struct{ API, Provider string }
	if err := json.Unmarshal(model, &identity); err != nil {
		return nil, err
	}
	if identity.API != VendorGoogleAntigravity || identity.Provider != VendorGoogleAntigravity {
		return nil, errors.New("Antigravity pool requires its bound google-antigravity model")
	}
	return nativeOAuthPoolStream(ctx, model, transcript, options, source, "Antigravity", streamAntigravityAccount)
}

func antigravityNativePayload(model completionsModel, transcript TranscriptContext, controls map[string]json.RawMessage, token OAuthToken, now int64) (json.RawMessage, error) {
	if strings.TrimSpace(token.ProjectID) == "" {
		return nil, errors.New("Antigravity account has no project ID")
	}
	transcript = ResolveTranscript(transcript, false)
	contents := []map[string]any{}
	messages := TransformMessages(transcript.Messages(), Model{ID: model.ID, API: model.API, Provider: model.Provider, Input: model.Input}, func(id string, _ *Model, _ *Message) string {
		normalized := strings.Map(func(r rune) rune {
			if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' {
				return r
			}
			return '_'
		}, id)
		if len(normalized) > 64 {
			normalized = normalized[:64]
		}
		return normalized
	})
	for _, message := range messages {
		if message.Role == "system" {
			continue
		}
		role := "user"
		parts := []map[string]any{}
		blocks := message.Content.Blocks.Values()
		if message.Content.Text != nil {
			blocks = []*ContentBlock{{Type: "text", Text: *message.Content.Text}}
		}
		same := message.Provider == model.Provider && message.Model == model.ID
		if message.Role == "assistant" {
			role = "model"
		}
		if message.Role == "toolResult" {
			texts := []string{}
			for _, block := range blocks {
				if block != nil && block.Type == "text" {
					texts = append(texts, block.Text)
				}
			}
			field := "output"
			if message.IsError {
				field = "error"
			}
			parts = append(parts, map[string]any{"functionResponse": map[string]any{"name": message.ToolName, "id": message.ToolCallID, "response": map[string]string{field: SanitizeSurrogates(strings.Join(texts, "\n"))}}})
		} else {
			for _, block := range blocks {
				if block == nil || block.IsNull() {
					continue
				}
				part := map[string]any{}
				var signature *string
				switch block.Type {
				case "text":
					part["text"] = SanitizeSurrogates(block.Text)
					signature = block.TextSignature
				case "thinking":
					part["text"] = SanitizeSurrogates(block.Thinking)
					if same {
						part["thought"] = true
					}
					signature = block.ThinkingSignature
				case "toolCall":
					args := block.Arguments
					if len(args) == 0 {
						args = json.RawMessage(`{}`)
					}
					part["functionCall"] = map[string]any{"name": block.Name, "id": block.ID, "args": args}
					signature = block.ThoughtSignature
				case "image":
					part["inlineData"] = map[string]string{"mimeType": block.MIMEType, "data": block.Data}
				default:
					continue
				}
				if same && signature != nil && *signature != "" {
					if _, err := base64.StdEncoding.DecodeString(*signature); err == nil {
						part["thoughtSignature"] = *signature
					}
				}
				parts = append(parts, part)
			}
		}
		if len(parts) > 0 {
			// Google expects parallel function responses together in one user turn.
			if message.Role == "toolResult" && len(contents) > 0 && contents[len(contents)-1]["role"] == "user" {
				prior := contents[len(contents)-1]
				prior["parts"] = append(prior["parts"].([]map[string]any), parts...)
			} else {
				contents = append(contents, map[string]any{"role": role, "parts": parts})
			}
		}
	}
	claude := strings.HasPrefix(strings.ToLower(model.ID), "claude-")
	ceiling := 65536
	if claude {
		ceiling = 64000
	}
	maxTokens := ceiling
	if model.MaxTokens > 0 {
		maxTokens = min(maxTokens, int(model.MaxTokens))
	}
	if value := antigravityOptionNumber(controls["maxTokens"]); value > 0 {
		maxTokens = min(ceiling, int(value))
	}
	effort := samplingString(controls["reasoning"])
	if effort == "" {
		effort = samplingString(controls["reasoningEffort"])
	}
	thinking, budget := antigravityThinking(model.ID, effort)
	generation := map[string]any{"maxOutputTokens": min(ceiling, maxTokens+budget)}
	if thinking != nil {
		generation["thinkingConfig"] = thinking
	}
	if raw, ok := controls["temperature"]; ok {
		generation["temperature"] = raw
	}
	trajectory := uuid.NewString()
	inner := map[string]any{"contents": contents, "generationConfig": generation, "labels": map[string]string{"used_claude": fmt.Sprint(claude), "used_claude_conservative": fmt.Sprint(claude), "trajectory_id": trajectory, "last_step_index": "1"}}
	if prompt := GetCurrentSystemPrompt(transcript.Messages()); prompt != "" {
		inner["systemInstruction"] = map[string]any{"role": "user", "parts": []map[string]string{{"text": SanitizeSurrogates(prompt)}}}
	}
	tools := GetCurrentTools(transcript.Messages())
	if len(tools) > 0 {
		declarations := []map[string]any{}
		for _, tool := range tools {
			var schema map[string]any
			if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
				return nil, err
			}
			declarations = append(declarations, map[string]any{"name": tool.Name, "description": tool.Description, "parameters": toolParameters(schema, toolSchemaCloudCodeAssist)})
		}
		inner["tools"] = []map[string]any{{"functionDeclarations": declarations}}
	}
	if len(tools) > 0 || claude {
		inner["toolConfig"] = map[string]any{"functionCallingConfig": map[string]string{"mode": "VALIDATED"}}
	}
	return json.Marshal(map[string]any{"project": token.ProjectID, "model": model.ID, "userAgent": "antigravity", "requestType": "agent", "requestId": fmt.Sprintf("agent/%s/%d/%s/2", uuid.NewString(), now, trajectory), "request": inner})
}

func streamAntigravityAccount(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, token OAuthToken) (*AssistantMessageEventStream, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return nil, err
		}
	}
	stream := NewAssistantMessageEventStreamFor(ctx)
	options, callbackFailure := nativeFailureCallbacks(options)
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	message := &Message{Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID, Timestamp: now, Content: BlockContent(), Usage: &Usage{}, StopReason: "stop"}
	go func() {
		defer stream.End()
		fail := func(err error) {
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			stream.Synchronize(func() {
				message.StopReason = "error"
				if ctx.Err() != nil {
					message.StopReason = "aborted"
				}
				text := err.Error()
				message.ErrorMessage = &text
				stream.Push(AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: message})
			})
		}
		defer func() {
			if recovered := recover(); recovered != nil {
				*callbackFailure = true
				fail(fmt.Errorf("Antigravity: %v", recovered))
			}
		}()
		err := runAntigravityNative(ctx, model, rawModel, transcript, controls, options, token, stream, message)
		if err != nil {
			fail(err)
		}
	}()
	return stream, nil
}

func runAntigravityNative(ctx context.Context, model completionsModel, rawModel json.RawMessage, transcript TranscriptContext, controls map[string]json.RawMessage, options OpenAICompletionsStreamOptions, token OAuthToken, stream *AssistantMessageEventStream, message *Message) error {
	payload, err := antigravityNativePayload(model, transcript, controls, token, message.Timestamp)
	if err != nil {
		return err
	}
	if options.OnPayload != nil {
		next, err := options.OnPayload(ctx, payload, rawModel)
		if err != nil {
			return err
		}
		if next != nil {
			payload = next
		}
	}
	if !json.Valid(payload) || bytes.Equal(bytes.TrimSpace(payload), []byte("null")) {
		return errors.New("Antigravity payload must be JSON")
	}
	client := options.Client
	if client == nil {
		client = NewStreamHTTPClient(0)
	}
	endpoints := []string{model.BaseURL}
	if model.BaseURL == "" || samplingTruthy(model.Compat["antigravityDefaultEndpoints"]) {
		endpoints = []string{antigravityDailyEndpoint, antigravitySandboxEndpoint}
	}
	if timeout := antigravityOptionNumber(controls["timeoutMs"]); timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(timeout)*time.Millisecond)
		defer cancel()
	}
	var response *http.Response
	for index, endpoint := range endpoints {
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+"/v1internal:streamGenerateContent?alt=sse", bytes.NewReader(payload))
		if err != nil {
			return err
		}
		request.Header.Set("Authorization", "Bearer "+token.AccessToken)
		request.Header.Set("User-Agent", AntigravityUserAgent)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Accept", "text/event-stream")
		response, err = client.Do(request)
		if err != nil {
			if ctx.Err() == nil && index+1 < len(endpoints) {
				continue
			}
			return err
		}
		if options.OnResponse != nil {
			headers := map[string]string{}
			for key, values := range response.Header {
				headers[strings.ToLower(key)] = strings.Join(values, ", ")
			}
			if err := options.OnResponse(ctx, CompletionsResponse{Status: response.StatusCode, Headers: headers}, rawModel); err != nil {
				response.Body.Close()
				return err
			}
		}
		if response.StatusCode < 400 {
			break
		}
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		response.Body.Close()
		if response.StatusCode >= 500 && index+1 < len(endpoints) {
			continue
		}
		detail := strings.TrimSpace(string(body))
		if detail == "" {
			detail = response.Status
		}
		return &completionsRequestError{status: response.StatusCode, body: body, headers: response.Header, message: detail}
	}
	defer response.Body.Close()
	stop := context.AfterFunc(ctx, func() { _ = response.Body.Close() })
	defer stop()
	stream.Push(AssistantMessageEvent{Type: "start", Partial: message})
	finished := false
	err = readNativeJSONSSE(ctx, response.Body, response.Header, false, func(raw json.RawMessage) error {
		if options.OnProviderStreamEvent != nil {
			if err := options.OnProviderStreamEvent(ctx, &raw, rawModel); err != nil {
				return err
			}
		}
		var chunk struct {
			Response struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							Text             string
							Thought          bool
							ThoughtSignature *string
							FunctionCall     *struct {
								Name string
								ID   string
								Args json.RawMessage
							}
						}
					}
					FinishReason string
				}
				UsageMetadata *struct{ PromptTokenCount, CandidatesTokenCount, CachedContentTokenCount, ThoughtsTokenCount float64 }
			}
			Error *struct {
				Code            int
				Status, Message string
			}
		}
		if err := json.Unmarshal(raw, &chunk); err != nil {
			return err
		}
		if chunk.Error != nil {
			status := chunk.Error.Code
			if status < 400 {
				status = 500
			}
			switch chunk.Error.Status {
			case "UNAUTHENTICATED":
				status = 401
			case "RESOURCE_EXHAUSTED":
				status = 429
			case "PERMISSION_DENIED":
				status = 403
			}
			return &completionsRequestError{status: status, headers: response.Header, message: chunk.Error.Message}
		}
		var chunkErr error
		stream.Synchronize(func() {
			if usage := chunk.Response.UsageMetadata; usage != nil {
				message.Usage.Input = max(0, usage.PromptTokenCount-usage.CachedContentTokenCount)
				message.Usage.CacheRead = usage.CachedContentTokenCount
				message.Usage.Output = usage.CandidatesTokenCount + usage.ThoughtsTokenCount
				message.Usage.Reasoning = &usage.ThoughtsTokenCount
				message.Usage.TotalTokens = message.Usage.Input + message.Usage.CacheRead + message.Usage.Output
				calculateNativeUsageCost(model.Cost, message.Usage)
			}
			if len(chunk.Response.Candidates) == 0 {
				return
			}
			candidate := chunk.Response.Candidates[0]
			for _, part := range candidate.Content.Parts {
				block := &ContentBlock{Type: "text", Text: part.Text, TextSignature: part.ThoughtSignature}
				kind, delta := "text", part.Text
				if call := part.FunctionCall; call != nil {
					kind = "toolcall"
					delta = string(call.Args)
					id := call.ID
					if id == "" {
						id = "ag-" + uuid.NewString()
					}
					args := call.Args
					if len(args) == 0 {
						args = json.RawMessage(`{}`)
						delta = "{}"
					}
					block = &ContentBlock{Type: "toolCall", ID: id, Name: call.Name, Arguments: args, ThoughtSignature: part.ThoughtSignature}
					message.StopReason = "toolUse"
				} else if part.Thought {
					kind = "thinking"
					block = &ContentBlock{Type: "thinking", Thinking: part.Text, ThinkingSignature: part.ThoughtSignature}
				}
				if delta == "" && part.ThoughtSignature == nil {
					continue
				}
				index := message.Content.Blocks.Len()
				message.Content.Blocks.Append(block)
				stream.Push(AssistantMessageEvent{Type: kind + "_start", ContentIndex: index, Partial: message})
				stream.Push(AssistantMessageEvent{Type: kind + "_delta", ContentIndex: index, Delta: delta, Partial: message})
				event := AssistantMessageEvent{Type: kind + "_end", ContentIndex: index, Content: delta, Partial: message}
				if kind == "toolcall" {
					event.ToolCall = block
				}
				stream.Push(event)
			}
			if candidate.FinishReason != "" {
				finished = true
				reason := candidate.FinishReason
				message.RawStopReason = &reason
				switch reason {
				case "STOP":
				case "MAX_TOKENS":
					message.StopReason = "length"
				default:
					chunkErr = fmt.Errorf("Antigravity finish reason: %s", reason)
				}
			}
		})
		return chunkErr
	})
	if err != nil {
		return err
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if !finished {
		return errors.New("Antigravity stream ended without a finish reason")
	}
	stream.Synchronize(func() { stream.Push(AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: message}) })
	return nil
}

func antigravityOptionNumber(raw json.RawMessage) float64 {
	var value float64
	_ = json.Unmarshal(raw, &value)
	return value
}
