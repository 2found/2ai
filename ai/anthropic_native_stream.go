package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Anthropic's wire index is transient. Lookup scans the live content in order,
// matching the source's first-match behavior even for repeated/missing indices.
type anthropicAccumulator struct {
	model           completionsModel
	cost            completionsCost
	oauth           bool
	tools           []Tool
	stream          *AssistantMessageEventStream
	push            func(AssistantMessageEvent)
	output          *Message
	transformations []json.RawMessage
	now             func() int64
	ended           bool
}

func newAnthropicAccumulator(model completionsModel, oauth bool, tools []Tool, options map[string]json.RawMessage, stream *AssistantMessageEventStream, now func() int64) *anthropicAccumulator {
	output := &Message{Role: "assistant", Content: BlockContent(), API: model.API, Provider: model.Provider, Model: model.ID, Usage: &Usage{}, StopReason: "pending", Timestamp: now()}
	if samplingTruthy(model.Compat["supportsMidConvoEffort"]) {
		effort := "high"
		if samplingNonNull(options["effort"]) {
			effort = samplingString(options["effort"])
		}
		output.ProviderThinkingLevel = &effort
	}
	return &anthropicAccumulator{model: model, cost: model.Cost, oauth: oauth, tools: tools, stream: stream, push: stream.Push, output: output, now: now}
}
func (a *anthropicAccumulator) publish(event AssistantMessageEvent) {
	if event.Type != "done" && event.Type != "error" {
		event.Partial = a.output
	}
	a.push(event)
}
func (a *anthropicAccumulator) start() {
	a.stream.Synchronize(func() { a.publish(AssistantMessageEvent{Type: "start"}) })
}
func (a *anthropicAccumulator) find(index json.RawMessage) (int, *ContentBlock) {
	key := responsesSlotKey(index)
	for i, block := range a.output.Content.Blocks.Values() {
		if responsesSlotKey(block.Extra["index"]) == key {
			return i, block
		}
	}
	return -1, nil
}
func (a *anthropicAccumulator) setString(key string, target **string, raw json.RawMessage) {
	*target = nil
	delete(a.output.Extra, key)
	if len(raw) == 0 {
		return
	}
	if !samplingNonNull(raw) {
		if a.output.Extra == nil {
			a.output.Extra = map[string]json.RawMessage{}
		}
		a.output.Extra[key] = raw
		return
	}
	value := samplingString(raw)
	*target = &value
}
func (a *anthropicAccumulator) setTransformations(raw json.RawMessage) {
	var items []json.RawMessage
	if len(raw) > 0 && raw[0] == '[' && json.Unmarshal(raw, &items) == nil {
		a.transformations = items
	}
}
func (a *anthropicAccumulator) updateCost() {
	usage := a.output.Usage
	usage.TotalTokens = usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
	calculateNativeUsageCost(a.cost, usage)
}
func anthropicNumber(raw json.RawMessage) float64 {
	value, _ := strconv.ParseFloat(string(raw), 64)
	return value
}
func (a *anthropicAccumulator) chunk(raw json.RawMessage) error {
	var failure error
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		event, ok := samplingObject(raw)
		if !ok {
			return
		}
		switch samplingString(event["type"]) {
		case "message_start":
			message, _ := samplingObject(event["message"])
			a.setString("responseId", &a.output.ResponseID, message["id"])
			a.setTransformations(message["input_transformations"])
			responseModel := samplingString(message["model"])
			if responseModel != a.model.ID {
				a.setString("responseModel", &a.output.ResponseModel, message["model"])
			}
			a.cost = a.model.Cost
			if responseModel != a.model.ID {
				var fallbacks []struct {
					Provider, Model string
					Cost            json.RawMessage
				}
				_ = json.Unmarshal(a.model.Compat["allowedFallbackModels"], &fallbacks)
				for _, fallback := range fallbacks {
					if fallback.Provider == a.model.Provider && fallback.Model == responseModel {
						if samplingTruthy(fallback.Cost) {
							a.cost = completionsCost{}
							_ = json.Unmarshal(fallback.Cost, &a.cost)
						}
						break
					}
				}
			}
			usage, _ := samplingObject(message["usage"])
			observeWireUsage(a.output.Usage, usage, "input_tokens", "output_tokens")
			observeOptionalWireUsage(a.output.Usage, usage, "cache_read_input_tokens", "cache_creation_input_tokens")
			cache, _ := samplingObject(usage["cache_creation"])
			observeOptionalWireUsage(a.output.Usage, cache, "ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens")
			a.output.Usage.Input = anthropicNumber(usage["input_tokens"])
			a.output.Usage.Output = anthropicNumber(usage["output_tokens"])
			a.output.Usage.CacheRead = anthropicNumber(usage["cache_read_input_tokens"])
			a.output.Usage.CacheWrite = anthropicNumber(usage["cache_creation_input_tokens"])
			long := anthropicNumber(cache["ephemeral_1h_input_tokens"])
			a.output.Usage.CacheWrite1h = &long
			a.updateCost()
		case "content_block_start":
			content, _ := samplingObject(event["content_block"])
			if samplingString(content["type"]) == "fallback" {
				if a.output.Content.Blocks.Len() > 0 {
					failure = fmt.Errorf("Anthropic performed an unsupported mid-output model fallback")
				}
				return
			}
			block := &ContentBlock{Extra: map[string]json.RawMessage{}}
			if index, exists := event["index"]; exists {
				block.Extra["index"] = index
			}
			kind := ""
			switch samplingString(content["type"]) {
			case "text":
				block.Type = "text"
				block.Text = samplingString(content["text"])
				kind = "text_start"
			case "thinking":
				block.Type = "thinking"
				block.Thinking = samplingString(content["thinking"])
				signature := samplingString(content["signature"])
				block.ThinkingSignature = &signature
				kind = "thinking_start"
			case "redacted_thinking":
				block.Type = "thinking"
				block.Thinking = "[Reasoning redacted]"
				redacted := true
				block.Redacted = &redacted
				kind = "thinking_start"
				if signature, exists := content["data"]; exists {
					if samplingNonNull(signature) {
						value := samplingString(signature)
						block.ThinkingSignature = &value
					} else {
						block.Extra["thinkingSignature"] = signature
					}
				}
			case "tool_use":
				block.Type = "toolCall"
				block.ID = samplingString(content["id"])
				block.Name = samplingString(content["name"])
				kind = "toolcall_start"
				if a.oauth {
					lower := func(name string) string { return strings.ToLower(strings.ReplaceAll(name, "İ", "i\u0307")) }
					for _, tool := range a.tools {
						if lower(tool.Name) == lower(block.Name) {
							block.Name = tool.Name
							break
						}
					}
				}
				block.Arguments = content["input"]
				if !samplingNonNull(block.Arguments) {
					block.Arguments = json.RawMessage(`{}`)
				}
				block.Extra["partialJson"] = json.RawMessage(`""`)
			default:
				return
			}
			a.output.Content.Blocks.Append(block)
			a.publish(AssistantMessageEvent{Type: kind, ContentIndex: a.output.Content.Blocks.Len() - 1})
		case "content_block_delta":
			delta, _ := samplingObject(event["delta"])
			index, block := a.find(event["index"])
			if block == nil {
				return
			}
			switch samplingString(delta["type"]) {
			case "text_delta":
				if block.Type == "text" {
					value := samplingString(delta["text"])
					block.Text += value
					a.publish(AssistantMessageEvent{Type: "text_delta", ContentIndex: index, Delta: value})
				}
			case "thinking_delta":
				if block.Type == "thinking" {
					value := samplingString(delta["thinking"])
					block.Thinking += value
					a.publish(AssistantMessageEvent{Type: "thinking_delta", ContentIndex: index, Delta: value})
				}
			case "input_json_delta":
				if block.Type == "toolCall" {
					partial := samplingString(block.Extra["partialJson"])
					if _, exists := block.Extra["partialJson"]; !exists {
						partial = "undefined"
					}
					value := samplingString(delta["partial_json"])
					partial += value
					block.Extra["partialJson"] = json.RawMessage(marshalSamplingString(partial))
					block.Arguments = ParseStreamingJSON(partial)
					a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: index, Delta: value})
				}
			case "signature_delta":
				if block.Type == "thinking" {
					signature := ""
					if block.ThinkingSignature != nil {
						signature = *block.ThinkingSignature
					}
					signature += samplingString(delta["signature"])
					block.ThinkingSignature = &signature
					delete(block.Extra, "thinkingSignature")
				}
			}
		case "content_block_stop":
			index, block := a.find(event["index"])
			if block == nil {
				return
			}
			delete(block.Extra, "index")
			switch block.Type {
			case "text":
				a.publish(AssistantMessageEvent{Type: "text_end", ContentIndex: index, Content: block.Text})
			case "thinking":
				a.publish(AssistantMessageEvent{Type: "thinking_end", ContentIndex: index, Content: block.Thinking})
			case "toolCall":
				block.Arguments = ParseStreamingJSON(samplingString(block.Extra["partialJson"]))
				delete(block.Extra, "partialJson")
				a.publish(AssistantMessageEvent{Type: "toolcall_end", ContentIndex: index, ToolCall: block})
			}
		case "message_delta":
			a.setTransformations(event["input_transformations"])
			delta, _ := samplingObject(event["delta"])
			if samplingTruthy(delta["stop_reason"]) {
				a.setString("rawStopReason", &a.output.RawStopReason, delta["stop_reason"])
				details, _ := samplingObject(delta["stop_details"])
				stop, message, err := anthropicStopReason(samplingString(delta["stop_reason"]), details)
				if err != nil {
					failure = err
					return
				}
				a.output.StopReason = stop
				if message != nil {
					a.output.ErrorMessage = message
				}
			}
			if samplingTruthy(event["usage"]) {
				usage, _ := samplingObject(event["usage"])
				for key, raw := range usage {
					switch key {
					case "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens":
						if !validObservedNumber(raw) {
							a.output.Usage.Observation.UsageObserved = false
							a.output.Usage.Observation.UsageSource = "invalid"
						}
					}
				}
				for _, field := range []struct {
					key    string
					target *float64
				}{{"input_tokens", &a.output.Usage.Input}, {"output_tokens", &a.output.Usage.Output}, {"cache_read_input_tokens", &a.output.Usage.CacheRead}, {"cache_creation_input_tokens", &a.output.Usage.CacheWrite}} {
					if samplingNonNull(usage[field.key]) {
						*field.target = anthropicNumber(usage[field.key])
					}
				}
				cache, _ := samplingObject(usage["cache_creation"])
				observeOptionalWireUsage(a.output.Usage, cache, "ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens")
				if samplingNonNull(cache["ephemeral_1h_input_tokens"]) {
					value := anthropicNumber(cache["ephemeral_1h_input_tokens"])
					a.output.Usage.CacheWrite1h = &value
				}
				output, _ := samplingObject(usage["output_tokens_details"])
				if samplingNonNull(output["thinking_tokens"]) {
					value := anthropicNumber(output["thinking_tokens"])
					a.output.Usage.Reasoning = &value
				}
			}
			a.updateCost()
		}
	})
	return failure
}
func anthropicStopReason(reason string, details map[string]json.RawMessage) (string, *string, error) {
	switch reason {
	case "end_turn", "pause_turn", "stop_sequence":
		return "stop", nil, nil
	case "max_tokens":
		return "length", nil, nil
	case "tool_use":
		return "toolUse", nil, nil
	case "refusal":
		message := samplingString(details["explanation"])
		if message == "" {
			message = "The model refused to complete the request"
		}
		return "error", &message, nil
	case "sensitive":
		message := "Provider stopped with: sensitive"
		return "error", &message, nil
	default:
		return "", nil, fmt.Errorf("Unhandled stop reason: %s", reason)
	}
}
func (a *anthropicAccumulator) finish(ctx context.Context) {
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		failure := ""
		if ctx.Err() != nil {
			failure = "Request was aborted"
		} else if a.output.StopReason == "pending" {
			failure = "Anthropic stream ended without a stop reason"
		} else if a.output.StopReason == "aborted" || a.output.StopReason == "error" {
			failure = "An unknown error occurred"
			if a.output.ErrorMessage != nil && *a.output.ErrorMessage != "" {
				failure = *a.output.ErrorMessage
			}
		}
		if failure != "" {
			a.failLocked(failure, ctx.Err() != nil)
			return
		}
		if len(a.transformations) > 0 {
			entries := []map[string]json.RawMessage{}
			for _, raw := range a.transformations {
				if strings.TrimSpace(string(raw)) == "null" {
					a.failLocked("null is not an object (evaluating 'transformation.type')", false)
					return
				}
				transformation, _ := samplingObject(raw)
				entry := map[string]json.RawMessage{}
				for _, key := range []string{"type", "path", "reason"} {
					if samplingNonNull(transformation[key]) {
						entry[key] = transformation[key]
					}
				}
				entries = append(entries, entry)
			}
			var diagnostics []json.RawMessage
			_ = json.Unmarshal(a.output.Diagnostics, &diagnostics)
			diagnostic, _ := json.Marshal(map[string]any{"type": "anthropic_input_transformations", "timestamp": a.now(), "details": map[string]any{"transformations": entries}})
			diagnostics = append(diagnostics, diagnostic)
			a.output.Diagnostics, _ = json.Marshal(diagnostics)
		}
		a.ended = true
		a.publish(AssistantMessageEvent{Type: "done", Reason: a.output.StopReason, Message: a.output})
		a.stream.End()
	})
}
func (a *anthropicAccumulator) fail(message string, aborted bool) {
	a.stream.Synchronize(func() {
		if !a.ended {
			a.failLocked(message, aborted)
		}
	})
}
func (a *anthropicAccumulator) failLocked(message string, aborted bool) {
	for _, block := range a.output.Content.Blocks.Values() {
		delete(block.Extra, "index")
		delete(block.Extra, "partialJson")
	}
	a.output.StopReason = "error"
	if aborted {
		a.output.StopReason = "aborted"
	}
	a.output.ErrorMessage = &message
	a.ended = true
	a.publish(AssistantMessageEvent{Type: "error", Reason: a.output.StopReason, Error: a.output})
	a.stream.End()
}
