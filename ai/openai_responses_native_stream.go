package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Responses slots are indexed by output_index, which is independent of the
// transcript content index. Block pointers remain stable across slice growth.
type responsesSlot struct {
	block        *ContentBlock
	contentIndex int
	custom       *completionsCustomInput
}
type responsesAccumulator struct {
	model                   completionsModel
	grammar                 map[string]string
	stream                  *AssistantMessageEventStream
	push                    func(AssistantMessageEvent)
	output                  *Message
	slots                   map[string]*responsesSlot
	reasoning               map[string]*ContentBlock
	sawTerminal, ended      bool
	serviceTier             json.RawMessage
	resolveServiceTier      func(responseTier, requestTier json.RawMessage) json.RawMessage
	applyServiceTierPricing func(*Usage, json.RawMessage)
	formatFailure           func(string) string
	missingStopReason       string
}

func newResponsesAccumulator(model completionsModel, grammar map[string]string, stream *AssistantMessageEventStream, timestamp int64) *responsesAccumulator {
	return &responsesAccumulator{model: model, grammar: grammar, stream: stream, push: stream.Push,
		output: &Message{Role: "assistant", Content: BlockContent(), API: model.API, Provider: model.Provider, Model: model.ID, Usage: &Usage{}, StopReason: "pending", Timestamp: timestamp},
		slots:  map[string]*responsesSlot{}, reasoning: map[string]*ContentBlock{}}
}
func (a *responsesAccumulator) publish(event AssistantMessageEvent) {
	if event.Type != "done" && event.Type != "error" {
		event.Partial = a.output
	}
	a.push(event)
}
func (a *responsesAccumulator) start() {
	a.stream.Synchronize(func() { a.publish(AssistantMessageEvent{Type: "start"}) })
}
func responsesSlotKey(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "undefined"
	}
	normalized, err := stringifyCompletionsJSON(raw)
	if err != nil {
		return string(raw)
	}
	return string(normalized)
}
func responsesNamespace(block *ContentBlock, item map[string]json.RawMessage) {
	if value, exists := item["namespace"]; exists {
		block.Namespace = nil
		delete(block.Extra, "namespace")
		if string(value) == "null" {
			block.Extra["namespace"] = value
		} else {
			name := samplingString(value)
			block.Namespace = &name
		}
	}
}
func (a *responsesAccumulator) applyPhase(item map[string]json.RawMessage) {
	if samplingString(item["type"]) == "message" && samplingString(item["phase"]) == "final_answer" {
		a.output.StopReason = "stop"
	}
}
func (a *responsesAccumulator) createSlot(key string, item map[string]json.RawMessage) *responsesSlot {
	block := &ContentBlock{Extra: map[string]json.RawMessage{}}
	slot := &responsesSlot{block: block, contentIndex: a.output.Content.Blocks.Len()}
	event := ""
	switch samplingString(item["type"]) {
	case "reasoning":
		block.Type = "thinking"
		event = "thinking_start"
	case "message":
		a.applyPhase(item)
		block.Type = "text"
		event = "text_start"
	case "function_call", "custom_tool_call":
		block.Type = "toolCall"
		event = "toolcall_start"
		block.ID = responsesJSString(item["call_id"]) + "|" + responsesJSString(item["id"])
		block.Name = samplingString(item["name"])
		responsesNamespace(block, item)
		if samplingString(item["type"]) == "function_call" {
			block.Arguments = json.RawMessage(`{}`)
			block.Extra["partialJson"] = json.RawMessage(marshalSamplingString(samplingString(item["arguments"])))
		} else {
			property, ok := a.grammar[block.Name]
			if !ok {
				property = "input"
			}
			slot.custom = &completionsCustomInput{Property: property}
			block.Arguments = marshalSamplingObject(map[string]json.RawMessage{property: json.RawMessage(marshalSamplingString(samplingString(item["input"])))}, []string{property})
			block.Extra["customInput"], _ = json.Marshal(slot.custom)
		}
	default:
		return nil
	}
	a.output.Content.Blocks.Append(block)
	a.slots[key] = slot
	a.publish(AssistantMessageEvent{Type: event, ContentIndex: slot.contentIndex})
	return slot
}
func (a *responsesAccumulator) customInput(slot *responsesSlot) string {
	args, _ := samplingObject(slot.block.Arguments)
	return samplingString(args[slot.custom.Property])
}
func (a *responsesAccumulator) appendCustom(slot *responsesSlot, next string, close bool) error {
	custom := slot.custom
	delta, err := AppendGrammarToolInputJSONDelta(&custom.JSONBuffer, custom.Property, next, close)
	if err != nil {
		return err
	}
	slot.block.Arguments = marshalSamplingObject(map[string]json.RawMessage{custom.Property: json.RawMessage(marshalSamplingString(next))}, []string{custom.Property})
	slot.block.Extra["customInput"], _ = json.Marshal(custom)
	if delta != nil {
		a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: slot.contentIndex, Delta: *delta})
	}
	return nil
}

func (a *responsesAccumulator) chunk(raw json.RawMessage) error {
	var failure error
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		event, ok := samplingObject(raw)
		if strings.TrimSpace(string(raw)) == "null" {
			failure = fmt.Errorf("null is not an object (evaluating 'event.type')")
			return
		}
		if !ok {
			return
		}
		key := responsesSlotKey(event["output_index"])
		slot := a.slots[key]
		item, _ := samplingObject(event["item"])
		delta := samplingString(event["delta"])
		switch samplingString(event["type"]) {
		case "response.created":
			response, _ := samplingObject(event["response"])
			a.setMessageString("responseId", &a.output.ResponseID, response["id"])
		case "response.output_item.added":
			a.createSlot(key, item)
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_part.done":
			if slot == nil || slot.block.Type != "thinking" {
				return
			}
			if samplingString(event["type"]) == "response.reasoning_summary_part.done" {
				delta = "\n\n"
			}
			slot.block.Thinking += delta
			a.publish(AssistantMessageEvent{Type: "thinking_delta", ContentIndex: slot.contentIndex, Delta: delta})
		case "response.output_text.delta", "response.refusal.delta":
			if slot == nil || slot.block.Type != "text" {
				return
			}
			slot.block.Text += delta
			a.publish(AssistantMessageEvent{Type: "text_delta", ContentIndex: slot.contentIndex, Delta: delta})
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			if slot == nil || slot.block.Type != "toolCall" {
				return
			}
			old, exists := slot.block.Extra["partialJson"]
			if !exists {
				return
			}
			previous := samplingString(old)
			next := previous + delta
			emit := true
			if samplingString(event["type"]) == "response.function_call_arguments.done" {
				next = samplingString(event["arguments"])
				emit = false
				if strings.HasPrefix(next, previous) {
					delta = next[len(previous):]
					emit = delta != ""
				}
			}
			slot.block.Extra["partialJson"] = json.RawMessage(marshalSamplingString(next))
			slot.block.Arguments = ParseStreamingJSON(next)
			if emit {
				a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: slot.contentIndex, Delta: delta})
			}
		case "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done":
			if slot == nil || slot.block.Type != "toolCall" || slot.custom == nil {
				return
			}
			close := samplingString(event["type"]) == "response.custom_tool_call_input.done"
			next := a.customInput(slot) + delta
			if close {
				next = samplingString(event["input"])
			}
			failure = a.appendCustom(slot, next, close)
		case "response.output_item.done":
			a.applyPhase(item)
			if slot == nil {
				slot = a.createSlot(key, item)
			}
			if slot == nil {
				return
			}
			block := slot.block
			switch samplingString(item["type"]) {
			case "reasoning":
				if block.Type != "thinking" {
					return
				}
				text := responsesJoinText(item["summary"], "\n\n", false)
				if text == "" {
					text = responsesJoinText(item["content"], "\n\n", false)
				}
				if text != "" {
					block.Thinking = text
				}
				signature, err := stringifyCompletionsJSON(event["item"])
				if err != nil {
					failure = err
					return
				}
				value := string(signature)
				block.ThinkingSignature = &value
				a.reasoning[responsesSlotKey(item["id"])] = block
				a.publish(AssistantMessageEvent{Type: "thinking_end", ContentIndex: slot.contentIndex, Content: block.Thinking})
			case "message":
				if block.Type != "text" {
					return
				}
				block.Text = responsesJoinText(item["content"], "", true)
				fields := map[string]json.RawMessage{"v": json.RawMessage(`1`)}
				keys := []string{"v"}
				if id, exists := item["id"]; exists {
					fields["id"] = id
					keys = append(keys, "id")
				}
				if samplingTruthy(item["phase"]) {
					fields["phase"] = item["phase"]
					keys = append(keys, "phase")
				}
				signature, _ := stringifyCompletionsJSON(marshalSamplingObject(fields, keys))
				value := string(signature)
				block.TextSignature = &value
				a.publish(AssistantMessageEvent{Type: "text_end", ContentIndex: slot.contentIndex, Content: block.Text})
			case "function_call":
				if block.Type != "toolCall" {
					return
				}
				partial, exists := block.Extra["partialJson"]
				if !exists {
					return
				}
				args := samplingString(item["arguments"])
				if args == "" {
					args = samplingString(partial)
				}
				if args == "" {
					args = "{}"
				}
				block.Arguments = ParseStreamingJSON(args)
				responsesNamespace(block, item)
				delete(block.Extra, "partialJson")
				a.publish(AssistantMessageEvent{Type: "toolcall_end", ContentIndex: slot.contentIndex, ToolCall: block})
			case "custom_tool_call":
				if block.Type != "toolCall" || slot.custom == nil {
					return
				}
				next := a.customInput(slot)
				if samplingNonNull(item["input"]) {
					next = samplingString(item["input"])
				}
				if failure = a.appendCustom(slot, next, true); failure != nil {
					return
				}
				responsesNamespace(block, item)
				delete(block.Extra, "customInput")
				slot.custom = nil
				a.publish(AssistantMessageEvent{Type: "toolcall_end", ContentIndex: slot.contentIndex, ToolCall: block})
			default:
				return
			}
			delete(a.slots, key)
		case "response.completed", "response.incomplete":
			failure = a.finalizeResponse(event["response"])
		case "error":
			failure = fmt.Errorf("Error Code %s: %s", responsesJSString(event["code"]), responsesJSString(event["message"]))
		case "response.failed":
			a.sawTerminal = true
			response, _ := samplingObject(event["response"])
			a.setMessageString("rawStopReason", &a.output.RawStopReason, response["status"])
			providerError, _ := samplingObject(response["error"])
			details, _ := samplingObject(response["incomplete_details"])
			message := "Unknown error (no error details in response)"
			if samplingTruthy(response["error"]) {
				code, text := "unknown", "no message"
				if samplingTruthy(providerError["code"]) {
					code = responsesJSString(providerError["code"])
				}
				if samplingTruthy(providerError["message"]) {
					text = responsesJSString(providerError["message"])
				}
				message = code + ": " + text
			} else if samplingTruthy(details["reason"]) {
				message = "incomplete: " + responsesJSString(details["reason"])
			}
			failure = fmt.Errorf("%s", message)
		}
	})
	return failure
}

func responsesJoinText(raw json.RawMessage, separator string, message bool) string {
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	texts := make([]string, len(parts))
	for i, part := range parts {
		value := part["text"]
		if message && samplingString(part["type"]) != "output_text" {
			value = part["refusal"]
		}
		if samplingNonNull(value) {
			texts[i] = responsesJSString(value)
		}
	}
	return strings.Join(texts, separator)
}
func responsesJSString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "undefined"
	}
	if raw[0] == '"' {
		return samplingString(raw)
	}
	if value, err := stringifyCompletionsJSON(raw); err == nil {
		return string(value)
	}
	return string(raw)
}
func (a *responsesAccumulator) setMessageString(key string, target **string, raw json.RawMessage) {
	*target = nil
	delete(a.output.Extra, key)
	if len(raw) == 0 {
		return
	}
	if string(raw) == "null" {
		if a.output.Extra == nil {
			a.output.Extra = map[string]json.RawMessage{}
		}
		a.output.Extra[key] = raw
	} else {
		value := samplingString(raw)
		*target = &value
	}
}
func (a *responsesAccumulator) finalizeResponse(raw json.RawMessage) error {
	a.sawTerminal = true
	response, _ := samplingObject(raw)
	var items []json.RawMessage
	_ = json.Unmarshal(response["output"], &items)
	for _, rawItem := range items {
		item, _ := samplingObject(rawItem)
		if samplingString(item["type"]) != "reasoning" || !samplingTruthy(item["encrypted_content"]) {
			continue
		}
		block := a.reasoning[responsesSlotKey(item["id"])]
		if block == nil || block.ThinkingSignature == nil || *block.ThinkingSignature == "" {
			continue
		}
		storedRaw := json.RawMessage(*block.ThinkingSignature)
		stored, _ := samplingObject(storedRaw)
		if samplingTruthy(stored["encrypted_content"]) {
			continue
		}
		keys := samplingObjectKeys(storedRaw)
		if _, exists := stored["encrypted_content"]; !exists {
			keys = append(keys, "encrypted_content")
		}
		stored["encrypted_content"] = item["encrypted_content"]
		signature, err := stringifyCompletionsJSON(marshalSamplingObject(stored, keys))
		if err != nil {
			return err
		}
		value := string(signature)
		block.ThinkingSignature = &value
	}
	if samplingTruthy(response["id"]) {
		a.setMessageString("responseId", &a.output.ResponseID, response["id"])
	}
	if samplingTruthy(response["usage"]) {
		fields, _ := samplingObject(response["usage"])
		input, _ := samplingObject(fields["input_tokens_details"])
		output, _ := samplingObject(fields["output_tokens_details"])
		number := func(raw json.RawMessage) float64 { value, _ := strconv.ParseFloat(string(raw), 64); return value }
		reasoning := number(output["reasoning_tokens"])
		usage := &Usage{Output: number(fields["output_tokens"]), CacheRead: number(input["cached_tokens"]), CacheWrite: number(input["cache_write_tokens"]), Reasoning: &reasoning, TotalTokens: number(fields["total_tokens"])}
		usage.Input = math.Max(0, number(fields["input_tokens"])-usage.CacheRead-usage.CacheWrite)
		observeWireUsage(usage, fields, "input_tokens", "output_tokens")
		if number(fields["input_tokens"]) < usage.CacheRead+usage.CacheWrite {
			usage.Observation.UsageObserved = false
			usage.Observation.UsageSource = "invalid"
		}
		for _, raw := range []json.RawMessage{input["cached_tokens"], input["cache_write_tokens"]} {
			if samplingNonNull(raw) && !validObservedNumber(raw) {
				usage.Observation.UsageObserved = false
				usage.Observation.UsageSource = "invalid"
			}
		}
		a.output.Usage = usage
	}
	calculateNativeUsageCost(a.model.Cost, a.output.Usage)
	if a.applyServiceTierPricing != nil {
		tier := response["service_tier"]
		if !samplingNonNull(tier) {
			tier = a.serviceTier
		}
		if a.resolveServiceTier != nil {
			tier = a.resolveServiceTier(response["service_tier"], a.serviceTier)
		}
		a.applyServiceTierPricing(a.output.Usage, tier)
	}
	status := samplingString(response["status"])
	details, _ := samplingObject(response["incomplete_details"])
	reason := samplingString(details["reason"])
	rawReason := response["status"]
	if reason != "" {
		rawReason = json.RawMessage(marshalSamplingString(responsesJSString(response["status"]) + "." + reason))
	}
	a.setMessageString("rawStopReason", &a.output.RawStopReason, rawReason)
	stop, errorMessage, err := responsesStopReason(status, reason)
	if err != nil {
		return err
	}
	a.output.StopReason = stop
	a.output.ErrorMessage = errorMessage
	if stop == "stop" {
		for _, block := range a.output.Content.Blocks.Values() {
			if block.Type == "toolCall" {
				a.output.StopReason = "toolUse"
				break
			}
		}
	}
	return nil
}
func responsesStopReason(status, reason string) (string, *string, error) {
	switch status {
	case "", "completed", "in_progress", "queued":
		return "stop", nil, nil
	case "failed", "cancelled":
		return "error", nil, nil
	case "incomplete":
		if reason == "max_output_tokens" {
			return "length", nil, nil
		}
		message := "Response incomplete without a provider reason"
		if reason != "" {
			message = "Response incomplete: " + reason
		}
		return "error", &message, nil
	default:
		return "", nil, fmt.Errorf("Unhandled stop reason: %s", status)
	}
}
func (a *responsesAccumulator) streamFailure() string {
	failure := ""
	if !a.sawTerminal {
		failure = "OpenAI Responses stream ended before a terminal response event"
	} else if a.output.StopReason == "toolUse" {
		for _, block := range a.output.Content.Blocks.Values() {
			if block.Type == "toolCall" && (block.Extra["partialJson"] != nil || block.Extra["customInput"] != nil) {
				failure = fmt.Sprintf("OpenAI Responses stream completed with an unfinished tool call: %s (%s)", block.Name, block.ID)
				break
			}
		}
	}
	return failure
}
func (a *responsesAccumulator) finish(ctx context.Context) {
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		failure := a.streamFailure()
		if failure == "" {
			if ctx.Err() != nil {
				failure = "Request was aborted"
			} else if a.output.StopReason == "pending" {
				failure = a.missingStopReason
				if failure == "" {
					failure = "OpenAI Responses stream ended without a stop reason"
				}
			} else if a.output.StopReason == "error" || a.output.StopReason == "aborted" {
				failure = "An unknown error occurred"
				if a.output.ErrorMessage != nil && *a.output.ErrorMessage != "" {
					failure = *a.output.ErrorMessage
				}
			}
		}
		if failure != "" {
			a.failLocked(failure, ctx.Err() != nil)
			return
		}
		a.ended = true
		a.publish(AssistantMessageEvent{Type: "done", Reason: a.output.StopReason, Message: a.output})
		a.stream.End()
	})
}
func (a *responsesAccumulator) fail(message string, aborted bool) {
	a.stream.Synchronize(func() {
		if !a.ended {
			a.failLocked(message, aborted)
		}
	})
}
func (a *responsesAccumulator) failLocked(message string, aborted bool) {
	for _, block := range a.output.Content.Blocks.Values() {
		delete(block.Extra, "index")
		delete(block.Extra, "partialJson")
		delete(block.Extra, "customInput")
	}
	a.output.StopReason = "error"
	if aborted {
		a.output.StopReason = "aborted"
	}
	if a.formatFailure != nil {
		message = a.formatFailure(message)
	} else if strings.Contains(message, "subscription_sharing_usage_limit_exceeded") {
		message += "\nCheck your ChatGPT usage: https://chatgpt.com/settings/usage"
	}
	a.output.ErrorMessage = &message
	a.ended = true
	a.publish(AssistantMessageEvent{Type: "error", Reason: a.output.StopReason, Error: a.output})
	a.stream.End()
}
func responsesServiceTierPricing(modelID string, usage *Usage, tier json.RawMessage) {
	multiplier := 1.0
	switch samplingString(tier) {
	case "flex":
		multiplier = 0.5
	case "priority", "fast":
		multiplier = 2
		if modelID == "gpt-5.5" {
			multiplier = 2.5
		}
	}
	if multiplier == 1 {
		return
	}
	usage.Cost.Input *= multiplier
	usage.Cost.Output *= multiplier
	usage.Cost.CacheRead *= multiplier
	usage.Cost.CacheWrite *= multiplier
	usage.Cost.Total = usage.Cost.Input + usage.Cost.Output + usage.Cost.CacheRead + usage.Cost.CacheWrite
}
