package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"strconv"
)

func samplingNonNull(raw json.RawMessage) bool {
	return len(raw) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// This accumulator consumes decoded Chat Completions chunks. Transport owns
// callbacks, cancellation and error normalization; all published payload
// mutations here share the original stream's lock and message identity.
type completionsAccumulator struct {
	model           completionsModel
	compat          OpenAICompletionsCompat
	grammar         map[string]string
	stream          *AssistantMessageEventStream
	push            func(AssistantMessageEvent)
	output          *Message
	text, thinking  *ContentBlock
	byIndex         map[float64]*ContentBlock
	byID            map[string]*ContentBlock
	custom          map[*ContentBlock]*completionsCustomInput
	details         []*completionsReasoningDetail
	hasFinishReason bool
	ended           bool
}

type completionsCustomInput struct {
	Property   string                     `json:"property"`
	JSONBuffer GrammarToolInputJSONBuffer `json:"jsonBuffer"`
}

// Undefined fields still occupy an insertion position in JS objects. Keep
// their keys while coalescing metadata; JSON signatures omit their values.
type completionsReasoningDetail struct {
	fields map[string]json.RawMessage
	keys   []string
}

func (d *completionsReasoningDetail) MarshalJSON() ([]byte, error) {
	keys := make([]string, 0, len(d.keys))
	for _, key := range d.keys {
		if len(d.fields[key]) > 0 {
			keys = append(keys, key)
		}
	}
	return marshalSamplingObject(d.fields, keys), nil
}

type completionsCostRates struct {
	Input, Output, CacheRead, CacheWrite float64
}
type completionsCost struct {
	completionsCostRates
	Tiers []struct {
		completionsCostRates
		InputTokensAbove float64
	}
}

func newCompletionsAccumulator(model completionsModel, compat OpenAICompletionsCompat, grammar map[string]string, stream *AssistantMessageEventStream, timestamp int64) *completionsAccumulator {
	return &completionsAccumulator{
		model: model, compat: compat, grammar: grammar, stream: stream, push: stream.Push,
		output:  &Message{Role: "assistant", Content: BlockContent(), API: model.API, Provider: model.Provider, Model: model.ID, Usage: &Usage{}, StopReason: "pending", Timestamp: timestamp},
		byIndex: map[float64]*ContentBlock{}, byID: map[string]*ContentBlock{}, custom: map[*ContentBlock]*completionsCustomInput{},
	}
}

func (a *completionsAccumulator) publish(event AssistantMessageEvent) {
	if event.Type != "done" && event.Type != "error" {
		event.Partial = a.output
	}
	a.push(event)
}

func (a *completionsAccumulator) start() {
	a.stream.Synchronize(func() { a.publish(AssistantMessageEvent{Type: "start"}) })
}

func (a *completionsAccumulator) index(block *ContentBlock) int {
	for i, candidate := range a.output.Content.Blocks.Values() {
		if block == candidate {
			return i
		}
	}
	return -1
}

func (a *completionsAccumulator) ensureThinking(signature string) *ContentBlock {
	if a.thinking == nil {
		a.thinking = &ContentBlock{Type: "thinking", ThinkingSignature: &signature}
		a.output.Content.Blocks.Append(a.thinking)
		a.publish(AssistantMessageEvent{Type: "thinking_start", ContentIndex: a.index(a.thinking)})
	}
	return a.thinking
}

func (a *completionsAccumulator) initCustom(block *ContentBlock) {
	property, ok := a.grammar[block.Name]
	if !ok {
		property = "input"
	}
	input := &completionsCustomInput{Property: property}
	a.custom[block] = input
	block.Arguments = marshalSamplingObject(map[string]json.RawMessage{property: json.RawMessage(`""`)}, []string{property})
	block.Extra["customInput"], _ = json.Marshal(input)
	delete(block.Extra, "partialArgs")
}

func (a *completionsAccumulator) ensureTool(call map[string]json.RawMessage) *ContentBlock {
	function, _ := samplingObject(call["function"])
	custom, _ := samplingObject(call["custom"])
	name := samplingString(function["name"])
	if !samplingNonNull(function["name"]) {
		name = samplingString(custom["name"])
	}
	id := samplingString(call["id"])
	index, err := strconv.ParseFloat(string(call["index"]), 64)
	hasIndex := err == nil
	var block *ContentBlock
	if hasIndex {
		block = a.byIndex[index]
	}
	if block == nil && id != "" {
		block = a.byID[id]
	}
	isCustom := samplingTruthy(call["custom"]) && !samplingTruthy(call["function"])
	if block == nil {
		block = &ContentBlock{Type: "toolCall", ID: id, Name: name, Arguments: json.RawMessage(`{}`), Extra: map[string]json.RawMessage{"partialArgs": json.RawMessage(`""`)}}
		if isCustom {
			a.initCustom(block)
		}
		if hasIndex {
			block.Extra["streamIndex"] = call["index"]
			a.byIndex[index] = block
		}
		if id != "" {
			a.byID[id] = block
		}
		a.output.Content.Blocks.Append(block)
		a.publish(AssistantMessageEvent{Type: "toolcall_start", ContentIndex: a.index(block)})
	}
	if hasIndex && block.Extra["streamIndex"] == nil {
		block.Extra["streamIndex"] = call["index"]
		a.byIndex[index] = block
	}
	if id != "" {
		a.byID[id] = block
	}
	if block.Name == "" && name != "" {
		block.Name = name
	}
	if isCustom && a.custom[block] == nil {
		a.initCustom(block)
	}
	return block
}

func (a *completionsAccumulator) appendCustom(block *ContentBlock, suffix string, close bool) (*string, error) {
	custom := a.custom[block]
	if custom == nil {
		return nil, nil
	}
	args, _ := samplingObject(block.Arguments)
	next := samplingString(args[custom.Property]) + suffix
	delta, err := AppendGrammarToolInputJSONDelta(&custom.JSONBuffer, custom.Property, next, close)
	if err != nil {
		return nil, err
	}
	block.Arguments = marshalSamplingObject(map[string]json.RawMessage{custom.Property: json.RawMessage(marshalSamplingString(next))}, []string{custom.Property})
	block.Extra["customInput"], _ = json.Marshal(custom)
	return delta, nil
}

func (a *completionsAccumulator) chunk(raw json.RawMessage) error {
	var failure error
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		chunk, ok := samplingObject(raw)
		if !ok {
			return
		}
		if a.output.ResponseID == nil || *a.output.ResponseID == "" {
			a.output.ResponseID = nil
			delete(a.output.Extra, "responseId")
			if rawID, exists := chunk["id"]; exists {
				if samplingNonNull(rawID) {
					id := samplingString(rawID)
					a.output.ResponseID = &id
				} else {
					if a.output.Extra == nil {
						a.output.Extra = map[string]json.RawMessage{}
					}
					a.output.Extra["responseId"] = rawID
				}
			}
		}
		if model := samplingString(chunk["model"]); model != "" && model != a.model.ID && a.output.ResponseModel == nil {
			a.output.ResponseModel = &model
		}
		if samplingTruthy(chunk["usage"]) {
			a.output.Usage = a.parseUsage(chunk["usage"])
		}
		var choices []json.RawMessage
		if json.Unmarshal(chunk["choices"], &choices) != nil || len(choices) == 0 {
			return
		}
		choice, _ := samplingObject(choices[0])
		if !samplingTruthy(chunk["usage"]) && samplingTruthy(choice["usage"]) {
			a.output.Usage = a.parseUsage(choice["usage"])
		}
		if reason := samplingString(choice["finish_reason"]); reason != "" {
			a.output.RawStopReason = &reason
			a.hasFinishReason = true
			switch reason {
			case "stop", "end":
				a.output.StopReason = "stop"
			case "length":
				a.output.StopReason = "length"
			case "function_call", "tool_calls":
				a.output.StopReason = "toolUse"
			default:
				a.output.StopReason = "error"
				message := "Provider finish_reason: " + reason
				a.output.ErrorMessage = &message
			}
		}
		delta, _ := samplingObject(choice["delta"])
		if text := samplingString(delta["content"]); text != "" {
			if a.text == nil {
				a.text = &ContentBlock{Type: "text"}
				a.output.Content.Blocks.Append(a.text)
				a.publish(AssistantMessageEvent{Type: "text_start", ContentIndex: a.index(a.text)})
			}
			a.text.Text += text
			a.publish(AssistantMessageEvent{Type: "text_delta", ContentIndex: a.index(a.text), Delta: text})
		}
		for _, field := range []string{"reasoning_content", "reasoning", "reasoning_text"} {
			if text := samplingString(delta[field]); text != "" {
				signature := field
				if a.model.Provider == "opencode-go" && field == "reasoning" {
					signature = "reasoning_content"
				}
				block := a.ensureThinking(signature)
				block.Thinking += text
				a.publish(AssistantMessageEvent{Type: "thinking_delta", ContentIndex: a.index(block), Delta: text})
				break
			}
		}
		var calls []json.RawMessage
		_ = json.Unmarshal(delta["tool_calls"], &calls)
		for _, rawCall := range calls {
			call, _ := samplingObject(rawCall)
			block := a.ensureTool(call)
			if id := samplingString(call["id"]); block.ID == "" && id != "" {
				block.ID = id
				a.byID[id] = block
			}
			function, _ := samplingObject(call["function"])
			custom, _ := samplingObject(call["custom"])
			text := ""
			if args := samplingString(function["arguments"]); args != "" {
				text = args
				partial := samplingString(block.Extra["partialArgs"]) + args
				block.Extra["partialArgs"] = json.RawMessage(marshalSamplingString(partial))
				block.Arguments = ParseStreamingJSON(partial)
			} else if input := samplingString(custom["input"]); input != "" {
				appended, err := a.appendCustom(block, input, false)
				if err != nil {
					failure = err
					return
				}
				if appended != nil {
					text = *appended
				}
			}
			a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: a.index(block), Delta: text})
		}
		var details []json.RawMessage
		_ = json.Unmarshal(delta["reasoning_details"], &details)
		for _, detail := range details {
			if !isCompletionsReasoningDetail(detail) {
				continue
			}
			a.ensureThinking("")
			a.appendDetail(detail)
		}
	})
	return failure
}

func (a *completionsAccumulator) appendDetail(raw json.RawMessage) {
	detail, _ := samplingObject(raw)
	kind := samplingString(detail["type"])
	if len(a.details) > 0 && (kind == "reasoning.text" || kind == "reasoning.summary") {
		lastDetail := a.details[len(a.details)-1]
		last := lastDetail.fields
		if samplingString(last["type"]) == kind {
			set := func(key string, value json.RawMessage) {
				if _, exists := last[key]; !exists {
					lastDetail.keys = append(lastDetail.keys, key)
				}
				last[key] = value
			}
			field := "text"
			if kind == "reasoning.summary" {
				field = "summary"
			}
			set(field, json.RawMessage(marshalSamplingString(samplingString(last[field])+samplingString(detail[field]))))
			if field == "text" && !samplingTruthy(last["signature"]) {
				set("signature", detail["signature"])
			}
			if !samplingNonNull(last["id"]) {
				set("id", detail["id"])
			}
			if !samplingTruthy(last["format"]) {
				set("format", detail["format"])
			}
			if !samplingNonNull(last["index"]) {
				set("index", detail["index"])
			}
			return
		}
	}
	a.details = append(a.details, &completionsReasoningDetail{fields: detail, keys: samplingObjectKeys(raw)})
}

func (a *completionsAccumulator) applyDetails(block *ContentBlock) {
	if block.Type == "thinking" && a.details != nil {
		raw, _ := json.Marshal(a.details)
		encoded, _ := stringifyCompletionsJSON(raw)
		signature := string(encoded)
		block.ThinkingSignature = &signature
	}
}

func (a *completionsAccumulator) clean(block *ContentBlock) {
	for _, key := range []string{"index", "partialArgs", "customInput", "streamIndex"} {
		delete(block.Extra, key)
	}
	delete(a.custom, block)
}

// finish follows iterator EOF, including block-end events before checking the
// finish reason or abort signal. fail handles midstream failures without ends.
func (a *completionsAccumulator) finish(ctx context.Context) {
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		for i, block := range a.output.Content.Blocks.Values() {
			switch block.Type {
			case "text":
				a.publish(AssistantMessageEvent{Type: "text_end", ContentIndex: i, Content: block.Text})
			case "thinking":
				a.applyDetails(block)
				a.publish(AssistantMessageEvent{Type: "thinking_end", ContentIndex: i, Content: block.Thinking})
			case "toolCall":
				if a.custom[block] != nil {
					delta, err := a.appendCustom(block, "", true)
					if err != nil {
						a.failLocked(err.Error(), ctx.Err() != nil)
						return
					}
					if delta != nil {
						a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: i, Delta: *delta})
					}
				} else {
					block.Arguments = ParseStreamingJSON(samplingString(block.Extra["partialArgs"]))
				}
				a.clean(block)
				a.publish(AssistantMessageEvent{Type: "toolcall_end", ContentIndex: i, ToolCall: block})
			}
		}
		if ctx.Err() != nil || a.output.StopReason == "aborted" {
			a.failLocked("Request was aborted", ctx.Err() != nil)
			return
		}
		if !a.hasFinishReason && !a.compat.SupportsFinishReason {
			a.output.StopReason = "stop"
			for _, block := range a.output.Content.Blocks.Values() {
				if block.Type == "toolCall" {
					a.output.StopReason = "toolUse"
					break
				}
			}
		}
		if a.output.StopReason == "error" {
			message := "Provider returned an error stop reason"
			if a.output.ErrorMessage != nil && *a.output.ErrorMessage != "" {
				message = *a.output.ErrorMessage
			}
			a.failLocked(message, false)
			return
		}
		if (a.compat.SupportsFinishReason && !a.hasFinishReason) || a.output.StopReason == "pending" {
			a.failLocked("Stream ended without finish_reason", false)
			return
		}
		a.ended = true
		a.publish(AssistantMessageEvent{Type: "done", Reason: a.output.StopReason, Message: a.output})
		a.stream.End()
	})
}

func (a *completionsAccumulator) fail(message string, aborted bool) {
	a.stream.Synchronize(func() {
		if !a.ended {
			a.failLocked(message, aborted)
		}
	})
}
func (a *completionsAccumulator) failLocked(message string, aborted bool) {
	for _, block := range a.output.Content.Blocks.Values() {
		a.applyDetails(block)
		a.clean(block)
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

func (a *completionsAccumulator) parseUsage(raw json.RawMessage) *Usage {
	fields, _ := samplingObject(raw)
	prompt, _ := samplingObject(fields["prompt_tokens_details"])
	completion, _ := samplingObject(fields["completion_tokens_details"])
	number := func(raw json.RawMessage) float64 { value, _ := strconv.ParseFloat(string(raw), 64); return value }
	cached := prompt["cached_tokens"]
	if !samplingNonNull(cached) {
		cached = fields["prompt_cache_hit_tokens"]
	}
	if !samplingNonNull(cached) {
		cached = fields["cached_tokens"]
	}
	reasoning := number(completion["reasoning_tokens"])
	usage := &Usage{Output: number(fields["completion_tokens"]), CacheRead: number(cached), CacheWrite: number(prompt["cache_write_tokens"]), Reasoning: &reasoning}
	usage.Input = math.Max(0, number(fields["prompt_tokens"])-usage.CacheRead-usage.CacheWrite)
	usage.TotalTokens = usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
	calculateNativeUsageCost(a.model.Cost, usage)
	return usage
}

func calculateNativeUsageCost(cost completionsCost, usage *Usage) {
	rates := cost.completionsCostRates
	input := usage.Input + usage.CacheRead + usage.CacheWrite
	threshold := -1.0
	for _, tier := range cost.Tiers {
		if input > tier.InputTokensAbove && tier.InputTokensAbove > threshold {
			rates = tier.completionsCostRates
			threshold = tier.InputTokensAbove
		}
	}
	// Explicit rounding prevents Go from fusing these multiplications with the
	// later total on architectures with FMA. JavaScript rounds each operation.
	usage.Cost.Input = float64(float64(rates.Input/1e6) * usage.Input)
	usage.Cost.Output = float64(float64(rates.Output/1e6) * usage.Output)
	usage.Cost.CacheRead = float64(float64(rates.CacheRead/1e6) * usage.CacheRead)
	longWrite := 0.0
	if usage.CacheWrite1h != nil {
		longWrite = *usage.CacheWrite1h
	}
	shortCost := float64(rates.CacheWrite * (usage.CacheWrite - longWrite))
	longCost := float64(float64(rates.Input*2) * longWrite)
	usage.Cost.CacheWrite = float64((shortCost + longCost) / 1e6)
	usage.Cost.Total = usage.Cost.Input + usage.Cost.Output + usage.Cost.CacheRead + usage.Cost.CacheWrite
}
