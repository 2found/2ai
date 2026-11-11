package llm

import (
	"encoding/json"
	"time"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
	"github.com/2found/2ai/telemetry"
	"github.com/2found/2ai/telemetry/export"
)

type Metadata struct {
	TraceID, SessionKey string
	Depth               int
	PricingKnown        bool
}

// ParseNative projects native telemetry for storage/display. NativeTrace keeps
// the original request and response; projections must never restore history.
func ParseNative(raw json.RawMessage, meta Metadata) (TraceRecord, error) {
	var trace struct {
		Attempt                 *struct{ PricingKnown *bool }
		StartedAtMs, DurationMs int64
		Model                   struct{ ID, Provider string }
		Context                 struct {
			SystemPrompt string
			Messages     []json.RawMessage
			Tools        []struct{ Name string }
		}
		Response json.RawMessage
		Error    *struct{ Name, Message string }
	}
	if err := json.Unmarshal(raw, &trace); err != nil {
		return TraceRecord{}, err
	}
	known := meta.PricingKnown
	if trace.Attempt != nil && trace.Attempt.PricingKnown != nil {
		known = *trace.Attempt.PricingKnown
	}
	runID, session := meta.TraceID, meta.SessionKey
	if session == "" {
		session = runID
	}
	record := TraceRecord{TraceID: runID, SessionKey: session, Depth: meta.Depth, Timestamp: time.UnixMilli(trace.StartedAtMs), Provider: trace.Model.Provider, Model: trace.Model.ID, LatencyMS: trace.DurationMs, Streamed: true, NativeTrace: append(json.RawMessage{}, raw...)}
	if trace.Context.SystemPrompt != "" {
		record.Messages = append(record.Messages, protocol.Message{Role: protocol.RoleSystem, Content: trace.Context.SystemPrompt})
	}
	for _, raw := range trace.Context.Messages {
		message, err := ai.ProjectMessage(raw)
		if err != nil {
			return TraceRecord{}, err
		}
		record.Messages = append(record.Messages, message)
	}
	for _, tool := range trace.Context.Tools {
		record.Tools = append(record.Tools, tool.Name)
	}
	if len(trace.Response) > 0 && string(trace.Response) != "null" {
		message, err := ai.ProjectMessage(trace.Response)
		if err != nil {
			return TraceRecord{}, err
		}
		record.Response, record.ToolCalls = message.Content, message.ToolCalls
		if message.Usage != nil {
			record.Usage = *message.Usage
			u := record.Usage
			record.Usage.CostUnpriced = u.CostUnpriced || (!known && u.InputTokens+u.OutputTokens+u.CacheReadTokens+u.CacheWriteTokens > 0)
		}
		var terminal struct{ StopReason, ErrorMessage string }
		_ = json.Unmarshal(trace.Response, &terminal)
		record.StopReason, record.Err = terminal.StopReason, terminal.ErrorMessage
	}
	if trace.Error != nil {
		record.Err = trace.Error.Message
	}
	return record, nil
}

// RecordBatch exports the model-attempt events recorded by AgentCore. An
// attempt with a captured "llm.trace" payload becomes a full record; without
// one (payload capture is opt-in) the record is built from the attempt's
// operational attributes: model, usage, cost, latency and outcome, but no
// messages.
func RecordBatch(batch export.Batch, sink Sink, meta Metadata) {
	if sink == nil {
		return
	}
	timings := map[int]telemetry.SpanTiming{}
	for _, timing := range batch.SpanTimings {
		timings[timing.SpanID] = timing
	}
	for _, span := range batch.Spans {
		for i, event := range span.Events {
			if event.Name != "agentray.ai.attempt" {
				continue
			}
			var record TraceRecord
			if raw, ok := event.Attributes.Get("llm.trace").(string); ok {
				parsed, err := ParseNative(json.RawMessage(raw), meta)
				if err != nil {
					continue
				}
				record = parsed
			} else {
				var at time.Time
				if timing := timings[span.ID]; i < len(timing.EventTimes) {
					at = timing.EventTimes[i]
				}
				record = operationalRecord(event.Attributes, at, meta)
			}
			func() { defer func() { _ = recover() }(); sink.Record(record) }()
		}
	}
}

func operationalRecord(attributes telemetry.Attributes, at time.Time, meta Metadata) TraceRecord {
	text := func(name string) string { value, _ := attributes.Get(name).(string); return value }
	count := func(name string) int {
		switch value := attributes.Get(name).(type) {
		case int64:
			return int(value)
		case int:
			return value
		case float64:
			return int(value)
		}
		return 0
	}
	session := meta.SessionKey
	if session == "" {
		session = meta.TraceID
	}
	latency := int64(count("ai.duration_ms"))
	record := TraceRecord{TraceID: meta.TraceID, SessionKey: session, Depth: meta.Depth, Provider: text("gen_ai.provider.name"), Model: text("gen_ai.request.model"), LatencyMS: latency, Streamed: true, Err: text("error.type")}
	if !at.IsZero() {
		record.Timestamp = at.Add(-time.Duration(latency) * time.Millisecond)
	}
	if reasons, ok := attributes.Get("gen_ai.response.finish_reasons").([]string); ok && len(reasons) > 0 {
		record.StopReason = reasons[0]
	}
	read, write := count("gen_ai.usage.cache_read.input_tokens"), count("gen_ai.usage.cache_creation.input_tokens")
	cost, priced := attributes.Get("ai.cost_usd").(float64)
	record.Usage = protocol.Usage{InputTokens: max(count("gen_ai.usage.input_tokens")-read-write, 0), OutputTokens: count("gen_ai.usage.output_tokens"),
		CacheReadTokens: read, CacheWriteTokens: write, ReasoningTokens: count("gen_ai.usage.reasoning.output_tokens"), CostUSD: cost}
	u := record.Usage
	record.Usage.CostUnpriced = (!priced || !meta.PricingKnown) && u.InputTokens+u.OutputTokens+u.CacheReadTokens+u.CacheWriteTokens > 0
	return record
}
