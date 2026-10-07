package llm

import (
	"encoding/json"
	nativehost "github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai/protocol"
	"github.com/2found/2ai/telemetry/export"
	"time"
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
		message, err := nativehost.ProjectMessage(raw)
		if err != nil {
			return TraceRecord{}, err
		}
		record.Messages = append(record.Messages, message)
	}
	for _, tool := range trace.Context.Tools {
		record.Tools = append(record.Tools, tool.Name)
	}
	if len(trace.Response) > 0 && string(trace.Response) != "null" {
		message, err := nativehost.ProjectMessage(trace.Response)
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

// RecordBatch exports the model-attempt events recorded by AgentCore.
func RecordBatch(batch export.Batch, sink Sink, meta Metadata) {
	if sink == nil {
		return
	}
	for _, span := range batch.Spans {
		for _, event := range span.Events {
			if event.Name != "agentray.ai.attempt" {
				continue
			}
			raw, ok := event.Attributes.Get("llm.trace").(string)
			if !ok {
				continue
			}
			record, err := ParseNative(json.RawMessage(raw), meta)
			if err != nil {
				continue
			}
			func() { defer func() { _ = recover() }(); sink.Record(record) }()
		}
	}
}
