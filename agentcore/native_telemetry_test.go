package agentcore_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/telemetry/export"
	"github.com/2found/2ai/telemetry/llm"
)

func TestNativeTelemetryIncludesFailedFallbackAndPreservesCheckpoint(t *testing.T) {
	signature := "opaque-native-signature"
	script := ai.ScriptedStream(
		ai.Message{Role: "assistant", StopReason: "toolUse", Usage: &ai.Usage{Input: 3}, Content: ai.BlockContent(
			ai.ContentBlock{Type: "thinking", Thinking: "private", ThinkingSignature: &signature},
			ai.ContentBlock{Type: "toolCall", ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)},
		)},
		ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"}), Usage: &ai.Usage{Input: 5}},
	)
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"unavailable"}`), Stream: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
			return nil, errors.New("admission failed")
		}},
		{Model: json.RawMessage(`{"id":"failed"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "error", Content: ai.BlockContent(), Usage: &ai.Usage{Input: 2}})},
		{Model: json.RawMessage(`{"id":"test"}`), Stream: script},
	}}
	retry := agentcore.RetryPolicy{MaxAttempts: 1}
	effects := 0
	a, err := agentcore.New(agentcore.Config{Model: "test", NativeProvider: provider, Retry: &retry,
		Tools: agentcore.NewToolSet(agentcore.StringTool{ToolName: "echo", Properties: agentcore.StringProperties(), Required: []string{}, Execute: func(context.Context, map[string]string) (string, error) { effects++; return "ok", nil }}), Policy: agentcore.NewAllowList("echo"),
	})
	if err != nil {
		t.Fatal(err)
	}
	var records []llm.TraceRecord
	backend := export.New(func(batch export.Batch) {
		if len(batch.SpanTimings) != len(batch.Spans) {
			t.Error("settled native spans lost individual timings")
		}
		for _, span := range batch.Spans {
			for _, event := range span.Events {
				if event.Name == "agentray.ai.attempt" {
					if ms, ok := event.Attributes.Get("ai.duration_ms").(int64); !ok || ms < 0 {
						t.Error("attempt lacks numeric latency without private trace parsing")
					}
					if number, ok := event.Attributes.Get("ai.attempt").(int); !ok || number < 1 {
						t.Error("attempt lacks numeric identity")
					}
					if kind, ok := event.Attributes.Get("ai.failure_kind").(string); !ok || (kind != "none" && kind != "unknown") {
						t.Error("attempt lacks bounded failure diagnostic")
					}
					if _, ok := event.Attributes.Get("ai.output_committed").(bool); !ok {
						t.Error("attempt lacks replay fence diagnostic")
					}
				}
			}
		}
		llm.RecordBatch(batch, llm.SinkFunc(func(record llm.TraceRecord) {
			records = append(records, record)
			if len(record.Messages) > 0 {
				record.Messages[0].Content = "sink-corruption"
			}
			panic("sink unavailable") // must not stop delivery of other attempts
		}), llm.Metadata{TraceID: "test-run"})
	})
	result, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "run echo"}}, Telemetry: backend})
	if err != nil {
		t.Fatal(err)
	}
	var models []string
	for _, record := range records {
		models = append(models, record.Model)
	}
	if !reflect.DeepEqual(models, []string{"unavailable", "failed", "test", "test"}) || records[0].Err != "admission failed" || records[1].StopReason != "error" {
		t.Fatalf("attempts lost/duplicated: %+v", records)
	}
	if result.Final != "finished" || result.Usage.InputTokens != 10 || effects != 1 {
		t.Fatalf("telemetry changed execution: %+v effects=%d", result, effects)
	}
	if !strings.Contains(string(result.NativeState), signature) || strings.Contains(string(result.NativeState), "sink-corruption") {
		t.Fatalf("telemetry changed checkpoint: %s", result.NativeState)
	}
}
