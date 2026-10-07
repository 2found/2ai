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
	"github.com/2found/2ai/telemetry"
	"github.com/2found/2ai/telemetry/export"
	"github.com/2found/2ai/telemetry/llm"
)

func TestNativeTelemetryIncludesFailedFallbackAndPreservesCheckpoint(t *testing.T) {
	signature := "opaque-native-signature"
	script := ai.ScriptedStream(
		ai.Message{Role: "assistant", StopReason: "toolUse", Usage: &ai.Usage{Input: 3, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}, Content: ai.BlockContent(
			ai.ContentBlock{Type: "thinking", Thinking: "private", ThinkingSignature: &signature},
			ai.ContentBlock{Type: "toolCall", ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)},
		)},
		ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "finished"}), Usage: &ai.Usage{Input: 5, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}},
	)
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{
		{Model: json.RawMessage(`{"id":"unavailable"}`), Stream: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
			return nil, errors.New("admission failed")
		}},
		{Model: json.RawMessage(`{"id":"failed"}`), Stream: ai.ScriptedStream(ai.Message{Role: "assistant", StopReason: "error", Content: ai.BlockContent(), Usage: &ai.Usage{Input: 2, Observation: ai.UsageObservation{UsageObserved: true, UsageSource: "explicit"}}})},
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
		llm.RecordBatch(batch, llm.SinkFunc(func(record llm.TraceRecord) {
			records = append(records, record)
			if len(record.Messages) > 0 {
				record.Messages[0].Content = "sink-corruption"
			}
			panic("sink unavailable") // must not stop delivery of other attempts
		}), llm.Metadata{TraceID: "test-run"})
	})
	safe, err := telemetry.NewSafeObserver(telemetry.SafeObserverOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := telemetry.WithSafeObserver(context.Background(), safe, telemetry.SafeExecution{ExecutionID: "native-run"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "run echo"}}, Telemetry: backend})
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
	projection := safe.Drain(256)
	attempts, observed := 0, 0
	input := 0.0
	for _, r := range projection {
		if r.Kind == "attempt_settled" {
			attempts++
			if r.ProducerAdmitted != (attempts != 1) {
				t.Fatal("admission failure presented as admitted producer")
			}
			if r.Accounting.UsageObserved {
				observed++
				input += r.Accounting.Tokens.Input
			}
			if r.Accounting.KnownCostUSD != nil {
				t.Fatal("default native cost became known")
			}
		}
		if r.Kind == "reconciliation" && (r.Reconciliation == nil || r.Reconciliation.ReportedTokens.Input != 10 || r.Reconciliation.Additive) {
			t.Fatal("parent aggregate double-counted")
		}
	}
	if attempts != 4 || observed != 3 || input != 10 {
		t.Fatalf("safe accounting attempts=%d observed=%d input=%g", attempts, observed, input)
	}
	raw, _ := json.Marshal(projection)
	for _, canary := range []string{signature, "private", "finished", "admission failed", "sink-corruption", "run echo"} {
		if strings.Contains(string(raw), canary) {
			t.Fatalf("safe projection leaked %q", canary)
		}
	}
	t.Logf("AC-TELEMETRY native actual JSON: %s", raw)
}
