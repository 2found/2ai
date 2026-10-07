package subagent_test

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/subagent"
)

const fruitSchema = `{"type":"object","required":["fruit"],"properties":{"fruit":{"type":"string"}},"additionalProperties":false}`

// spawnResult extracts the spawn_subagent tool result from a parent transcript.
func spawnResult(t *testing.T, msgs []agentcore.Message) string {
	t.Helper()
	for _, m := range msgs {
		if m.Role == agentcore.RoleTool && m.Name == subagent.ToolSpawnSubagent {
			return m.Content
		}
	}
	t.Fatal("no spawn_subagent tool result in parent transcript")
	return ""
}

// TestSubagentOutputSchemaAccept: a child whose final answer satisfies
// output_schema returns it to the parent verbatim, and the child's task
// carried the schema instruction.
func TestSubagentOutputSchemaAccept(t *testing.T) {
	agent, provider := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent,
			`{"task":"name a fruit","output_schema":`+fruitSchema+`}`),
		nativeAnswer(`{"fruit":"banana"}`),
		nativeAnswer("child said banana"),
	)
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := spawnResult(t, res.Messages); got != `{"fruit":"banana"}` {
		t.Fatalf("spawn result = %q", got)
	}
	// The child's seeded task carried the schema and the bare-JSON contract.
	childReq := provider.Recorded[1]
	var sawSchema bool
	for _, m := range childReq.Messages {
		if strings.Contains(m.Content, `"required":["fruit"]`) && strings.Contains(m.Content, "JSON Schema") {
			sawSchema = true
		}
	}
	if !sawSchema {
		t.Fatalf("child task missing output_schema instruction: %+v", childReq.Messages)
	}
}

// TestSubagentOutputSchemaRejectRetry: a schema-violating answer re-opens the
// child once with the validation error; the corrected answer is what the
// parent receives.
func TestSubagentOutputSchemaRejectRetry(t *testing.T) {
	agent, provider := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent,
			`{"task":"name a fruit","output_schema":`+fruitSchema+`}`),
		// First child answer: prose, not the required JSON object.
		nativeAnswer("the fruit is banana"),
		// Re-opened child sees its transcript + the error and corrects.
		nativeAnswer(`{"fruit":"banana"}`),
		nativeAnswer("child said banana"),
	)
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := spawnResult(t, res.Messages); got != `{"fruit":"banana"}` {
		t.Fatalf("spawn result = %q, want the corrected JSON", got)
	}
	// The retry request carried the rejected answer and the validation error.
	retryReq := provider.Recorded[2]
	var sawRejected, sawError bool
	for _, m := range retryReq.Messages {
		if strings.Contains(m.Content, "the fruit is banana") {
			sawRejected = true
		}
		if strings.Contains(m.Content, "failed output_schema validation") {
			sawError = true
		}
	}
	if !sawRejected || !sawError {
		t.Fatalf("retry child missing transcript/error (rejected=%v error=%v): %+v", sawRejected, sawError, retryReq.Messages)
	}
}

// TestSubagentOutputSchemaRetryAlsoFails: when the retry also violates the
// schema the parent gets the raw answer marked validation-failed — not an
// error, not silent acceptance.
func TestSubagentOutputSchemaRetryAlsoFails(t *testing.T) {
	agent, _ := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent,
			`{"task":"name a fruit","output_schema":`+fruitSchema+`}`),
		nativeAnswer("the fruit is banana"),
		nativeAnswer(`{"fruit":42}`),
		nativeAnswer("done"),
	)
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	got := spawnResult(t, res.Messages)
	if !strings.Contains(got, `{"fruit":42}`) || !strings.Contains(got, "[validation failed:") {
		t.Fatalf("spawn result = %q, want raw answer + validation-failed note", got)
	}
}

// TestSubagentOutputSchemaAbsentUnchanged: without output_schema the child is
// not given the JSON contract and its answer passes through untouched.
func TestSubagentOutputSchemaAbsentUnchanged(t *testing.T) {
	agent, provider := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"name a fruit"}`),
		nativeAnswer("banana, obviously"),
		nativeAnswer("done"),
	)
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := spawnResult(t, res.Messages); got != "banana, obviously" {
		t.Fatalf("spawn result = %q", got)
	}
	for _, m := range provider.Recorded[1].Messages {
		if strings.Contains(m.Content, "JSON Schema") {
			t.Fatalf("schema instruction leaked into schema-less spawn: %q", m.Content)
		}
	}
}

// TestSubagentOutputSchemaInvalidArg: a malformed schema fails the call before
// any child runs — no provider call is spent on the child.
func TestSubagentOutputSchemaInvalidArg(t *testing.T) {
	agent, provider := subagentAgent(t, &subagent.Plugin{},
		AssistantToolCall("c1", subagent.ToolSpawnSubagent,
			`{"task":"name a fruit","output_schema":{"type":"bogus-type"}}`),
		nativeAnswer("spawn failed as expected"),
	)
	res, err := agent.Prompt(context.Background(), "delegate")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := spawnResult(t, res.Messages); !strings.Contains(got, "output_schema") {
		t.Fatalf("spawn result = %q, want an output_schema error", got)
	}
	// Only the parent's two turns hit the provider — no child call.
	if len(provider.Recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2 (no child run)", len(provider.Recorded))
	}
}

func TestDelegateSchemaCorrectionIncludesRejectedAnswerAndUsage(t *testing.T) {
	calls := 0
	settings := subagent.Plugin{Delegates: []subagent.Delegate{{Name: "Writer", Run: func(_ context.Context, task string, _ agentcore.StreamSink) (string, agentcore.Usage, error) {
		calls++
		if calls == 1 {
			return "the fruit is banana", agentcore.Usage{InputTokens: 7}, nil
		}
		if !strings.Contains(task, "Rejected answer:\nthe fruit is banana") || !strings.Contains(task, "failed output_schema validation") {
			t.Errorf("delegate correction lost rejected answer: %s", task)
		}
		return `{"fruit":"banana"}`, agentcore.Usage{InputTokens: 11}, nil
	}}}}
	agent, _ := subagentAgent(t, &settings,
		AssistantToolCall("c1", subagent.ToolSpawnSubagent, `{"task":"name a fruit","agent":"Writer","output_schema":`+fruitSchema+`}`), nativeAnswer("done"))
	result, err := agent.Prompt(context.Background(), "delegate")
	if err != nil || calls != 2 || result.Usage.InputTokens != 18 {
		t.Fatalf("delegate correction: calls=%d usage=%+v err=%v", calls, result.Usage, err)
	}
	if got := spawnResult(t, result.Messages); got != `{"fruit":"banana"}` {
		t.Fatal(got)
	}
}
