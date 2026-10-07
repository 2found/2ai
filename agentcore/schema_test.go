package agentcore

import (
	"context"
	"strings"
	"testing"
)

func verdictOutputSchema() *OutputSchema {
	return &OutputSchema{
		Name: "verdict",
		Schema: map[string]any{
			"type":     "object",
			"required": []string{"verdict"},
			"properties": map[string]any{
				"verdict": map[string]any{"type": "string", "enum": []string{"allow", "deny"}},
			},
			"additionalProperties": false,
		},
	}
}

func TestOutputSchemaRejectsInvalidSchemaAtBuild(t *testing.T) {
	_, err := New(Config{
		Provider: NewFauxProvider(AssistantText(`{"verdict":"allow"}`)),
		Model:    "test",
		Policy:   DenyAll{},
		OutputSchema: &OutputSchema{Schema: map[string]any{
			"type":    "string",
			"pattern": "[",
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "output schema does not compile") {
		t.Fatalf("New error = %v, want schema compilation error", err)
	}
}

func TestOutputSchemaAcceptsMatchingFinalAnswer(t *testing.T) {
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText(`{"verdict":"allow"}`)),
		Model:        "test",
		Policy:       DenyAll{},
		OutputSchema: verdictOutputSchema(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "classify")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != `{"verdict":"allow"}` {
		t.Fatalf("Final = %q", res.Final)
	}
}

func TestOutputSchemaRejectsMismatchedFinalAnswer(t *testing.T) {
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText(`{"verdict":42}`)),
		Model:        "test",
		Policy:       DenyAll{},
		OutputSchema: verdictOutputSchema(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "classify")
	if err == nil || !strings.Contains(err.Error(), "structured output does not match schema") {
		t.Fatalf("Prompt error = %v, want output validation error", err)
	}
	if res.Final != "" {
		t.Fatalf("invalid answer was accepted as Final: %q", res.Final)
	}
}

func TestOutputSchemaRejectsNonJSONFinalAnswer(t *testing.T) {
	agent, err := New(Config{
		NativeProvider: scriptedNativeProvider(AssistantText("allow")),
		Model:        "test",
		Policy:       DenyAll{},
		OutputSchema: verdictOutputSchema(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = agent.Prompt(context.Background(), "classify")
	if err == nil || !strings.Contains(err.Error(), "structured output is not valid JSON") {
		t.Fatalf("Prompt error = %v, want JSON validation error", err)
	}
}

func TestOutputSchemaDoesNotRejectToolCallTurns(t *testing.T) {
	provider := scriptedNativeProvider(
		AssistantToolCall("call-1", "lookup", `{}`),
		AssistantText(`{"verdict":"deny"}`),
	)
	agent, err := New(Config{
		NativeProvider: provider,
		Model:        "test",
		Tools:        NewToolSet(schemaLookupTool{}),
		Policy:       NewAllowList("lookup"),
		OutputSchema: verdictOutputSchema(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "classify after lookup")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != `{"verdict":"deny"}` || len(res.Tools) != 1 {
		t.Fatalf("unexpected result: final=%q tools=%d", res.Final, len(res.Tools))
	}
}

type schemaLookupTool struct{}

func (schemaLookupTool) Name() string { return "lookup" }

func (schemaLookupTool) Schema() ToolSchema {
	return ToolSchema{Name: "lookup", Description: "look something up", Parameters: map[string]any{"type": "object"}}
}

func (schemaLookupTool) Run(context.Context, string) (string, error) { return "found", nil }
