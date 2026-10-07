package subagent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/2found/2ai/agentcore"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

// output_schema support: the caller hands the child a JSON Schema and the
// plugin guarantees the answer the parent sees either satisfies it or is
// explicitly marked as having failed validation. Constraint is prompt-level —
// a forked child cannot carry the provider's structured-output seam (Fork
// deliberately drops outputSchema, and no per-run setter exists), so the
// schema is spelled out in the task and the plugin validates the answer
// itself. agentcore's own validateArgs stays unexported and shallow by
// design, so validation uses a real JSON Schema implementation.

// compileOutputSchema decodes the raw output_schema argument into a compiled
// validator. A nil schema disables the feature entirely; a present-but-invalid
// schema fails the call before any child runs, so a misspelled schema never
// burns a spawn.
func compileOutputSchema(raw json.RawMessage) (*jsonschema.Schema, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	doc, err := jsonschema.UnmarshalJSON(strings.NewReader(string(raw)))
	if err != nil {
		return nil, fmt.Errorf("output_schema is not valid JSON: %w", err)
	}
	obj, ok := doc.(map[string]any)
	if !ok || len(obj) == 0 {
		return nil, fmt.Errorf("output_schema must be a non-empty JSON Schema object")
	}
	c := jsonschema.NewCompiler()
	if err := c.AddResource("https://subagent.local/output_schema.json", doc); err != nil {
		return nil, fmt.Errorf("output_schema: %w", err)
	}
	s, err := c.Compile("https://subagent.local/output_schema.json")
	if err != nil {
		return nil, fmt.Errorf("output_schema does not compile: %w", err)
	}
	return s, nil
}

// outputSchemaInstruction is appended to the child's task when output_schema
// is set. It is deliberately blunt about the contract — the child must end
// with a bare JSON value, no prose, no fence — because the plugin validates
// the final answer verbatim.
func outputSchemaInstruction(raw json.RawMessage) string {
	return "\n\nYour final answer MUST be a single JSON value matching this JSON Schema — " +
		"no prose, no markdown fence, nothing before or after it:\n" + string(raw)
}

// stripCodeFence removes a single surrounding ``` / ```json fence. Models
// wrap JSON in fences often enough that validating the raw text would fail
// every fenced-but-valid answer.
func stripCodeFence(s string) string {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") {
		return t
	}
	// Drop the opening fence line (``` or ```json) and a closing fence.
	if i := strings.IndexByte(t, '\n'); i >= 0 {
		t = t[i+1:]
	} else {
		return s
	}
	if j := strings.LastIndex(t, "```"); j >= 0 {
		t = t[:j]
	}
	return strings.TrimSpace(t)
}

// validateOutput checks the child's final answer against the compiled schema.
// The returned error is written for the model to read on retry — it names the
// failure, not internals.
func validateOutput(answer string, schema *jsonschema.Schema) error {
	doc, err := jsonschema.UnmarshalJSON(strings.NewReader(stripCodeFence(answer)))
	if err != nil {
		return fmt.Errorf("final answer is not valid JSON: %w", err)
	}
	if err := schema.Validate(doc); err != nil {
		return fmt.Errorf("final answer does not match output_schema: %w", err)
	}
	return nil
}

// correctionMessage adds host-authored input to the child's native checkpoint.
// Provider history is never rebuilt from display messages.
func correctionMessage(validationErr error) agentcore.Message {
	return agentcore.Message{Role: agentcore.RoleUser,
		Content: "Your previous final answer failed output_schema validation: " + validationErr.Error() +
			"\nReply with ONLY a corrected JSON value matching the schema — no prose, no markdown fence.",
	}
}
