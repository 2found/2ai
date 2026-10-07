package ai

import (
	"reflect"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestToolParametersNormalizesEmptyRootWithoutMutatingCanonical(t *testing.T) {
	for _, input := range []map[string]any{nil, {}} {
		got := toolParameters(input, toolSchemaGeneric)
		if got["type"] != "object" || !reflect.DeepEqual(got["properties"], map[string]any{}) {
			t.Fatalf("normalized empty schema = %#v", got)
		}
	}

	original := map[string]any{"type": "object"}
	got := toolParameters(original, toolSchemaGeneric)
	if _, changed := original["properties"]; changed {
		t.Fatalf("canonical schema mutated: %#v", original)
	}
	if _, ok := got["properties"].(map[string]any); !ok {
		t.Fatalf("wire schema lacks object properties: %#v", got)
	}
}

func TestOpenAIResponsesToolSchemaRewritesOneOfRecursively(t *testing.T) {
	original := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"choice": map[string]any{
				"oneOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "number"},
				},
			},
			"config": map[string]any{"type": "object"},
			"free":   map[string]any{},
			"name":   map[string]any{"type": "string", "pattern": "(?<=prefix)value"},
		},
	}
	got := toolParameters(original, toolSchemaOpenAIResponses)
	props := got["properties"].(map[string]any)
	choice := props["choice"].(map[string]any)
	if _, exists := choice["oneOf"]; exists || len(choice["anyOf"].([]any)) != 2 {
		t.Fatalf("Responses union = %#v, want anyOf", choice)
	}
	config := props["config"].(map[string]any)
	if _, ok := config["properties"].(map[string]any); !ok {
		t.Fatalf("nested object lacks properties: %#v", config)
	}
	if props["free"] != true {
		t.Fatalf("empty subschema = %#v, want true", props["free"])
	}
	if name := props["name"].(map[string]any); name["pattern"] != nil {
		t.Fatalf("unsupported regex lookaround leaked: %#v", name)
	}
	canonicalChoice := original["properties"].(map[string]any)["choice"].(map[string]any)
	if _, exists := canonicalChoice["oneOf"]; !exists {
		t.Fatalf("canonical schema was mutated: %#v", original)
	}
}

func TestCloudCodeToolSchemaNormalizesSupportedSubset(t *testing.T) {
	original := map[string]any{
		"$schema":               "https://json-schema.org/draft/2020-12/schema",
		"type":                  "object",
		"additional_properties": false,
		"required":              []string{"maybe", "mode"},
		"properties": map[string]any{
			"maybe": map[string]any{
				"any_of": []any{map[string]any{"type": "string", "pattern": "^[a-z]+$"}, map[string]any{"type": "null"}},
			},
			"mode": map[string]any{"const": "safe"},
		},
	}
	got := toolParameters(original, toolSchemaCloudCodeAssist)
	if _, exists := got["$schema"]; exists {
		t.Fatalf("$schema leaked onto CCA wire: %#v", got)
	}
	if _, exists := got["additionalProperties"]; exists {
		t.Fatalf("additionalProperties leaked onto CCA wire: %#v", got)
	}
	props := got["properties"].(map[string]any)
	maybe := props["maybe"].(map[string]any)
	if maybe["type"] != "string" {
		t.Fatalf("nullable union = %#v", maybe)
	}
	if required := got["required"]; !reflect.DeepEqual(required, []any{"mode"}) {
		t.Fatalf("nullable property remains required: %#v", required)
	}
	if _, exists := maybe["pattern"]; exists || maybe["description"] == "" {
		t.Fatalf("stripped constraint was not lifted: %#v", maybe)
	}
	mode := props["mode"].(map[string]any)
	if !reflect.DeepEqual(mode["enum"], []any{"safe"}) {
		t.Fatalf("const was not converted to enum: %#v", mode)
	}
	if _, stillCanonical := original["$schema"]; !stillCanonical {
		t.Fatalf("canonical schema was mutated: %#v", original)
	}
}

func TestCloudCodeToolSchemaFallsBackForResidualCombinerOrReference(t *testing.T) {
	for _, input := range []map[string]any{
		{"type": "object", "properties": map[string]any{"x": map[string]any{"anyOf": []any{
			map[string]any{"type": "string"}, map[string]any{"type": "number"},
		}}}},
		{"type": "object", "$ref": "#/$defs/input", "$defs": map[string]any{}},
	} {
		got := toolParameters(input, toolSchemaCloudCodeAssist)
		if !reflect.DeepEqual(got, emptyObjectToolSchema()) {
			t.Fatalf("fallback = %#v, want open object", got)
		}
	}
}

func TestProviderEncodersUseTheirToolSchemaDialect(t *testing.T) {
	schema := protocol.ToolSchema{Name: "run", Parameters: map[string]any{
		"type": "object", "properties": map[string]any{
			"choice": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "number"},
			}},
		},
	}}
	req := protocol.ChatRequest{Model: "m", Tools: []protocol.ToolSchema{schema}}

	responses := NewOpenAIResponsesProvider("key", "").encode(req)
	choice := responses.Tools[0].Parameters["properties"].(map[string]any)["choice"].(map[string]any)
	if _, ok := choice["anyOf"]; !ok {
		t.Fatalf("Responses parameters were not normalized: %#v", choice)
	}
	codex := NewCodexProvider().encode(req)
	choice = codex.Tools[0].Parameters["properties"].(map[string]any)["choice"].(map[string]any)
	if _, ok := choice["anyOf"]; !ok {
		t.Fatalf("Codex parameters were not normalized: %#v", choice)
	}
	antigravity := NewAntigravityProvider().encode(req)
	if got := antigravity.Request.Tools[0].FunctionDeclarations[0].Parameters; !reflect.DeepEqual(got, emptyObjectToolSchema()) {
		t.Fatalf("Antigravity residual-combiner fallback = %#v", got)
	}
}

func TestAntigravityMapsReasoningEffortToNativeControls(t *testing.T) {
	p := NewAntigravityProvider()
	low := p.encode(protocol.ChatRequest{Model: "gemini-2.5-pro", MaxTokens: 2048, ReasoningEffort: "low"})
	lowConfig := low.Request.GenerationConfig
	if lowConfig.ThinkingConfig == nil || lowConfig.ThinkingConfig.ThinkingBudget != 4096 ||
		lowConfig.MaxOutputTokens != 6144 {
		t.Fatalf("low budget config = %+v", lowConfig)
	}
	high := p.encode(protocol.ChatRequest{Model: "gemini-2.5-pro", MaxTokens: 2048, ReasoningEffort: "high"})
	if high.Request.GenerationConfig.ThinkingConfig.ThinkingBudget != 16384 ||
		high.Request.GenerationConfig.MaxOutputTokens != 18432 {
		t.Fatalf("high budget config = %+v", high.Request.GenerationConfig)
	}
	level := p.encode(protocol.ChatRequest{Model: "gemini-3-pro", MaxTokens: 2048, ReasoningEffort: "xhigh"})
	if level.Request.GenerationConfig.ThinkingConfig.ThinkingLevel != "HIGH" ||
		level.Request.GenerationConfig.MaxOutputTokens != 2048 {
		t.Fatalf("Gemini 3 level config = %+v", level.Request.GenerationConfig)
	}
	off := p.encode(protocol.ChatRequest{Model: "gemini-3-pro", ReasoningEffort: "off"})
	if off.Request.GenerationConfig.ThinkingConfig != nil {
		t.Fatalf("off reasoning emitted thinking config: %+v", off.Request.GenerationConfig.ThinkingConfig)
	}
}

func TestAntigravityReasoningBudgetHonorsProviderCeiling(t *testing.T) {
	p := NewAntigravityProvider()
	gemini := p.encode(protocol.ChatRequest{Model: "gemini-2.5-pro", MaxTokens: 60_000, ReasoningEffort: "high"})
	if got := gemini.Request.GenerationConfig.MaxOutputTokens; got != 65_536 {
		t.Fatalf("Gemini max output = %d, want 65536", got)
	}
	claude := p.encode(protocol.ChatRequest{Model: "claude-sonnet", MaxTokens: 60_000, ReasoningEffort: "high"})
	if got := claude.Request.GenerationConfig.MaxOutputTokens; got != 64_000 {
		t.Fatalf("Claude max output = %d, want 64000", got)
	}
}
