package ai

import (
	"encoding/json"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func toolChoiceRequest(choice protocol.ToolChoice, parallel *bool) protocol.ChatRequest {
	return protocol.ChatRequest{
		Model: "m", ToolChoice: choice, ParallelToolCalls: parallel,
		Tools: []protocol.ToolSchema{{
			Name: "query", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
	}
}

func TestOpenAIToolChoiceWireDialects(t *testing.T) {
	parallel := false
	named := protocol.ToolChoice{Mode: protocol.ToolChoiceNamed, Name: "query"}

	chat := NewOpenAIProvider("key", "", DefaultCompat()).encode(toolChoiceRequest(named, &parallel))
	choice, ok := chat.ToolChoice.(map[string]any)
	if !ok || choice["type"] != "function" || choice["function"].(map[string]any)["name"] != "query" {
		t.Fatalf("chat tool_choice = %#v", chat.ToolChoice)
	}
	if chat.ParallelToolCalls == nil || *chat.ParallelToolCalls {
		t.Fatalf("chat parallel_tool_calls = %v", chat.ParallelToolCalls)
	}

	responses := NewOpenAIResponsesProvider("key", "").encode(toolChoiceRequest(named, &parallel))
	rchoice, ok := responses.ToolChoice.(map[string]any)
	if !ok || rchoice["type"] != "function" || rchoice["name"] != "query" {
		t.Fatalf("responses tool_choice = %#v", responses.ToolChoice)
	}
	if responses.ParallelToolCalls == nil || *responses.ParallelToolCalls {
		t.Fatalf("responses parallel_tool_calls = %v", responses.ParallelToolCalls)
	}

	codex := NewCodexProvider().encode(toolChoiceRequest(named, &parallel))
	cchoice, ok := codex.ToolChoice.(map[string]any)
	if !ok || cchoice["name"] != "query" || codex.ParallelToolCalls == nil || *codex.ParallelToolCalls {
		t.Fatalf("codex controls = choice=%#v parallel=%v", codex.ToolChoice, codex.ParallelToolCalls)
	}
}

func TestAnthropicToolChoiceAndParallelWire(t *testing.T) {
	parallel := false
	req := toolChoiceRequest(protocol.ToolChoice{Mode: protocol.ToolChoiceRequired}, &parallel)
	body := NewAnthropicProvider("key", "").encode(req)
	if body.ToolChoice == nil || body.ToolChoice.Type != "any" ||
		body.ToolChoice.DisableParallelToolUse == nil || !*body.ToolChoice.DisableParallelToolUse {
		t.Fatalf("anthropic tool_choice = %+v", body.ToolChoice)
	}

	parallel = true
	req.ToolChoice = protocol.ToolChoice{Mode: protocol.ToolChoiceNamed, Name: "query"}
	body = NewAnthropicProvider("key", "").encode(req)
	if body.ToolChoice == nil || body.ToolChoice.Type != "tool" || body.ToolChoice.Name != "query" ||
		body.ToolChoice.DisableParallelToolUse == nil || *body.ToolChoice.DisableParallelToolUse {
		t.Fatalf("anthropic named choice = %+v", body.ToolChoice)
	}
}

func TestAntigravityToolChoiceWire(t *testing.T) {
	p := NewAntigravityProvider()
	p.tok = OAuthToken{ProjectID: "p"}
	named := p.encode(toolChoiceRequest(protocol.ToolChoice{Mode: protocol.ToolChoiceNamed, Name: "query"}, nil))
	config := named.Request.ToolConfig.FunctionCallingConfig
	if config.Mode != "ANY" || len(config.AllowedFunctionNames) != 1 || config.AllowedFunctionNames[0] != "query" {
		t.Fatalf("named function calling config = %+v", config)
	}

	none := p.encode(toolChoiceRequest(protocol.ToolChoice{Mode: protocol.ToolChoiceNone}, nil))
	if got := none.Request.ToolConfig.FunctionCallingConfig.Mode; got != "NONE" {
		t.Fatalf("none mode = %q", got)
	}
}

func TestToolControlsOmittedByDefaultAndWithoutTools(t *testing.T) {
	parallel := true
	req := protocol.ChatRequest{
		Model: "m", ToolChoice: protocol.ToolChoice{Mode: protocol.ToolChoiceRequired}, ParallelToolCalls: &parallel,
	}
	for name, body := range map[string]any{
		"chat":      NewOpenAIProvider("key", "", DefaultCompat()).encode(req),
		"responses": NewOpenAIResponsesProvider("key", "").encode(req),
		"codex":     NewCodexProvider().encode(req),
		"anthropic": NewAnthropicProvider("key", "").encode(req),
	} {
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatal(err)
		}
		if _, ok := decoded["tool_choice"]; ok {
			t.Fatalf("%s emitted tool_choice without tools: %s", name, raw)
		}
		if _, ok := decoded["parallel_tool_calls"]; ok {
			t.Fatalf("%s emitted parallel_tool_calls without tools: %s", name, raw)
		}
	}
}
