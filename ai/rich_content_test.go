package ai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func richToolMessage(id string) protocol.Message {
	return protocol.Message{
		Role: protocol.RoleTool, ToolCallID: id, Name: "eval", Content: "plot ready",
		ContentParts: []protocol.ContentPart{{
			Type: protocol.ContentPartImage, MIMEType: "image/png", Data: "aW1hZ2U=", Detail: "high",
		}},
	}
}

func TestOpenAIChatHoistsRichToolImagesAfterResultBatch(t *testing.T) {
	p := NewOpenAIProvider("sk", "", DefaultCompat())
	req := protocol.ChatRequest{Model: "vision", Messages: []protocol.Message{
		richToolMessage("c1"), richToolMessage("c2"),
	}}
	body := p.encode(req)
	if len(body.Messages) != 3 {
		t.Fatalf("messages = %d, want two tool results plus one image message: %+v", len(body.Messages), body.Messages)
	}
	for i := 0; i < 2; i++ {
		text, ok := body.Messages[i].Content.(string)
		if !ok || body.Messages[i].Role != "tool" || !strings.Contains(text, "attached") {
			t.Fatalf("tool message %d = %+v", i, body.Messages[i])
		}
	}
	parts, ok := body.Messages[2].Content.([]oaiContentPart)
	if !ok || body.Messages[2].Role != "user" || len(parts) != 3 {
		t.Fatalf("hoisted image message = %+v", body.Messages[2])
	}
	if parts[1].ImageURL == nil || parts[1].ImageURL.URL != "data:image/png;base64,aW1hZ2U=" || parts[1].ImageURL.Detail != "high" {
		t.Fatalf("first image part = %+v", parts[1])
	}
}

func TestResponsesAndCodexEncodeRichToolOutputNatively(t *testing.T) {
	message := richToolMessage("c1")

	responses := NewOpenAIResponsesProvider("sk", "").encode(protocol.ChatRequest{
		Model: "vision", Messages: []protocol.Message{message},
	})
	rparts, ok := responses.Input[0].Output.([]responsesContent)
	if !ok || len(rparts) != 2 || rparts[1].Type != "input_image" || rparts[1].Detail != "high" {
		t.Fatalf("Responses output = %#v", responses.Input[0].Output)
	}

	codex := NewCodexProvider().encode(protocol.ChatRequest{
		Model: "vision", Messages: []protocol.Message{message},
	})
	cparts, ok := codex.Input[0].Output.([]codexContent)
	if !ok || len(cparts) != 2 || cparts[1].Type != "input_image" || cparts[1].ImageURL != "data:image/png;base64,aW1hZ2U=" {
		t.Fatalf("Codex output = %#v", codex.Input[0].Output)
	}

	// Ensure the interface-valued output remains a native JSON content array,
	// not an escaped JSON string.
	raw, err := json.Marshal(responses)
	if err != nil {
		t.Fatalf("marshal Responses: %v", err)
	}
	if !strings.Contains(string(raw), `"output":[{"type":"input_text"`) || !strings.Contains(string(raw), `"type":"input_image"`) {
		t.Fatalf("Responses rich output wire = %s", raw)
	}
}

func TestAnthropicEncodesRichToolOutputInsideToolResult(t *testing.T) {
	p := NewAnthropicProvider("sk", "")
	body := p.encode(protocol.ChatRequest{Model: "vision", Messages: []protocol.Message{richToolMessage("c1")}})
	if len(body.Messages) != 1 || len(body.Messages[0].Content) != 1 {
		t.Fatalf("Anthropic messages = %+v", body.Messages)
	}
	content, ok := body.Messages[0].Content[0].Content.([]antContentBlock)
	if !ok || len(content) != 2 || content[1].Type != "image" || content[1].Source == nil {
		t.Fatalf("Anthropic tool result content = %#v", body.Messages[0].Content[0].Content)
	}
	if content[1].Source.MediaType != "image/png" || content[1].Source.Data != "aW1hZ2U=" {
		t.Fatalf("Anthropic image source = %+v", content[1].Source)
	}
}
