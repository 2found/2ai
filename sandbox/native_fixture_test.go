package sandbox

import (
	"encoding/json"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

// Translate scripted test responses only; execution remains in the native
// engine, including nested tools, schema checks and permission gates.
func sandboxNativeFixture(responses ...agentcore.ChatResponse) *ai.FallbackProvider {
	var messages []ai.Message
	for _, response := range responses {
		var blocks []ai.ContentBlock
		if response.Message.Content != "" {
			blocks = append(blocks, ai.ContentBlock{Type: "text", Text: response.Message.Content})
		}
		for _, call := range response.Message.ToolCalls {
			blocks = append(blocks, ai.ContentBlock{Type: "toolCall", ID: call.ID, Name: call.Name, Arguments: json.RawMessage(call.Arguments)})
		}
		stop := "stop"
		if len(response.Message.ToolCalls) > 0 {
			stop = "toolUse"
		}
		messages = append(messages, ai.Message{Role: "assistant", Content: ai.BlockContent(blocks...), StopReason: stop})
	}
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"faux","contextWindow":128000}`), Stream: ai.ScriptedStream(messages...)}}}
}
