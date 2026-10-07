package subagent_test

import (
	"context"
	"encoding/json"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai"
)

func nativeAnswer(text string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})}
}
func nativeCall(id, name, args string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: id, Name: name, Arguments: json.RawMessage(args)})}
}
func nativeProvider(stream ai.StreamFn) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: stream}}}
}

type recordedView struct {
	Messages []agentcore.Message
	Tools    []agentcore.ToolSchema
}
type nativeScript struct {
	Provider *ai.FallbackProvider
	Recorded []recordedView
}

// Display projections are used only for assertions. The engine and scripted
// provider exchange the original native messages throughout the test.
func newNativeScript(replies ...ai.Message) *nativeScript {
	s := &nativeScript{}
	script := ai.ScriptedStream(replies...)
	s.Provider = nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		var recorded recordedView
		for _, msg := range view.Messages() {
			raw, err := json.Marshal(msg)
			if err != nil {
				return nil, err
			}
			projected, err := host.ProjectMessage(raw)
			if err != nil {
				return nil, err
			}
			recorded.Messages = append(recorded.Messages, projected)
		}
		tools, err := json.Marshal(ai.GetCurrentTools(view.Messages()))
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(tools, &recorded.Tools); err != nil {
			return nil, err
		}
		s.Recorded = append(s.Recorded, recorded)
		return script(ctx, model, view, options)
	})
	return s
}
