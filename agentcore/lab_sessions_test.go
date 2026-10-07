package agentcore

import "testing"

func TestFoldStepsKeepsChildAndRepeatedCallResultsSeparate(t *testing.T) {
	call := ToolCall{ID: "reused", Name: "read", Arguments: "{}"}
	result := func(text string) Message { return Message{Role: RoleTool, ToolCallID: "reused", Content: text} }
	records := []TurnRecord{
		{SessionKey: "parent", ToolCalls: []ToolCall{call}},
		{SessionKey: "child", Depth: 1, ToolCalls: []ToolCall{call}},
		{SessionKey: "parent", Messages: []Message{result("parent first")}, ToolCalls: []ToolCall{call}},
		{SessionKey: "child", Depth: 1, Messages: []Message{result("child result")}},
		{SessionKey: "parent", Messages: []Message{result("parent first"), result("parent second")}},
	}
	steps := FoldSteps(records)
	for index, want := range map[int]string{0: "parent first", 1: "child result", 2: "parent second"} {
		if steps[index].ToolCalls[0].Result != want {
			t.Fatalf("step %d got another invocation's result: %+v", index, steps[index])
		}
	}
	if steps[1].SessionKey != "child" || steps[1].Depth != 1 {
		t.Fatal("child identity lost")
	}
}

func TestFoldStepsNativeGatesComeFromTheirOwnInvocation(t *testing.T) {
	call := ToolCall{ID: "reused", Name: "write", Arguments: "{}"}
	records := []TurnRecord{
		{SessionKey: "parent", NativeTrace: []byte(`{"context":{"messages":[]}}`), ToolCalls: []ToolCall{call}, ToolGates: []ToolGate{{CallID: "reused", Allowed: true}}},
		{SessionKey: "child", NativeTrace: []byte(`{"context":{"messages":[]}}`), ToolCalls: []ToolCall{call}},
		{SessionKey: "parent", NativeTrace: []byte(`{"context":{"messages":[{"role":"toolResult","toolCallId":"reused","details":{"trace":{"call_id":"reused","allowed":false,"reason":"parent denied"}}}]}}`)},
		{SessionKey: "child", NativeTrace: []byte(`{"context":{"messages":[{"role":"toolResult","toolCallId":"reused","details":{"trace":{"call_id":"reused","allowed":true}}}]}}`)},
	}
	records = ApplyLiveGates(records, []ToolTrace{{CallID: "reused", Allowed: false, Reason: "wrong global gate"}})
	steps := FoldSteps(records)
	if steps[0].ToolCalls[0].Allowed || steps[0].ToolCalls[0].Error != "parent denied" || !steps[1].ToolCalls[0].Allowed || steps[1].ToolCalls[0].Error != "" {
		t.Fatalf("native gates crossed invocation boundaries: %+v %+v", steps[0].ToolCalls, steps[1].ToolCalls)
	}
}
