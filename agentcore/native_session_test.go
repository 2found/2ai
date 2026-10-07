package agentcore_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	nativehost "github.com/2found/2ai/agentcore/host"
)

func TestPiSessionRecoveryKeepsNativeMessages(t *testing.T) {
	initial := agentcore.SessionEntry{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"system","content":"system","timestamp":1}],"model":{"id":"test"},"tools":[],"thinkingLevel":"off","isStreaming":false}`}
	event := `{"type":"message_end","message":{"role":"custom","timestamp":2,"content":[{"type":"image","mimeType":"image/png","data":"abc"}],"extension":{"signature":"opaque > & \u2028","opaque":9007199254740993}}}`
	state, err := agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{initial, {Kind: agentcore.EntryPiEvent, Content: event}})
	if err != nil {
		t.Fatal(err)
	}
	var restored map[string]json.RawMessage
	_ = json.Unmarshal(state, &restored)
	if _, ok := restored["isStreaming"]; ok {
		t.Fatal("restored runtime-owned state")
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(restored["messages"], &messages); err != nil || len(messages) != 2 {
		t.Fatalf("native messages lost: %s, %v", state, err)
	}
	if !strings.Contains(string(messages[1]["extension"]), "9007199254740993") {
		t.Fatalf("journal recovery rounded an opaque number: %s", state)
	}
	if string(messages[1]["role"]) != `"custom"` || len(messages[1]["extension"]) == 0 || len(messages[1]["content"]) == 0 {
		t.Fatalf("message projected through legacy schema: %s", state)
	}
}

func TestPiSessionRecoveryCannotMaskUnsettledEffectsWithReusedCallID(t *testing.T) {
	entries := []agentcore.SessionEntry{
		{Kind: agentcore.EntryPiState, Content: `{"messages":[],"tools":[]}`},
		{Kind: agentcore.EntryPiEffectStart, CallID: "reused", Content: `{"effectId":"first"}`},
		{Kind: agentcore.EntryPiEffectStart, CallID: "reused", Content: `{"effectId":"second"}`},
		{Kind: agentcore.EntryPiEffectDone, CallID: "reused", Content: `{"effectId":"second"}`},
		{Kind: agentcore.EntryPiState, Content: `{"messages":[],"tools":[]}`},
	}
	_, err := agentcore.RecoverNativeTranscript(entries)
	if !errors.Is(err, nativehost.ErrUnsettledEffect) || !strings.Contains(err.Error(), "reused/first") {
		t.Fatalf("lost first effect behind second receipt or checkpoint: %v", err)
	}
}

func TestPiSessionRecoveryRejectsLegacyAndUnplacedResults(t *testing.T) {
	for name, entries := range map[string][]agentcore.SessionEntry{
		"legacy":         {{Kind: agentcore.EntryMessage}},
		"empty":          nil,
		"broken":         {{Kind: agentcore.EntryPiState, Content: `{`}},
		"missing result": {{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"assistant","content":[{"type":"toolCall","id":"call","name":"write","arguments":{}}]}]}`}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := agentcore.RecoverNativeTranscript(entries); err == nil {
				t.Fatal("accepted unsafe recovery")
			}
		})
	}
}

func TestPiSessionRecoversExactStartedResult(t *testing.T) {
	result := `{"role":"toolResult","toolCallId":"call","toolName":"write","content":[{"type":"text","text":"ok"}],"timestamp":1234,"details":{"signature":"opaque"},"isError":false}`
	initial := agentcore.SessionEntry{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"assistant","content":[{"type":"toolCall","id":"call","name":"write"}]}]}`}
	start := agentcore.SessionEntry{Kind: agentcore.EntryPiEvent, Content: `{"type":"message_start","message":` + result + `}`}
	end := agentcore.SessionEntry{Kind: agentcore.EntryPiEvent, Content: `{"type":"message_end","message":` + result + `}`}
	for _, finished := range []bool{false, true} {
		entries := []agentcore.SessionEntry{initial, start}
		if finished {
			entries = append(entries, end)
		}
		raw, err := agentcore.RecoverNativeTranscript(entries)
		if err != nil {
			t.Fatal(err)
		}
		var state struct{ Messages []json.RawMessage }
		if err := json.Unmarshal(raw, &state); err != nil || len(state.Messages) != 2 || string(state.Messages[1]) != result {
			t.Fatalf("native result lost or duplicated (finished=%v): %s, %v", finished, raw, err)
		}
	}
}

func TestPiSessionRecoveryRejectsCrossBatchResult(t *testing.T) {
	_, err := agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{{Kind: agentcore.EntryPiState, Content: `{"messages":[
		{"role":"assistant","content":[{"type":"toolCall","id":"reused","name":"write"}]},
		{"role":"assistant","content":[{"type":"toolCall","id":"reused","name":"write"}]},
		{"role":"toolResult","toolCallId":"reused","content":[]}
	]}`}})
	if !errors.Is(err, nativehost.ErrUnsettledEffect) {
		t.Fatalf("later result masked earlier missing result: %v", err)
	}
}

func TestPiSessionRecoveryRequiresNativeResult(t *testing.T) {
	_, err := agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{
		{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"assistant","content":[{"type":"toolCall","id":"call","name":"write"}]}]}`},
		{Kind: agentcore.EntryPiEffectStart, CallID: "call", Content: `{"effectId":"effect"}`},
		{Kind: agentcore.EntryPiEffectDone, CallID: "call", Content: `{"effectId":"effect","result":{"content":[]}}`},
		{Kind: agentcore.EntryPiEvent, Content: `{"type":"tool_execution_end","toolCallId":"call","toolName":"write","result":{"content":[]},"isError":false}`},
	})
	if !errors.Is(err, nativehost.ErrUnsettledEffect) {
		t.Fatalf("invented native message from an execution receipt: %v", err)
	}
}

func TestPiSessionRecoveryPreservesRejectedCallsWithoutInventingEffects(t *testing.T) {
	for _, stop := range []string{"error", "aborted"} {
		t.Run(stop, func(t *testing.T) {
			message := `{"role":"assistant","stopReason":"` + stop + `","content":[{"type":"toolCall","id":"reused","name":"write","arguments":{"partial":true}}],"extension":{"signature":"keep"}}`
			initial := agentcore.SessionEntry{Kind: agentcore.EntryPiState, Content: `{"messages":[` + message + `],"tools":[]}`}
			state, err := agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{initial})
			if err != nil {
				t.Fatal(err)
			}
			var restored struct{ Messages []json.RawMessage }
			if err := json.Unmarshal(state, &restored); err != nil {
				t.Fatal(err)
			}
			if len(restored.Messages) != 1 || string(restored.Messages[0]) != message {
				t.Fatalf("rejected transcript changed: %s", state)
			}
			_, err = agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{initial, {Kind: agentcore.EntryPiEffectStart, CallID: "reused", Content: `{"effectId":"write-effect"}`}})
			if !errors.Is(err, nativehost.ErrUnsettledEffect) {
				t.Fatalf("rejected message masked a physical effect: %v", err)
			}
			_, err = agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"assistant","stopReason":"toolUse","content":[{"type":"toolCall","id":"reused","name":"write","arguments":{}}]},` + message + `]}`}})
			if !errors.Is(err, nativehost.ErrUnsettledEffect) {
				t.Fatalf("rejected message masked an earlier batch: %v", err)
			}
		})
	}
	// Unlike error/aborted, Pi emits tool-result failures for length stops.
	_, err := agentcore.RecoverNativeTranscript([]agentcore.SessionEntry{{Kind: agentcore.EntryPiState, Content: `{"messages":[{"role":"assistant","stopReason":"length","content":[{"type":"toolCall","id":"truncated","name":"write"}]}]}`}})
	if !errors.Is(err, nativehost.ErrUnsettledEffect) {
		t.Fatalf("length message lost its native result obligation: %v", err)
	}
}
