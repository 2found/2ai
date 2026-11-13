package agentcore

import (
	"testing"
)

func TestCanonicalToolResultSupersedesOutcomeSideRecord(t *testing.T) {
	toolMessage := Message{Role: RoleTool, ToolCallID: "c1", Name: "read", Content: "done"}
	state := ReduceSession([]SessionEntry{
		{Seq: 1, Kind: EntryMessage, Message: &Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "read", Arguments: "{}"}}}},
		{Seq: 2, Kind: EntryToolOutcome, CallID: "c1", Outcome: &ToolOutcomeRecord{Message: toolMessage}},
		{Seq: 3, Kind: EntryMessage, Message: &toolMessage},
	})
	if len(state.ToolOutcomes) != 0 {
		t.Fatalf("settled side outcome remained live: %+v", state.ToolOutcomes)
	}
}

func TestRewindDropsOutcomeFromAbandonedBranch(t *testing.T) {
	state := ReduceSession([]SessionEntry{
		{Seq: 1, ID: "user", Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "start"}},
		{Seq: 2, ID: "calls", ParentID: "user", Kind: EntryMessage, Message: &Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "write", Arguments: "{}"}}}},
		{Seq: 3, Kind: EntryToolOutcome, CallID: "c1", Outcome: &ToolOutcomeRecord{Message: Message{Role: RoleTool, ToolCallID: "c1", Content: "done"}}},
		{Seq: 4, Kind: EntryLeafMove, Target: "user"},
	})
	if len(state.ToolOutcomes) != 0 {
		t.Fatalf("abandoned branch outcome leaked into active branch: %+v", state.ToolOutcomes)
	}
}

func TestOutcomeUsesExplicitIntentAnchorAcrossInterleavedBranches(t *testing.T) {
	state := ReduceSession([]SessionEntry{
		{Seq: 1, ID: "root", Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "start"}},
		{Seq: 2, ID: "calls-a", ParentID: "root", Kind: EntryMessage, Message: &Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Name: "write", Arguments: "{}"}}}},
		{Seq: 3, Kind: EntryLeafMove, Target: "root"},
		{Seq: 4, ID: "branch-b", ParentID: "root", Kind: EntryMessage, Message: &Message{Role: RoleUser, Content: "different branch"}},
		// A late completion from branch A lands after branch B's node. ParentID,
		// not raw append position, determines its ownership.
		{Seq: 5, Kind: EntryToolOutcome, ParentID: "calls-a", CallID: "a", Outcome: &ToolOutcomeRecord{Message: Message{Role: RoleTool, ToolCallID: "a", Content: "done"}}},
	})
	if len(state.ToolOutcomes) != 0 {
		t.Fatalf("interleaved abandoned outcome attached to the active branch: %+v", state.ToolOutcomes)
	}
}
