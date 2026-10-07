package ai

import (
	"encoding/json"
	"slices"

	"github.com/2found/2ai/jsonjs"
)

// Stream snapshots copy the typed graph without serializing it. Serialization
// is an observation made by the caller, not a prerequisite for consuming Pi's
// events: live numbers may be nonfinite and extension JSON may be unreadable.
// All mutable public fields detach; immutable transcript encoding metadata can
// remain shared. Memoization preserves repeated message/list/block/usage objects.
type streamSnapshot struct {
	values   jsonjs.ValueCloner
	messages map[*Message]*Message
	blocks   map[*ContentBlock]*ContentBlock
	lists    map[*BlockList]*BlockList
	usages   map[*Usage]*Usage
}

func snapshotAssistantEvent(event AssistantMessageEvent) AssistantMessageEvent {
	copy := streamSnapshot{messages: map[*Message]*Message{}, blocks: map[*ContentBlock]*ContentBlock{}, usages: map[*Usage]*Usage{}}
	event.Partial = copy.message(event.Partial)
	event.Message = copy.message(event.Message)
	event.Error = copy.message(event.Error)
	event.ToolCall = copy.block(event.ToolCall)
	event.Extra = snapshotJSONFields(event.Extra)
	return event
}

func (s *streamSnapshot) message(source *Message) *Message {
	if source == nil {
		return nil
	}
	if copy, ok := s.messages[source]; ok {
		return copy
	}
	copy := *source
	s.messages[source] = &copy
	copy.Content.Text = copyStreamValue(source.Content.Text)
	copy.Content.Blocks = s.blockList(source.Content.Blocks)
	copy.Sections = slices.Clone(source.Sections)
	for i, section := range source.Sections {
		copy.Sections[i].Value = copyStreamValue(section.Value)
	}
	copy.ToolsAdded = slices.Clone(source.ToolsAdded)
	for i, tool := range source.ToolsAdded {
		copy.ToolsAdded[i].Parameters = slices.Clone(tool.Parameters)
		copy.ToolsAdded[i].ConstrainedSampling = slices.Clone(tool.ConstrainedSampling)
		copy.ToolsAdded[i].Extra = snapshotJSONFields(tool.Extra)
	}
	copy.ToolsRemoved = slices.Clone(source.ToolsRemoved)
	copy.ResponseModel = copyStreamValue(source.ResponseModel)
	copy.ResponseID = copyStreamValue(source.ResponseID)
	copy.ProviderThinkingLevel = copyStreamValue(source.ProviderThinkingLevel)
	copy.ThinkingLevel = copyStreamValue(source.ThinkingLevel)
	copy.ErrorMessage = copyStreamValue(source.ErrorMessage)
	copy.RawStopReason = copyStreamValue(source.RawStopReason)
	copy.EndTurn = copyStreamValue(source.EndTurn)
	copy.Diagnostics = slices.Clone(source.Diagnostics)
	copy.Deferred = slices.Clone(source.Deferred)
	copy.Details = s.values.Clone(source.Details)
	copy.NestedCalls = slices.Clone(source.NestedCalls)
	copy.Extra = snapshotJSONFields(source.Extra)
	copy.Usage = s.usage(source.Usage)
	return &copy
}

func (s *streamSnapshot) blockList(source *BlockList) *BlockList {
	if source == nil {
		return nil
	}
	if copy, ok := s.lists[source]; ok {
		return copy
	}
	if s.lists == nil {
		s.lists = map[*BlockList]*BlockList{}
	}
	copy := NewBlockList()
	s.lists[source] = copy
	for _, block := range source.Values() {
		copy.Append(s.block(block))
	}
	return copy
}

func (s *streamSnapshot) block(source *ContentBlock) *ContentBlock {
	if source == nil {
		return nil
	}
	if copy, ok := s.blocks[source]; ok {
		return copy
	}
	copy := *source
	s.blocks[source] = &copy
	copy.TextSignature = copyStreamValue(source.TextSignature)
	copy.ThinkingSignature = copyStreamValue(source.ThinkingSignature)
	copy.ThoughtSignature = copyStreamValue(source.ThoughtSignature)
	copy.Namespace = copyStreamValue(source.Namespace)
	copy.Redacted = copyStreamValue(source.Redacted)
	copy.Arguments = slices.Clone(source.Arguments)
	copy.Extra = snapshotJSONFields(source.Extra)
	return &copy
}

func (s *streamSnapshot) usage(source *Usage) *Usage {
	if source == nil {
		return nil
	}
	if copy, ok := s.usages[source]; ok {
		return copy
	}
	copy := *source
	s.usages[source] = &copy
	copy.CacheWrite1h = copyStreamValue(source.CacheWrite1h)
	copy.Reasoning = copyStreamValue(source.Reasoning)
	return &copy
}

func copyStreamValue[T any](source *T) *T {
	if source == nil {
		return nil
	}
	copy := *source
	return &copy
}

func snapshotJSONFields(source map[string]json.RawMessage) map[string]json.RawMessage {
	if source == nil {
		return nil
	}
	copy := make(map[string]json.RawMessage, len(source))
	for key, value := range source {
		copy[key] = slices.Clone(value)
	}
	return copy
}
