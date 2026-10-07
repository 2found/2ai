package ai

import (
	"bytes"
	"encoding/json"
	"reflect"

	"github.com/2found/2ai/jsonjs"
)

// The protocol's live objects map to stable native pointers. Encoding metadata
// retains missing/null/non-string wire fields without rejecting an event that
// the source accepts. Changing a projected native field replaces its wire value.
type piMessagesTypedAdapter struct {
	messages map[*Object]*Message
	blocks   map[*Object]*ContentBlock
	lists    map[*Array]*BlockList
	usages   map[*Object]*Usage
}

func newPiMessagesTypedAdapter() *piMessagesTypedAdapter {
	return &piMessagesTypedAdapter{messages: map[*Object]*Message{}, blocks: map[*Object]*ContentBlock{}, lists: map[*Array]*BlockList{}, usages: map[*Object]*Usage{}}
}

// Malformed provider fields remain serializable even where Go has a stricter
// public field type. Only those fields are omitted from the native projection;
// the encoding metadata below restores their exact JSON value.
func decodePiMessagesProjection[T any](value *Object) (T, error) {
	var result T
	raw, err := jsonjs.MarshalValue(value)
	if err != nil {
		return result, err
	}
	if err = json.Unmarshal(raw, &result); err == nil {
		return result, nil
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(raw, &fields); err != nil {
		return result, err
	}
	accepted := map[string]json.RawMessage{}
	for key, field := range fields {
		one, err := json.Marshal(map[string]json.RawMessage{key: field})
		if err != nil {
			return result, err
		}
		var trial T
		if json.Unmarshal(one, &trial) == nil {
			accepted[key] = field
		}
	}
	raw, err = json.Marshal(accepted)
	if err != nil {
		return result, err
	}
	var clean T
	err = json.Unmarshal(raw, &clean)
	return clean, err
}

func piMessagesEncoding(source *Object, projected any) (*transcriptEncoding, error) {
	raw, err := jsonjs.MarshalValue(source)
	if err != nil {
		return nil, err
	}
	normalized, err := json.Marshal(projected)
	if err != nil {
		return nil, err
	}
	var original, baseline map[string]json.RawMessage
	if err = json.Unmarshal(raw, &original); err != nil {
		return nil, err
	}
	if err = json.Unmarshal(normalized, &baseline); err != nil {
		return nil, err
	}
	encoding := &transcriptEncoding{original: map[string]json.RawMessage{}, normalized: map[string]json.RawMessage{}}
	for key := range original {
		if _, exists := baseline[key]; !exists {
			baseline[key] = nil
		}
	}
	for key, current := range baseline {
		before := original[key]
		if bytes.Equal(before, current) {
			continue
		}
		a, ae := jsonjs.DecodeJSON(before)
		b, be := jsonjs.DecodeJSON(current)
		if before != nil && current != nil && ae == nil && be == nil && reflect.DeepEqual(a, b) {
			continue
		}
		encoding.normalized[key] = current
		if before != nil {
			encoding.original[key] = before
		}
	}
	if len(encoding.normalized) == 0 {
		return nil, nil
	}
	return encoding, nil
}

func (a *piMessagesTypedAdapter) block(value *Object) (*ContentBlock, error) {
	if value == nil {
		return nil, nil
	}
	block, err := decodePiMessagesProjection[ContentBlock](value)
	if err != nil {
		return nil, err
	}
	block.encoding = nil
	block.encoding, err = piMessagesEncoding(value, block)
	if err != nil {
		return nil, err
	}
	target := a.blocks[value]
	if target == nil {
		target = &ContentBlock{}
		a.blocks[value] = target
	}
	*target = block
	return target, nil
}

func (a *piMessagesTypedAdapter) message(value *Object) (*Message, error) {
	if value == nil {
		return nil, nil
	}
	projection := authSpread(value)
	projection.Delete("content")
	projection.Delete("usage")
	message, err := decodePiMessagesProjection[Message](projection)
	if err != nil {
		return nil, err
	}
	if content, ok := value.Get("content").(*Array); ok {
		blocks := make([]*ContentBlock, content.Len())
		for _, i := range content.Keys() {
			if item, ok := content.Get(i).(*Object); ok {
				blocks[i], err = a.block(item)
			} else if content.Get(i) == nil || jsonjs.IsNull(content.Get(i)) {
				blocks[i] = NullContentBlock()
			}
			if err != nil {
				return nil, err
			}
		}
		list := a.lists[content]
		if list == nil {
			list = NewBlockList()
			a.lists[content] = list
		}
		list.SetLength(0)
		list.Append(blocks...)
		message.Content = MessageContent{Blocks: list}
	}
	if usage, ok := value.Get("usage").(*Object); ok {
		next, err := decodePiMessagesProjection[Usage](usage)
		if err != nil {
			return nil, err
		}
		target := a.usages[usage]
		if target == nil {
			target = &Usage{}
			a.usages[usage] = target
		}
		*target = next
		message.Usage = target
	}
	message.encoding = nil
	message.encoding, err = piMessagesEncoding(value, message)
	if err != nil {
		return nil, err
	}
	target := a.messages[value]
	if target == nil {
		target = &Message{}
		a.messages[value] = target
	}
	*target = message
	return target, nil
}

func (a *piMessagesTypedAdapter) event(value *Object) (AssistantMessageEvent, error) {
	projection := authSpread(value)
	for _, field := range []string{"partial", "message", "error", "toolCall"} {
		projection.Delete(field)
	}
	event, err := decodePiMessagesProjection[AssistantMessageEvent](projection)
	if err != nil {
		return event, err
	}
	raw, err := jsonjs.MarshalValue(projection)
	if err != nil {
		return event, err
	}
	// Unknown fields are forwarded by the source's object spread.
	if err = json.Unmarshal(raw, &event.Extra); err != nil {
		return event, err
	}
	for _, key := range []string{"type", "contentIndex", "delta", "content", "reason"} {
		delete(event.Extra, key)
	}
	for _, field := range []struct {
		name   string
		target **Message
	}{{"partial", &event.Partial}, {"message", &event.Message}, {"error", &event.Error}} {
		object, _ := value.Get(field.name).(*Object)
		*field.target, err = a.message(object)
		if err != nil {
			return event, err
		}
	}
	tool, _ := value.Get("toolCall").(*Object)
	event.ToolCall, err = a.block(tool)
	if err != nil {
		return event, err
	}
	event.encoding, err = piMessagesEncoding(value, event)
	return event, err
}
