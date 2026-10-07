package engine

import (
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

// Pi's transcript replay visits every slot, including holes. Validate while
// preserving that distinction before projecting onto non-nullable AI messages.
func replayMessageValues(messages *MessageList) ([]ai.Message, error) {
	snapshot := messages.indexedSnapshot()
	result := []ai.Message{}
	for index := 0; index < snapshot.length; index++ {
		message, present := snapshot.values[index]
		if message == nil {
			return nil, jsonjs.PropertyReadError(present, "message.role")
		}
		result = append(result, *message)
	}
	return result, nil
}

func messageListOrEmpty(messages *MessageList) *MessageList {
	if messages == nil {
		return NewList[*ai.Message]()
	}
	return messages
}

// MessagePointers exposes the entries of a value slice to the native loop.
// The loop and its callbacks retain these objects; callers must synchronize
// any concurrent writes. Replacing a list entry does not replace that object.
func MessagePointers(messages []ai.Message) []*ai.Message {
	result := make([]*ai.Message, len(messages))
	for i := range messages {
		result[i] = &messages[i]
	}
	return result
}

// MessageValues projects shared message objects at a snapshot or provider
// boundary. Nested payloads remain shared and must be treated as immutable.
func MessageValues(messages []*ai.Message) []ai.Message {
	result := make([]ai.Message, len(messages))
	for i, message := range messages {
		result[i] = *message
	}
	return result
}
