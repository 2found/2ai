package ai

import (
	"slices"
	"strings"
	"time"
)

const nonVisionUserImagePlaceholder = "(image omitted: model does not support images)"
const nonVisionToolImagePlaceholder = "(tool image omitted: model does not support images)"

type ToolCallIDNormalizer func(id string, model *Model, source *Message) string

// TransformMessages ports Pi's api/transform-messages.ts: replay signatures
// only on the originating model, repair unanswered tool calls, and downgrade
// images for models that accept only text. Transformation itself does not mutate
// input messages; the normalizer receives their shared objects and may edit them.
// This value adapter snapshots the model and returned message structs. Use
// TransformMessageReferences to retain message/model identity throughout.
func TransformMessages(messages []Message, model Model, normalize ToolCallIDNormalizer) []Message {
	return transformMessagesAt(messages, model, normalize, func() int64 { return time.Now().UnixMilli() })
}

// TransformMessageReferences retains Pi's message/model object identity and
// shallow-copy boundaries, including edits made by the ID normalizer. Callers
// must synchronize concurrent access to these shared objects.
func TransformMessageReferences(messages []*Message, model *Model, normalize ToolCallIDNormalizer) []*Message {
	return transformMessageReferencesAt(messages, model, normalize, func() int64 { return time.Now().UnixMilli() })
}

func transformMessagesAt(messages []Message, model Model, normalize ToolCallIDNormalizer, now func() int64) []Message {
	refs := make([]*Message, len(messages))
	for i := range messages {
		refs[i] = &messages[i]
	}
	transformed := transformMessageReferencesAt(refs, &model, normalize, now)
	result := make([]Message, len(transformed))
	for i, message := range transformed {
		result[i] = *message
	}
	return result
}

func replaceImagesWithPlaceholder(content []*ContentBlock, placeholder string) []*ContentBlock {
	result := []*ContentBlock{}
	previousWasPlaceholder := false
	for _, block := range content {
		if block.Type == "image" {
			if !previousWasPlaceholder {
				result = append(result, &ContentBlock{Type: "text", Text: placeholder})
			}
			previousWasPlaceholder = true
			continue
		}
		result = append(result, block)
		previousWasPlaceholder = block.Text == placeholder
	}
	return result
}

func transformMessageReferencesAt(messages []*Message, model *Model, normalize ToolCallIDNormalizer, now func() int64) []*Message {
	ids := map[string]string{}
	// Pi completes content normalization and image conversion for the whole
	// transcript before invoking any ID callback. Later callback edits must not
	// change which images were converted or leak top-level edits into copies.
	prepared := make([]*Message, len(messages))
	for i, message := range messages {
		if message.Content.Text == nil && message.Content.Blocks == nil {
			copy := *message
			message = &copy
			message.Content = BlockContent()
		}
		prepared[i] = message
	}
	if !slices.Contains(model.Input, "image") {
		for i, message := range prepared {
			if message.Role == "user" && message.Content.Text == nil {
				copy := *message
				message = &copy
				message.Content = BlockReferences(replaceImagesWithPlaceholder(message.Content.Blocks.Values(), nonVisionUserImagePlaceholder)...)
			} else if message.Role == "toolResult" {
				copy := *message
				message = &copy
				message.Content = BlockReferences(replaceImagesWithPlaceholder(message.Content.Blocks.Values(), nonVisionToolImagePlaceholder)...)
			}
			prepared[i] = message
		}
	}
	transformed := make([]*Message, 0, len(prepared))
	for _, message := range prepared {
		if message.Role == "toolResult" {
			if id := ids[message.ToolCallID]; id != "" && id != message.ToolCallID {
				copy := *message
				message = &copy
				message.ToolCallID = id
			}
		}
		if message.Role == "assistant" {
			same := message.Provider == model.Provider && message.API == model.API && message.Model == model.ID
			blocks := []*ContentBlock{}
			// flatMap captures the array and length, then reads each live slot.
			// A normalizer may replace later entries or the message's content.
			content := message.Content.Blocks
			for index, length := 0, content.Len(); index < length; index++ {
				block := content.Get(index)
				if block == nil {
					continue
				}
				switch block.Type {
				case "thinking":
					if block.Redacted != nil && *block.Redacted {
						if same {
							blocks = append(blocks, block)
						}
						continue
					}
					if same && block.ThinkingSignature != nil && *block.ThinkingSignature != "" {
						blocks = append(blocks, block)
						continue
					}
					if strings.TrimFunc(block.Thinking, jsWhitespace) == "" {
						continue
					}
					if !same {
						block = &ContentBlock{Type: "text", Text: block.Thinking}
					}
				case "text":
					if !same {
						block = &ContentBlock{Type: "text", Text: block.Text}
					}
				case "toolCall":
					original := block
					if !same && block.ThoughtSignature != nil && *block.ThoughtSignature != "" {
						copy := *block
						block = &copy
						block.ThoughtSignature = nil
					}
					if !same && normalize != nil {
						id := normalize(original.ID, model, message)
						if id != original.ID {
							ids[original.ID] = id
							copy := *block
							block = &copy
							block.ID = id
						}
					}
				}
				blocks = append(blocks, block)
			}
			copy := *message
			message = &copy
			message.Content = BlockReferences(blocks...)
		}
		transformed = append(transformed, message)
	}

	result := []*Message{}
	pending := []*ContentBlock{}
	answered := map[string]bool{}
	held := []*Message{}
	closePending := func() {
		if len(pending) > 0 {
			for _, call := range pending {
				if !answered[call.ID] {
					result = append(result, &Message{Role: "toolResult", ToolCallID: call.ID, ToolName: call.Name,
						Content: BlockContent(ContentBlock{Type: "text", Text: "No result provided"}), IsError: true, Timestamp: now()})
				}
			}
			pending = nil
			answered = map[string]bool{}
		}
		result = append(result, held...)
		held = nil
	}
	for _, message := range transformed {
		switch message.Role {
		case "assistant":
			closePending()
			if message.StopReason == "error" || message.StopReason == "aborted" {
				continue
			}
			for _, block := range message.Content.Blocks.Values() {
				if block.Type == "toolCall" {
					pending = append(pending, block)
				}
			}
			if len(pending) > 0 {
				answered = map[string]bool{}
			}
		case "toolResult":
			answered[message.ToolCallID] = true
		case "system":
			if len(pending) > 0 {
				held = append(held, message)
				continue
			}
		case "user":
			closePending()
		}
		result = append(result, message)
	}
	closePending()
	return result
}

// ECMAScript trim includes BOM, excludes U+0085, and is not Go's TrimSpace.
func jsWhitespace(r rune) bool {
	return r == '\t' || r == '\n' || r == '\v' || r == '\f' || r == '\r' || r == ' ' ||
		r == '\u00a0' || r == '\u1680' || (r >= '\u2000' && r <= '\u200a') ||
		r == '\u2028' || r == '\u2029' || r == '\u202f' || r == '\u205f' || r == '\u3000' || r == '\ufeff'
}
