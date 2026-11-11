package ai

import (
	"encoding/json"
	"math"
)

// EstimatedCharsPerToken is Pi's conservative fallback ratio; settled provider
// usage remains authoritative. This is the module's only token estimator:
// hosts estimate serialized text with EstimateTextTokens and transcripts with
// EstimateContextTokens.
const EstimatedCharsPerToken = 3.5

// estimatedImageChars is the flat character charge for one inline image.
const estimatedImageChars = 4800

// EstimatedImageTokens is the flat token estimate for one inline image:
// providers bill an image by its dimensions, never by the length of its base64
// text. A host estimating serialized messages replaces an image's data with it.
const EstimatedImageTokens = estimatedImageChars / EstimatedCharsPerToken

// ContextUsageEstimate follows Pi's usage-aware estimate. Usage applies only
// when its response is at least as new as every preceding transcript message.
type ContextUsageEstimate struct {
	Tokens         float64 `json:"tokens"`
	UsageTokens    float64 `json:"usageTokens"`
	TrailingTokens float64 `json:"trailingTokens"`
	LastUsageIndex *int    `json:"lastUsageIndex"`
}

func CalculateContextTokens(usage Usage) float64 {
	if usage.TotalTokens != 0 {
		return usage.TotalTokens
	}
	return usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
}

func estimateUTF16Length(text string) int {
	length := 0
	for _, r := range text {
		length++
		if r > 0xffff {
			length++
		}
	}
	return length
}

func EstimateTextTokens(text string) float64 {
	return math.Ceil(float64(estimateUTF16Length(text)) / EstimatedCharsPerToken)
}

func estimateJSONString(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "[unserializable]"
	}
	encoded, err = stringifyCompletionsJSON(encoded)
	if err != nil {
		return "[unserializable]"
	}
	return string(encoded)
}

func EstimateMessageTokens(message Message) float64 {
	switch message.Role {
	case "system":
		tokens := EstimateTextTokens(GetSystemMessageText(message))
		if len(message.ToolsAdded) > 0 {
			tokens += EstimateTextTokens(estimateJSONString(message.ToolsAdded))
		}
		if len(message.ToolsRemoved) > 0 {
			tokens += EstimateTextTokens(estimateJSONString(message.ToolsRemoved))
		}
		return tokens
	case "user", "toolResult":
		if message.Content.Text != nil {
			return EstimateTextTokens(*message.Content.Text)
		}
	}
	chars := 0
	for _, block := range message.Content.Blocks.Values() {
		switch block.Type {
		case "image":
			chars += estimatedImageChars
		case "text":
			chars += estimateUTF16Length(block.Text)
		case "thinking":
			chars += estimateUTF16Length(block.Thinking)
		default:
			args := "undefined"
			if len(block.Arguments) > 0 {
				args = estimateJSONString(block.Arguments)
			}
			chars += estimateUTF16Length(block.Name) + estimateUTF16Length(args)
		}
	}
	return math.Ceil(float64(chars) / EstimatedCharsPerToken)
}

func EstimateContextTokens(messages []Message) ContextUsageEstimate {
	latest := int64(math.MinInt64)
	index := -1
	for i, message := range messages {
		if message.Role == "assistant" && message.Timestamp >= latest && message.StopReason != "aborted" && message.StopReason != "error" && message.Usage != nil && CalculateContextTokens(*message.Usage) > 0 {
			index = i
		}
		latest = max(latest, message.Timestamp)
	}
	estimate := ContextUsageEstimate{}
	if index >= 0 {
		estimate.UsageTokens = CalculateContextTokens(*messages[index].Usage)
		estimate.LastUsageIndex = &index
	}
	for _, message := range messages[index+1:] {
		estimate.TrailingTokens += EstimateMessageTokens(message)
	}
	estimate.Tokens = estimate.UsageTokens + estimate.TrailingTokens
	return estimate
}
