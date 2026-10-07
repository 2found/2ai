package ai

import (
	"encoding/json"
	"math"
)

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
	return math.Ceil(float64(estimateUTF16Length(text)) / 4)
}

func EstimateTextAndImageContentTokens(content MessageContent) float64 {
	if content.Text != nil {
		return EstimateTextTokens(*content.Text)
	}
	chars := 0
	for _, block := range content.Blocks.Values() {
		if block.Type == "text" {
			chars += estimateUTF16Length(block.Text)
		} else {
			chars += 4800
		}
	}
	return math.Ceil(float64(chars) / 4)
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
		return EstimateTextAndImageContentTokens(message.Content)
	}
	chars := 0
	for _, block := range message.Content.Blocks.Values() {
		switch block.Type {
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
	return math.Ceil(float64(chars) / 4)
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
