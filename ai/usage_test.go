package ai

import (
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// Per-vendor usage normalization: each wire adapter must report full-price
// input tokens separately from cached ones, so pricing (agentcore/plugins/
// monitor) does not bill a cache hit at the fresh-input rate.

// TestOpenAIUsageNormalizesCachedTokens verifies the OpenAI adapter pulls cached
// tokens out of prompt_tokens so InputTokens stays full-price-only and the cached
// portion is reported separately.
func TestOpenAIUsageNormalizesCachedTokens(t *testing.T) {
	u := oaiUsage{PromptTokens: 1000, CompletionTokens: 200}
	u.PromptTokensDetails.CachedTokens = 800
	got := u.usage()
	if got.InputTokens != 200 {
		t.Fatalf("InputTokens = %d, want 200 (1000 total - 800 cached)", got.InputTokens)
	}
	if got.CacheReadTokens != 800 {
		t.Fatalf("CacheReadTokens = %d, want 800", got.CacheReadTokens)
	}
	if got.OutputTokens != 200 {
		t.Fatalf("OutputTokens = %d, want 200", got.OutputTokens)
	}
}

// TestAnthropicUsageMapsCacheCounters verifies the Anthropic adapter maps its
// two cache counters onto the neutral read/write fields without touching
// input_tokens (which already excludes the cached prefix).
func TestAnthropicUsageMapsCacheCounters(t *testing.T) {
	got := antUsage{
		InputTokens:              50,
		OutputTokens:             10,
		CacheReadInputTokens:     400,
		CacheCreationInputTokens: 120,
	}.usage()
	want := protocol.Usage{InputTokens: 50, OutputTokens: 10, CacheReadTokens: 400, CacheWriteTokens: 120}
	if got != want {
		t.Fatalf("usage = %+v, want %+v", got, want)
	}
}
