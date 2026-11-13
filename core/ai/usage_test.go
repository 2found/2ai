package ai

import (
	"encoding/json"
	"testing"

	"github.com/2found/2ai/core/ai/protocol"
)

// Per-vendor usage normalization: each wire adapter must report full-price
// input tokens separately from cached ones, so pricing (feature/plugins/
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

// TestUsageSubsetsReachProtocolUsage pins issue: reasoning tokens and the
// 1-hour cache-write split were dropped between native usage and the neutral
// contract (RunResult.Usage), so a host could neither show thinking spend nor
// price 1h writes at their real premium.
func TestUsageSubsetsReachProtocolUsage(t *testing.T) {
	var wire antUsage
	if err := json.Unmarshal([]byte(`{"input_tokens":5,"output_tokens":90,"cache_creation_input_tokens":300,"cache_creation":{"ephemeral_1h_input_tokens":200,"ephemeral_5m_input_tokens":100},"output_tokens_details":{"thinking_tokens":60}}`), &wire); err != nil {
		t.Fatal(err)
	}
	if got := wire.usage(); got.CacheWrite1hTokens != 200 || got.ReasoningTokens != 60 || got.CacheWriteTokens != 300 {
		t.Fatalf("legacy Anthropic usage lost subsets: %+v", got)
	}
	var openai oaiUsage
	_ = json.Unmarshal([]byte(`{"prompt_tokens":10,"completion_tokens":40,"completion_tokens_details":{"reasoning_tokens":30}}`), &openai)
	if got := openai.usage(); got.ReasoningTokens != 30 || got.OutputTokens != 40 {
		t.Fatalf("OpenAI reasoning lost: %+v", got)
	}
	long, reasoning := 200.0, 60.0
	native := Message{Role: "assistant", Content: BlockContent(), Usage: &Usage{Input: 5, Output: 90, CacheWrite: 300, CacheWrite1h: &long, Reasoning: &reasoning, Cost: UsageCost{Total: 0.5}}}
	raw, err := json.Marshal(native)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := ProjectMessage(raw)
	if err != nil || projected.Usage == nil {
		t.Fatal(err)
	}
	want := protocol.Usage{InputTokens: 5, OutputTokens: 90, CacheWriteTokens: 300, CacheWrite1hTokens: 200, ReasoningTokens: 60, CostUSD: 0.5}
	if *projected.Usage != want {
		t.Fatalf("projection = %+v, want %+v", *projected.Usage, want)
	}
	// A usage object without cost is unknown, not free.
	unpriced, err := ProjectMessage(json.RawMessage(`{"role":"assistant","content":[],"usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2}}`))
	if err != nil || unpriced.Usage == nil || !unpriced.Usage.CostUnpriced || unpriced.Usage.ReasoningTokens != 0 {
		t.Fatalf("missing cost must project as unpriced: %+v %v", unpriced.Usage, err)
	}
	if _, err := ProjectMessage(json.RawMessage(`{"role":"assistant","usage":{"input":"x"}}`)); err == nil {
		t.Fatal("malformed usage must fail projection")
	}
}

func TestPricingBillsOneHourCacheWritesAtTheirPremium(t *testing.T) {
	p := Pricing{"m": {InputPerM: 10, OutputPerM: 30}}
	short, _ := p.Cost("m", protocol.Usage{CacheWriteTokens: 1_000_000})
	long, _ := p.Cost("m", protocol.Usage{CacheWriteTokens: 1_000_000, CacheWrite1hTokens: 1_000_000})
	mixed, _ := p.Cost("m", protocol.Usage{CacheWriteTokens: 1_000_000, CacheWrite1hTokens: 250_000})
	if short != 12.5 || long != 20 || mixed != 0.75*12.5+0.25*20 {
		t.Fatalf("1h writes mispriced: short=%v long=%v mixed=%v", short, long, mixed)
	}
	// Reasoning is a subset of output, never billed twice.
	if got, _ := p.Cost("m", protocol.Usage{OutputTokens: 1_000_000, ReasoningTokens: 900_000}); got != 30 {
		t.Fatalf("reasoning double-billed: %v", got)
	}
	// A corrupt subset larger than the total cannot produce a negative 5m share.
	if got, _ := p.Cost("m", protocol.Usage{CacheWriteTokens: 100, CacheWrite1hTokens: 1_000_000}); got != 100.0/1e6*20 {
		t.Fatalf("oversized 1h subset not clamped: %v", got)
	}
	explicit := Pricing{"m": {InputPerM: 10, CacheWrite1hPerM: 15}}
	if got, _ := explicit.Cost("m", protocol.Usage{CacheWriteTokens: 1_000_000, CacheWrite1hTokens: 1_000_000}); got != 15 {
		t.Fatalf("explicit 1h rate ignored: %v", got)
	}
}
