package ai

import (
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// TestPricingChargesCacheTokensSeparately verifies cache reads bill at the
// discounted rate and cache writes at the premium, instead of the full input
// rate — the whole point of the long-session caching steal.
func TestPricingChargesCacheTokensSeparately(t *testing.T) {
	p := Pricing{"m": {InputPerM: 10, OutputPerM: 30}}

	full, priced := p.Cost("m", protocol.Usage{InputTokens: 1_000_000})
	if full != 10 || !priced {
		t.Fatalf("full input cost = %v priced=%v, want 10 true", full, priced)
	}
	// A million cache-read tokens must cost a fraction of a million fresh ones.
	read, _ := p.Cost("m", protocol.Usage{CacheReadTokens: 1_000_000})
	if read != 10*defaultCacheReadMultiplier {
		t.Fatalf("cache-read cost = %v, want %v", read, 10*defaultCacheReadMultiplier)
	}
	// Cache writes carry a small premium over fresh input.
	write, _ := p.Cost("m", protocol.Usage{CacheWriteTokens: 1_000_000})
	if write != 10*defaultCacheWriteMultiplier {
		t.Fatalf("cache-write cost = %v, want %v", write, 10*defaultCacheWriteMultiplier)
	}
	// Explicit cache rates override the derived defaults.
	pe := Pricing{"m": {InputPerM: 10, OutputPerM: 30, CacheReadPerM: 0.5}}
	if got, _ := pe.Cost("m", protocol.Usage{CacheReadTokens: 1_000_000}); got != 0.5 {
		t.Fatalf("explicit cache-read cost = %v, want 0.5", got)
	}
}

func TestPricingCost(t *testing.T) {
	p := Pricing{"gpt-4o": {InputPerM: 2.50, OutputPerM: 10.00}}

	// 1M input + 1M output at the listed price.
	got, priced := p.Cost("gpt-4o", protocol.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if want := 12.50; got != want || !priced {
		t.Fatalf("exact: got %v priced=%v want %v true", got, priced, want)
	}
	// Prefix match: a dated variant resolves to the family price.
	if got, priced := p.Cost("gpt-4o-2024-08-06", protocol.Usage{InputTokens: 1_000_000}); got != 2.50 || !priced {
		t.Fatalf("prefix: got %v priced=%v want 2.50 true", got, priced)
	}
	// Unknown model prices at zero AND reports itself unpriced — a caller must
	// not mistake this for a genuinely free call.
	if got, priced := p.Cost("mystery-model", protocol.Usage{InputTokens: 1_000_000}); got != 0 || priced {
		t.Fatalf("unknown: got %v priced=%v want 0 false", got, priced)
	}
}
