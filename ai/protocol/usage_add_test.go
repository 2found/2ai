package protocol

import "testing"

func TestUsageAddSumsEveryFieldAndKeepsUnpriced(t *testing.T) {
	a := Usage{InputTokens: 1, OutputTokens: 10, CacheReadTokens: 2, CacheWriteTokens: 5, CacheWrite1hTokens: 3, ReasoningTokens: 4, CostUSD: 0.25}
	b := Usage{InputTokens: 2, OutputTokens: 20, CacheReadTokens: 3, CacheWriteTokens: 6, CacheWrite1hTokens: 1, ReasoningTokens: 7, CostUSD: 0.5, CostUnpriced: true}
	want := Usage{InputTokens: 3, OutputTokens: 30, CacheReadTokens: 5, CacheWriteTokens: 11, CacheWrite1hTokens: 4, ReasoningTokens: 11, CostUSD: 0.75, CostUnpriced: true}
	if got := a.Add(b); got != want {
		t.Fatalf("Add = %+v, want %+v", got, want)
	}
	if got := (Usage{}).Add(a); got != a || got.CostUnpriced {
		t.Fatalf("zero identity broken: %+v", got)
	}
}
