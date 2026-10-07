package host

import (
	"context"
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"github.com/2found/2ai/ai/protocol"
)

// FoldSummary treats oversized transcript JSON as source segments, not new
// native messages. Every bounded call includes the preceding summary. This
// keeps a smaller summarizer/fallback model from receiving the entire prefix
// at once while leaving the signed transcript and tool groups untouched.
func FoldSummary(ctx context.Context, prefix json.RawMessage, budget int, revision string, summarize func(context.Context, json.RawMessage, string) (string, protocol.Usage, error)) (string, protocol.Usage, error) {
	var total protocol.Usage
	if estimateTokens(string(prefix)) <= budget {
		return summarize(ctx, prefix, revision)
	}
	if budget < 256 {
		return "", total, fmt.Errorf("summary input budget is too small")
	}
	carry := ""
	for offset := 0; offset < len(prefix); {
		if err := ctx.Err(); err != nil {
			return "", total, err
		}
		payload := func(size int) json.RawMessage {
			raw, _ := json.Marshal(map[string]any{"previous_summary": carry, "source_segment": string(prefix[offset : offset+size]), "note": "Source segments may split a long message; preserve earlier facts and fold this segment into the handoff."})
			return raw
		}
		lo, hi := 0, min(len(prefix)-offset, budget*4)
		for lo < hi {
			mid := (lo + hi + 1) / 2
			if estimateTokens(string(payload(mid))) <= budget {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		for lo > 0 && !utf8.Valid(prefix[offset:offset+lo]) {
			lo--
		}
		if lo == 0 {
			return "", total, fmt.Errorf("summary did not leave room for the next source segment")
		}
		text, usage, err := summarize(ctx, payload(lo), revision)
		total.InputTokens += usage.InputTokens
		total.OutputTokens += usage.OutputTokens
		total.CacheReadTokens += usage.CacheReadTokens
		total.CacheWriteTokens += usage.CacheWriteTokens
		total.CostUSD += usage.CostUSD
		total.CostUnpriced = total.CostUnpriced || usage.CostUnpriced
		if err != nil {
			return "", total, err
		}
		if !validSummaryText(text) {
			return "", total, fmt.Errorf("invalid folded summary")
		}
		carry = text
		offset += lo
	}
	return carry, total, nil
}
