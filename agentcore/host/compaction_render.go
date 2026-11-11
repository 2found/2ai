package host

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/ai/protocol"
)

// Ceilings applied when a transcript is rendered for a standalone summarizer.
// Arguments are bounded per value before per call, so one large value (a file
// body, a SQL blob) cannot push the names of its sibling arguments out of the
// text the summarizer reads. Results keep both ends: a tool puts its
// conclusion last.
const (
	renderUserBytes     = 8000
	renderProseBytes    = 4000
	renderResultBytes   = 2000
	renderArgsBytes     = 600
	renderArgValueBytes = 200
)

// RenderTranscript renders native messages as role-labelled text for a
// summarizer, one block per message. It drops what a summary cannot use and
// what only costs tokens: thinking signatures, usage records, timestamps and
// inline image bytes. System messages are omitted because SummaryView keeps
// them verbatim. The native messages are not changed.
func RenderTranscript(messages []json.RawMessage) []string {
	blocks := make([]string, 0, len(messages))
	for _, raw := range messages {
		var m struct {
			Role, ToolName, AgentrayContextSummary string
			IsError                                bool
			Content                                json.RawMessage
		}
		if json.Unmarshal(raw, &m) != nil {
			continue
		}
		text, images, calls := renderContent(m.Content)
		note := ""
		if images > 0 {
			note = fmt.Sprintf(" [%d image(s) omitted]", images)
		}
		block := ""
		switch m.Role {
		case "user":
			switch {
			case m.AgentrayContextSummary != "":
				// Reattached context is rebuilt from the messages a new summary
				// stands for; summarizing it again would only repeat it.
				body, ok := SummaryText(text)
				if !ok {
					body = strings.TrimSpace(text)
				}
				block = "PREVIOUS SUMMARY (stands for earlier work that is no longer visible):\n" + body
			case strings.TrimSpace(text) != "" || images > 0:
				block = "USER: " + truncateMiddle(strings.TrimSpace(text), renderUserBytes) + note
			}
		case "assistant":
			var b strings.Builder
			if prose := strings.TrimSpace(text); prose != "" {
				b.WriteString("ASSISTANT: " + truncateMiddle(prose, renderProseBytes))
			}
			for _, call := range calls {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				fmt.Fprintf(&b, "ASSISTANT called %s(%s)", call.name, renderArguments(call.arguments))
			}
			block = b.String()
		case "toolResult":
			label := "TOOL " + m.ToolName
			if m.IsError {
				label += " (error)"
			}
			block = label + " -> " + truncateMiddle(strings.TrimSpace(text), renderResultBytes) + note
		}
		if block != "" {
			blocks = append(blocks, block)
		}
	}
	return blocks
}

type renderedCall struct {
	name      string
	arguments json.RawMessage
}

func renderContent(raw json.RawMessage) (text string, images int, calls []renderedCall) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", 0, nil
	}
	if json.Unmarshal(raw, &text) == nil {
		return text, 0, nil
	}
	var blocks []struct {
		Type, Text, Name string
		Arguments        json.RawMessage
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return "", 0, nil
	}
	parts := []string{}
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, block.Text)
		case "image":
			images++
		case "toolCall":
			calls = append(calls, renderedCall{block.Name, block.Arguments})
		}
	}
	return strings.Join(parts, "\n"), images, calls
}

// renderArguments emits keys sorted: the rendering is part of a model request,
// and Go's randomized map order would change it between identical calls.
func renderArguments(raw json.RawMessage) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || fields == nil {
		return truncateMiddle(string(raw), renderArgsBytes)
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, key+"="+truncateMiddle(string(fields[key]), renderArgValueBytes))
	}
	return truncateMiddle(strings.Join(parts, ", "), renderArgsBytes)
}

// truncateMiddle keeps both ends of s inside limit bytes and never splits a
// UTF-8 sequence.
func truncateMiddle(s string, limit int) string {
	if len(s) <= limit || limit <= 0 {
		return s
	}
	marker := fmt.Sprintf("\n…[%d bytes omitted]…\n", len(s)-limit)
	keep := limit - len(marker)
	if keep < 16 {
		return validPrefix(s, limit)
	}
	head := validPrefix(s, keep/2)
	tail := s[len(s)-(keep-keep/2):]
	for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
		tail = tail[1:]
	}
	return head + marker + tail
}

func validPrefix(s string, limit int) string {
	s = s[:min(len(s), limit)]
	for len(s) > 0 && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

// imageDiscount is how far the text estimate overstates one message: its inline
// image data, less ai.EstimatedImageTokens per image.
func imageDiscount(raw json.RawMessage) int {
	if !bytes.Contains(raw, []byte(`"image"`)) {
		return 0
	}
	var m struct{ Content json.RawMessage }
	var blocks []struct{ Type, Data string }
	if json.Unmarshal(raw, &m) != nil || json.Unmarshal(m.Content, &blocks) != nil {
		return 0
	}
	discount := 0
	for _, block := range blocks {
		if block.Type == "image" {
			discount += estimateTokens(block.Data) - int(math.Ceil(ai.EstimatedImageTokens))
		}
	}
	return max(0, discount)
}

// FoldRendered summarizes rendered blocks inside an input budget. Blocks that
// fit are summarized in one call. A longer transcript is folded in order, each
// call carrying the summary so far; a segment always ends on a block boundary,
// so no call reads half a message. summarize receives the summary so far
// (empty on the first call) and the next segment.
func FoldRendered(ctx context.Context, blocks []string, budget int, summarize func(ctx context.Context, previous, segment string) (string, protocol.Usage, error)) (string, protocol.Usage, error) {
	var total protocol.Usage
	if len(blocks) == 0 {
		return "", total, fmt.Errorf("nothing to summarize")
	}
	if budget < 256 {
		return "", total, fmt.Errorf("summary input budget is too small")
	}
	carry := ""
	for start := 0; start < len(blocks); {
		if err := ctx.Err(); err != nil {
			return "", total, err
		}
		room := budget - estimateTokens(carry) - 64
		if room < 128 {
			return "", total, fmt.Errorf("summary did not leave room for the next source segment")
		}
		end, used := start, 0
		for end < len(blocks) {
			size := estimateTokens(blocks[end]) + 1
			if used+size > room && end > start {
				break
			}
			used += size
			end++
		}
		segment := strings.Join(blocks[start:end], "\n")
		if used > room {
			// One block alone exceeds the room left; its own ceilings bound it
			// far below any real window, so this only binds on a tiny budget.
			segment = truncateMiddle(segment, tokenBytes(room))
		}
		text, usage, err := summarize(ctx, carry, segment)
		total = total.Add(usage)
		if err != nil {
			return "", total, err
		}
		if !validSummaryText(text) {
			return "", total, fmt.Errorf("invalid folded summary")
		}
		carry = text
		start = end
	}
	return carry, total, nil
}
