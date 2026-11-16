package host

import (
	"crypto/sha256"
	"encoding"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"hash/maphash"
	"math"

	"github.com/2found/2ai/core/ai"
)

// messageFacts is what the compactor reads from one native message. Decoding
// it means scanning the whole message, so it is done once per distinct message
// and cached by fingerprint.
type messageFacts struct {
	role, toolCallID, toolName, stopReason string
	isError, summary, hasUsage             bool
	timestamp, reducedAt                   int64
	usage                                  float64
	calls                                  []string // tool call IDs of a settled assistant message
	callArgs                               map[string]json.RawMessage
	bytes                                  int
	// tokens is the message's text estimate, counted from raw on first use:
	// most requests read the estimate of their view only.
	tokens  int
	counted bool
	raw     json.RawMessage
}

func (f *messageFacts) tokenCount() int {
	if !f.counted {
		f.tokens, f.counted = max(1, int(ai.EstimateTextTokensBytes(f.raw))-imageDiscount(f.raw)), true
	}
	return f.tokens
}

func decodeFacts(raw json.RawMessage) messageFacts {
	var m struct {
		Role, ToolCallID, ToolName, StopReason string
		IsError                                bool
		Timestamp, AgentrayContextReducedAt    int64
		AgentrayContextSummary                 string
		Content                                json.RawMessage
		Usage                                  *struct{ Input, Output, CacheRead, CacheWrite float64 }
	}
	_ = json.Unmarshal(raw, &m)
	f := messageFacts{role: m.Role, toolCallID: m.ToolCallID, toolName: m.ToolName, stopReason: m.StopReason, isError: m.IsError,
		timestamp: m.Timestamp, reducedAt: m.AgentrayContextReducedAt, summary: m.AgentrayContextSummary != "", bytes: len(raw)}
	if u := m.Usage; u != nil {
		f.usage, f.hasUsage = u.Input+u.Output+u.CacheRead+u.CacheWrite, true
	}
	if m.Role == "assistant" && m.StopReason != "error" && m.StopReason != "aborted" {
		var blocks []struct {
			Type, ID  string
			Arguments json.RawMessage
		}
		_ = json.Unmarshal(m.Content, &blocks)
		for _, block := range blocks {
			if block.Type == "toolCall" {
				f.calls = append(f.calls, block.ID)
				if f.callArgs == nil {
					f.callArgs = map[string]json.RawMessage{}
				}
				f.callArgs[block.ID] = block.Arguments
			}
		}
	}
	f.tokens, f.counted = max(1, estimateTokens(string(raw))-imageDiscount(raw)), true
	return f
}

// transcriptMemo caches what earlier requests already derived. The native
// transcript only grows, so a request re-derives facts and the prefix digest
// only for messages it has not seen; the rest costs one fingerprint pass.
type transcriptMemo struct {
	seed  maphash.Seed
	facts map[uint64]*messageFacts
	// slab holds new facts in chunks, so caching a message costs no
	// allocation of its own.
	slab []messageFacts
	// marks[i] fingerprints authoritative message i; states[i] is the SHA-256
	// state after "[" and messages 0..i-1, so a prefix digest of any length
	// up to len(states)-1 needs no rehash.
	marks  []uint64
	states [][]byte
	// canonicalized counts messages canonicalized for the digest, for tests.
	canonicalized int
}

func (m *transcriptMemo) mark(raw json.RawMessage) uint64 {
	if m.facts == nil {
		m.seed, m.facts = maphash.MakeSeed(), map[uint64]*messageFacts{}
	}
	return maphash.Bytes(m.seed, raw)
}

// of returns the cached facts of each message.
func (m *transcriptMemo) of(messages []json.RawMessage) []*messageFacts {
	if m.facts == nil {
		m.seed, m.facts = maphash.MakeSeed(), make(map[uint64]*messageFacts, len(messages))
	}
	out := make([]*messageFacts, len(messages))
	for i, raw := range messages {
		key := m.mark(raw)
		f := m.facts[key]
		if f == nil {
			if len(m.slab) == cap(m.slab) {
				m.slab = make([]messageFacts, 0, min(max(64, len(messages)-i), 4096))
			}
			m.slab = append(m.slab, factsOf(raw))
			f = &m.slab[len(m.slab)-1]
			m.facts[key] = f
		}
		out[i] = f
	}
	return out
}

// digest is PrefixDigest(messages[:count]) without rehashing a prefix an
// earlier request already hashed. The bytes hashed are those of PrefixDigest:
// the canonical JSON array of the prefix.
func (m *transcriptMemo) digest(messages []json.RawMessage, count int) string {
	valid := 0
	for valid < len(m.marks) && valid < count && m.marks[valid] == m.mark(messages[valid]) {
		valid++
	}
	m.marks, m.states = m.marks[:valid], m.states[:min(len(m.states), valid+1)]
	if len(m.states) == 0 {
		h := sha256.New()
		h.Write([]byte("["))
		m.states = append(m.states, marshalState(h))
	}
	h := unmarshalState(m.states[len(m.states)-1])
	for i := len(m.marks); i < count; i++ {
		canonical, err := CanonicalJSON(messages[i])
		if err != nil {
			return ""
		}
		m.canonicalized++
		if i > 0 {
			h.Write([]byte(","))
		}
		h.Write(canonical)
		m.marks = append(m.marks, m.mark(messages[i]))
		m.states = append(m.states, marshalState(h))
	}
	h = unmarshalState(m.states[count])
	h.Write([]byte("]"))
	return hex.EncodeToString(h.Sum(nil))
}

func marshalState(h hash.Hash) []byte {
	state, err := h.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("sha256 state: %v", err))
	}
	return state
}

func unmarshalState(state []byte) hash.Hash {
	h := sha256.New()
	if err := h.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
		panic(fmt.Sprintf("sha256 state: %v", err))
	}
	return h
}

// joinMessages encodes a message array without re-validating each element:
// every element came out of a decoded array or a json.Marshal.
func joinMessages(messages []json.RawMessage) json.RawMessage {
	size := 2
	for _, raw := range messages {
		size += len(raw) + 1
	}
	out := make([]byte, 0, size)
	out = append(out, '[')
	for i, raw := range messages {
		if i > 0 {
			out = append(out, ',')
		}
		out = append(out, raw...)
	}
	return append(out, ']')
}

func splitMessages(raw json.RawMessage) ([]json.RawMessage, error) { return SplitMessages(raw) }

// validFacts is validateMessages over cached facts.
func validFacts(facts []*messageFacts) error {
	open := map[string]int{}
	for _, f := range facts {
		if f.role != "toolResult" && len(open) > 0 {
			return fmt.Errorf("%w: incomplete tool batch", ErrUnsettledEffect)
		}
		for _, id := range f.calls {
			open[id]++
		}
		if f.role == "toolResult" {
			if open[f.toolCallID] > 1 {
				open[f.toolCallID]--
			} else {
				delete(open, f.toolCallID)
			}
		}
	}
	if len(open) > 0 {
		return fmt.Errorf("%w: tool results need recovery", ErrUnsettledEffect)
	}
	return nil
}

// factTokens is ContextTokens over cached facts: settled provider usage plus
// the appended tail, floored by the text estimate. A rewrite (summary, cleared
// or previewed result) makes earlier usage stale.
func factTokens(facts []*messageFacts) int {
	floor := 1 // the array's brackets and separators
	var rewrittenAt int64
	for _, f := range facts {
		floor += f.tokenCount()
		if f.summary {
			rewrittenAt = max(rewrittenAt, f.timestamp)
		}
		rewrittenAt = max(rewrittenAt, f.reducedAt)
	}
	tail := 0
	for i := len(facts) - 1; i >= 0; i-- {
		f := facts[i]
		if f.role == "assistant" && f.stopReason != "error" && f.stopReason != "aborted" && f.hasUsage && (rewrittenAt == 0 || f.timestamp > rewrittenAt) {
			count := f.usage
			if count > 0 && !math.IsNaN(count) && !math.IsInf(count, 0) && count < float64(math.MaxInt-tail) {
				return max(floor, int(math.Ceil(count))+tail)
			}
		}
		tail += f.tokenCount()
	}
	return max(1, floor)
}
