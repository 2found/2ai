package host

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"hash/maphash"
)

// ErrUnsettledEffect marks a missing physical effect receipt or native tool result.
var ErrUnsettledEffect = errors.New("native session has an unsettled tool effect")

// ValidateMessages checks completed native tool batches without projecting or
// changing messages. Provider error/aborted partial calls require no results.
func ValidateMessages(raw json.RawMessage) error {
	if string(bytes.Trim(raw, " \t\r\n")) == "null" {
		return nil // json.Unmarshal reads null as no messages
	}
	messages, err := SplitMessages(raw)
	if err != nil {
		return err
	}
	return validateMessages(messages)
}

// Transcript is a stored transcript read once: its messages, validated, with
// what batch validation and the compactor read from each. A run reads every
// stored message this way exactly once, and hands the result to its
// Compactor (Learn) instead of reading the messages again per request.
type Transcript struct {
	messages []json.RawMessage
	facts    []messageFacts
	scanned  []bool
}

// ReadMessages is ValidateMessageList that keeps what it read.
func ReadMessages(messages []json.RawMessage) (*Transcript, error) {
	t := &Transcript{messages: messages, facts: make([]messageFacts, len(messages)), scanned: make([]bool, len(messages))}
	for i, raw := range messages {
		t.facts[i], t.scanned[i] = scanFacts(raw)
	}
	err := validateBatches(len(messages), func(i int) (string, string, []string, error) {
		if t.scanned[i] {
			f := &t.facts[i]
			return f.role, f.toolCallID, f.calls, nil
		}
		role, toolCallID, calls, err := batchFacts(messages[i])
		if err == nil {
			t.facts[i] = decodeFacts(messages[i])
		}
		return role, toolCallID, calls, err
	})
	if err != nil {
		return nil, err
	}
	return t, nil
}

// PlainRole is the role of message i, read without decoding it, when the
// message is in the shape 2ai writes; ok is false for any other message:
// decode that one to learn its role.
func (t *Transcript) PlainRole(i int) (role string, ok bool) {
	return t.facts[i].role, t.scanned[i]
}

// Learn gives the compactor what t read, so a request over these messages
// does not read them again.
func (c *Compactor) Learn(t *Transcript) {
	if c.memo.facts == nil {
		c.memo.seed, c.memo.facts = maphash.MakeSeed(), make(map[uint64]*messageFacts, len(t.messages))
	}
	for i, raw := range t.messages {
		key := c.memo.mark(raw)
		if c.memo.facts[key] == nil {
			c.memo.facts[key] = &t.facts[i]
		}
	}
}

// ValidateMessageList is ValidateMessages over messages already split, such
// as the stored elements SplitMessages returned plus a run's appended ones.
// Called on the messages a run appended after a closed history, it is the
// check of the whole transcript: a closed batch leaves nothing open.
func ValidateMessageList(messages []json.RawMessage) error { return validateMessages(messages) }

func validateMessages(messages []json.RawMessage) error {
	return validateBatches(len(messages), func(i int) (string, string, []string, error) { return batchFacts(messages[i]) })
}

// validateBatches checks the tool batches of n messages, reading message i's
// role, answered call and opened calls with read.
func validateBatches(n int, read func(i int) (role, toolCallID string, calls []string, err error)) error {
	// Results must finish their own assistant batch. A reused provider call ID
	// in a later batch cannot settle an earlier missing result.
	open := map[string]int{}
	for i := range n {
		role, toolCallID, calls, err := read(i)
		if err != nil {
			return err
		}
		if role != "toolResult" && len(open) > 0 {
			return fmt.Errorf("%w: incomplete tool batch", ErrUnsettledEffect)
		}
		for _, id := range calls {
			open[id]++
		}
		if role == "toolResult" {
			if open[toolCallID] > 1 {
				open[toolCallID]--
			} else {
				delete(open, toolCallID)
			}
		}
	}
	if len(open) > 0 {
		return fmt.Errorf("%w: tool results need recovery", ErrUnsettledEffect)
	}
	return nil
}

// batchFacts is what batch validation reads from one message: its role, the
// call a result answers and the calls a settled assistant message opens. A
// message in the shape 2ai writes is scanned; any other is decoded.
func batchFacts(raw json.RawMessage) (role, toolCallID string, calls []string, err error) {
	if f, ok := scanFacts(raw); ok {
		return f.role, f.toolCallID, f.calls, nil
	}
	var m struct {
		Role, ToolCallID, StopReason string
		Content                      json.RawMessage
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return "", "", nil, err
	}
	// Pi exits the turn before tool admission on error/aborted messages.
	// Their partial calls remain in history but have no native results to
	// await. Physical effects are checked independently above; this cannot
	// clear an effect that started before a provider failure. A length stop
	// still produces failure results, so it must finish its result batch.
	if m.Role == "assistant" && m.StopReason != "error" && m.StopReason != "aborted" {
		var blocks []struct{ Type, ID string }
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return "", "", nil, err
		}
		for _, block := range blocks {
			if block.Type == "toolCall" {
				calls = append(calls, block.ID)
			}
		}
	}
	return m.Role, m.ToolCallID, calls, nil
}
