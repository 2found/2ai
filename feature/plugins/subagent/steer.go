package subagent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/2found/2ai/core/agentcore"
)

// maxPendingSteers bounds the messages a child may have queued and unread.
const maxPendingSteers = 8

// mailbox carries the parent's follow-up messages (job_send) to a running
// background child. The child's run drains it at each step boundary, before the
// model is asked again, and once more before it may finish; the loop persists
// whatever is drained as user messages of the child's own transcript.
type mailbox struct {
	depth int // delegation depth of the child it serves

	mu      sync.Mutex
	queue   []string
	closed  bool
	claimed bool
}

type mailboxKey struct{}

// push queues a message; it is the handler the job registers.
func (m *mailbox) push(message string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.closed:
		return fmt.Errorf("the sub-agent has already finished and can no longer receive messages")
	case len(m.queue) >= maxPendingSteers:
		return fmt.Errorf("the sub-agent has %d unread messages; wait for it to read them before sending more", len(m.queue))
	}
	m.queue = append(m.queue, message)
	return nil
}

// drain takes the queued messages as the user messages the child reads.
func (m *mailbox) drain() []agentcore.Message {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]agentcore.Message, 0, len(m.queue))
	for _, text := range m.queue {
		out = append(out, agentcore.Message{Role: agentcore.RoleUser, Content: "Message from the parent agent, sent while you were working:\n" + text})
	}
	m.queue = nil
	return out
}

// close refuses further messages and returns the ones the child never read.
func (m *mailbox) close() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	left := m.queue
	m.queue = nil
	return left
}

// claim binds the mailbox to the one run of the child it serves: a run at that
// delegation depth, once. A grandchild inherits the context but is a different
// depth, so it never reads its ancestor's mail.
func (m *mailbox) claim(depth int) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.claimed || m.closed || m.depth != depth {
		return false
	}
	m.claimed = true
	return true
}

// mailboxFor returns the mailbox the context carries for a run at this depth.
func mailboxFor(ctx context.Context, depth int) *mailbox {
	if m, _ := ctx.Value(mailboxKey{}).(*mailbox); m != nil && m.claim(depth) {
		return m
	}
	return nil
}

// undelivered renders the note appended to a child's answer when messages
// arrived after its last step.
func undelivered(left []string) string {
	if len(left) == 0 {
		return ""
	}
	return fmt.Sprintf("\n\n[%d message(s) from you arrived after the sub-agent's last step and were not read: %s]", len(left), strings.Join(left, " | "))
}

// steerer is the extension of a child run that has nothing to offer but its
// mailbox (a child at the depth cap carries no spawn_subagent tool).
type steerer struct{ mail *mailbox }

func (steerer) Name() string { return ToolSpawnSubagent }

func (s steerer) BeforeStep(context.Context, agentcore.StepInfo) agentcore.StepDecision {
	return agentcore.StepDecision{AdditionalContexts: s.mail.drain()}
}

func (s steerer) TurnStopping(context.Context, agentcore.StopInfo) agentcore.StopDecision {
	return stopForMail(s.mail)
}

// stopForMail keeps a child from finishing past a message that arrived while
// it composed its final answer.
func stopForMail(m *mailbox) agentcore.StopDecision {
	msgs := m.drain()
	if len(msgs) == 0 {
		return agentcore.StopDecision{}
	}
	return agentcore.StopDecision{Continue: true, Inject: msgs}
}

// BeforeStep and TurnStopping make the spawn tool's own extension deliver its
// run's mailbox, for a child that can itself spawn. Both are no-ops for a run
// without one.
func (t *subagentTool) BeforeStep(context.Context, agentcore.StepInfo) agentcore.StepDecision {
	return agentcore.StepDecision{AdditionalContexts: t.mail.drain()}
}

func (t *subagentTool) TurnStopping(context.Context, agentcore.StopInfo) agentcore.StopDecision {
	return stopForMail(t.mail)
}
