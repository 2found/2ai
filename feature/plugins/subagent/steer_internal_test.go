package subagent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/2found/2ai/core/agentcore"
)

func TestMailboxDeliversEachMessageOnce(t *testing.T) {
	m := &mailbox{depth: 1}
	if err := m.push("one"); err != nil {
		t.Fatal(err)
	}
	if err := m.push("two"); err != nil {
		t.Fatal(err)
	}
	got := m.drain()
	if len(got) != 2 || got[0].Role != agentcore.RoleUser || !strings.HasSuffix(got[0].Content, "one") || !strings.HasSuffix(got[1].Content, "two") {
		t.Fatalf("drained %+v", got)
	}
	if again := m.drain(); len(again) != 0 {
		t.Fatalf("redelivered %+v", again)
	}
	var nilBox *mailbox
	if len(nilBox.drain()) != 0 {
		t.Fatal("a run without a mailbox received mail")
	}
}

func TestMailboxIsBoundedAndClosesWithALeftoverReport(t *testing.T) {
	m := &mailbox{depth: 1}
	for range maxPendingSteers {
		if err := m.push("x"); err != nil {
			t.Fatal(err)
		}
	}
	if err := m.push("one too many"); err == nil || !strings.Contains(err.Error(), "unread") {
		t.Fatalf("overflow = %v", err)
	}
	left := m.close()
	if len(left) != maxPendingSteers {
		t.Fatalf("leftover = %d", len(left))
	}
	if err := m.push("late"); err == nil || !strings.Contains(err.Error(), "already finished") {
		t.Fatalf("push after close = %v", err)
	}
	if note := undelivered([]string{"a", "b"}); !strings.Contains(note, "2 message(s)") || !strings.Contains(note, "a | b") {
		t.Fatalf("note = %q", note)
	}
	if undelivered(nil) != "" {
		t.Fatal("a note for no messages")
	}
}

// A grandchild inherits its parent's context but is another depth: it never
// reads, nor takes, its ancestor's mail; the child's own run claims it once.
func TestMailboxBindsToOneRunAtItsDepth(t *testing.T) {
	m := &mailbox{depth: 2}
	ctx := context.WithValue(context.Background(), mailboxKey{}, m)
	if mailboxFor(ctx, 3) != nil || mailboxFor(ctx, 1) != nil {
		t.Fatal("a run at another depth claimed the mailbox")
	}
	if mailboxFor(ctx, 2) != m {
		t.Fatal("the child's run could not claim its mailbox")
	}
	if mailboxFor(ctx, 2) != nil {
		t.Fatal("a mailbox was claimed twice")
	}
	if mailboxFor(context.Background(), 2) != nil {
		t.Fatal("a mailbox out of thin air")
	}
}

// A message queued while the child composes its final answer holds the finish
// for one more step, instead of being dropped with the run.
func TestFinalAnswerIsHeldForAQueuedMessage(t *testing.T) {
	m := &mailbox{depth: 1}
	s := steerer{m}
	if d := s.TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue {
		t.Fatalf("held with nothing queued: %+v", d)
	}
	_ = m.push("one more thing")
	d := s.TurnStopping(context.Background(), agentcore.StopInfo{})
	if !d.Continue || len(d.Inject) != 1 || !strings.Contains(d.Inject[0].Content, "one more thing") {
		t.Fatalf("decision = %+v", d)
	}
	if d := s.TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue {
		t.Fatal("held twice for one message")
	}
	// The same through the spawn tool's own extension, and a run without mail.
	m2 := &mailbox{depth: 1}
	_ = m2.push("hi")
	tool := &subagentTool{mail: m2}
	if got := tool.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts; len(got) != 1 {
		t.Fatalf("BeforeStep delivered %d", len(got))
	}
	if d := (&subagentTool{}).TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue {
		t.Fatal("a run without mail was held")
	}
}

func TestPushIsRaceFree(t *testing.T) {
	m := &mailbox{depth: 1}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				_ = m.push("x")
				m.drain()
			}
		}()
	}
	wg.Wait()
	m.close()
}
