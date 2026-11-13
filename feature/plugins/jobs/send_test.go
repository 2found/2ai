package jobs

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

func sendTool(b *board) jobSendTool { return jobSendTool{jobTool{b, "job_send"}} }

// job_send reaches the handler the work registered, only while the job runs,
// and says why when it cannot.
func TestJobSendDeliversAndRefuses(t *testing.T) {
	b := begin(t, Plugin{AllowSend: true})
	var mu sync.Mutex
	var got []string
	ready := make(chan struct{})
	child, _ := b.start(context.Background(), "spawn_subagent", "child", func(ctx context.Context) (string, error) {
		agentcore.HandleJobMessages(ctx, func(m string) error { mu.Lock(); got = append(got, m); mu.Unlock(); return nil })
		close(ready)
		<-ctx.Done()
		return "", ctx.Err()
	})
	shell, _ := b.start(context.Background(), "run_shell", "server", block)
	quick, _ := b.start(context.Background(), "t", "quick", func(context.Context) (string, error) { return "ok", nil })
	settled(t, b, quick.ID)
	<-ready

	if out, err := sendTool(b).Run(context.Background(), `{"id":"`+child.ID+`","message":"  switch to port 9000 "}`); err != nil || !strings.Contains(out, child.ID) {
		t.Fatalf("send = %q %v", out, err)
	}
	mu.Lock()
	if len(got) != 1 || got[0] != "switch to port 9000" {
		t.Fatalf("handler got %q", got)
	}
	mu.Unlock()

	for name, tc := range map[string]struct{ args, want string }{
		"not a sub-agent": {`{"id":"` + shell.ID + `","message":"hi"}`, "does not accept messages"},
		"finished":        {`{"id":"` + quick.ID + `","message":"hi"}`, "already finished"},
		"unknown":         {`{"id":"job_77","message":"hi"}`, "job not found"},
		"empty message":   {`{"id":"` + child.ID + `","message":"  "}`, "message is required"},
		"missing id":      {`{"message":"hi"}`, "id is required"},
		"too long":        {`{"id":"` + child.ID + `","message":"` + strings.Repeat("x", maxSendSize+1) + `"}`, "at most"},
		"not json":        {`{`, "invalid arguments"},
	} {
		if _, err := sendTool(b).Run(context.Background(), tc.args); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %v, want %q", name, err, tc.want)
		}
	}
	mu.Lock()
	n := len(got)
	mu.Unlock()
	if n != 1 {
		t.Fatalf("a refused send reached the handler (%d messages)", n)
	}
	cancelJob(t, b, child.ID)
	settled(t, b, child.ID)
	if _, err := sendTool(b).Run(context.Background(), `{"id":"`+child.ID+`","message":"hi"}`); err == nil || !strings.Contains(err.Error(), "cancelled") {
		t.Fatalf("send to a cancelled job = %v", err)
	}
	b.CloseRun()
}

// A handler's own refusal (a full queue) is returned to the model as is.
func TestJobSendReturnsTheHandlersRefusal(t *testing.T) {
	b := begin(t, Plugin{AllowSend: true})
	ready := make(chan struct{})
	r, _ := b.start(context.Background(), "spawn_subagent", "child", func(ctx context.Context) (string, error) {
		agentcore.HandleJobMessages(ctx, func(string) error { return errors.New("the sub-agent has 8 unread messages") })
		close(ready)
		<-ctx.Done()
		return "", ctx.Err()
	})
	<-ready
	if _, err := sendTool(b).Run(context.Background(), `{"id":"`+r.ID+`","message":"hi"}`); err == nil || !strings.Contains(err.Error(), "unread") {
		t.Fatalf("send = %v", err)
	}
	b.CloseRun()
}

// A job that has only just started may not have registered its handler yet; a
// send right behind the receipt still gets through.
func TestJobSendWaitsForAJustStartedHandler(t *testing.T) {
	b := begin(t, Plugin{AllowSend: true})
	delivered := make(chan string, 1)
	r, _ := b.start(context.Background(), "spawn_subagent", "child", func(ctx context.Context) (string, error) {
		time.Sleep(60 * time.Millisecond)
		agentcore.HandleJobMessages(ctx, func(m string) error { delivered <- m; return nil })
		<-ctx.Done()
		return "", ctx.Err()
	})
	if _, err := sendTool(b).Run(context.Background(), `{"id":"`+r.ID+`","message":"early"}`); err != nil {
		t.Fatalf("send = %v", err)
	}
	if m := <-delivered; m != "early" {
		t.Fatalf("delivered %q", m)
	}
	b.CloseRun()
}

func TestJobSendToolIsOffUnlessEnabled(t *testing.T) {
	if got := strings.Join(toolNames(begin(t, Plugin{})), ","); strings.Contains(got, "job_send") {
		t.Fatalf("tools = %s", got)
	}
	on := begin(t, Plugin{AllowSend: true})
	if got := strings.Join(toolNames(on), ","); !strings.HasSuffix(got, ",job_send") {
		t.Fatalf("tools = %s", got)
	}
	tool := sendTool(on)
	if !tool.SelfGated() || !tool.Bookkeeping() {
		t.Fatal("job_send is not governed like the other job tools")
	}
}
