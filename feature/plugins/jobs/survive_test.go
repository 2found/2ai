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

// ends collects the Hold's callbacks.
type ends struct {
	mu  sync.Mutex
	got []Ended
	ch  chan Ended
}

func newEnds() *ends { return &ends{ch: make(chan Ended, 16)} }

func (e *ends) record(x Ended) {
	e.mu.Lock()
	e.got = append(e.got, x)
	e.mu.Unlock()
	e.ch <- x
}

func (e *ends) count() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.got)
}

func (e *ends) next(t *testing.T) Ended {
	t.Helper()
	select {
	case x := <-e.ch:
		return x
	case <-time.After(5 * time.Second):
		t.Fatal("the Hold's callback never fired")
		return Ended{}
	}
}

// run begins a board for one run of a conversation and returns the context the
// host would pass to RunNative.
func run(t *testing.T, s *Survival, conv string, opts ...func(*Plugin)) (*board, context.Context) {
	t.Helper()
	ctx := WithConversation(context.Background(), conv)
	p := Plugin{Survive: s}
	for _, o := range opts {
		o(&p)
	}
	ext, err := p.BeginRun(ctx, agentcore.RunInfo{Limits: agentcore.DefaultLimits()})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	return ext.(*board), ctx
}

func keepJob(t *testing.T, b *board, id string) {
	t.Helper()
	if _, err := (jobKeepTool{jobTool{b, "job_keep"}}).Run(context.Background(), `{"id":"`+id+`"}`); err != nil {
		t.Fatalf("job_keep %s: %v", id, err)
	}
}

func toolNames(b *board) []string {
	var names []string
	for _, tool := range b.Tools() {
		names = append(names, tool.Name())
	}
	return names
}

// Without the option nothing changes: the same four tools, and a job a run
// started is cancelled when the run ends.
func TestSurvivalOffIsTheOldBehaviour(t *testing.T) {
	b := begin(t, Plugin{})
	if got := strings.Join(toolNames(b), ","); got != "job_list,job_status,job_wait,job_cancel" {
		t.Fatalf("tools = %s", got)
	}
	if b.survive != nil || b.conv != "" {
		t.Fatal("survival state set without the option")
	}
	r, _ := b.start(context.Background(), "t", "long", block)
	b.CloseRun()
	if got, _ := b.get(r.ID); got.State != cancelled {
		t.Fatalf("job outlived its run: %+v", got)
	}
}

func TestSurviveRequiresAHold(t *testing.T) {
	if _, err := (Plugin{Survive: &Survival{}}).BeginRun(context.Background(), agentcore.RunInfo{}); err == nil {
		t.Fatal("Survive without a Hold accepted")
	}
	if _, err := (Plugin{Survive: &Survival{Hold: NewHold(nil), MaxAge: -1}}).BeginRun(context.Background(), agentcore.RunInfo{}); err == nil {
		t.Fatal("negative MaxAge accepted")
	}
}

// A job the model did not keep is cancelled at the end of the run, even with
// the option on; a kept one is not.
func TestOnlyAKeptJobSurvivesItsRun(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-1")
	if got := strings.Join(toolNames(b), ","); !strings.HasSuffix(got, ",job_keep") {
		t.Fatalf("tools = %s", got)
	}
	forgotten, _ := b.start(context.Background(), "run_shell", "forgotten", block)
	kept, _ := b.start(context.Background(), "run_shell", "dev server", block)
	keepJob(t, b, kept.ID)
	b.CloseRun()

	if got, _ := b.get(forgotten.ID); got.State != cancelled {
		t.Fatalf("a forgotten job survived: %+v", got)
	}
	next, _ := run(t, s, "conv-1")
	defer next.CloseRun()
	got, err := next.get(kept.ID)
	if err != nil || got.State != running || !got.Keep {
		t.Fatalf("kept job in the next run = %+v %v", got, err)
	}
	if _, err := next.get(forgotten.ID); !errors.Is(err, errNotFound) {
		t.Fatalf("a forgotten job reached the next run: %v", err)
	}
	// The next run observes it, then cancels it.
	out, _ := (jobListTool{jobTool{next, "job_list"}}).Run(context.Background(), "")
	if !strings.Contains(out, kept.ID+"\trunning (kept)") {
		t.Fatalf("list = %q", out)
	}
	if _, err := (jobCancelTool{jobTool{next, "job_cancel"}}).Run(context.Background(), `{"id":"`+kept.ID+`"}`); err != nil {
		t.Fatal(err)
	}
	if got := settled(t, next, kept.ID); got.State != cancelled {
		t.Fatalf("cancelled from the next run: %+v", got)
	}
}

// The work's context outlives the run that started it, yet still carries that
// run's values, and a new run adds nothing to it.
func TestKeptJobKeepsItsOwnRunContext(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-ctx")
	type key struct{}
	startCtx, stopRun := context.WithCancel(context.WithValue(context.Background(), key{}, "run one"))
	var seen atomicString
	release := make(chan struct{})
	r, err := b.start(startCtx, "run_shell", "watch", func(ctx context.Context) (string, error) {
		<-release
		seen.set(ctx.Value(key{}))
		return "", ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	keepJob(t, b, r.ID)
	stopRun() // the run's context ends, as it does when a run is torn down
	b.CloseRun()

	next, nextCtx := run(t, s, "conv-ctx")
	defer next.CloseRun()
	_ = context.WithValue(nextCtx, key{}, "run two")
	if got, _ := next.get(r.ID); got.State != running {
		t.Fatalf("kept job did not survive its run's context ending: %+v", got)
	}
	close(release)
	if got := settled(t, next, r.ID); got.State != succeeded {
		t.Fatalf("job = %+v", got)
	}
	if seen.get() != "run one" {
		t.Fatalf("job saw %v, want the starting run's value", seen.get())
	}
}

type atomicString struct {
	mu sync.Mutex
	v  any
}

func (a *atomicString) set(v any) { a.mu.Lock(); a.v = v; a.mu.Unlock() }
func (a *atomicString) get() any  { a.mu.Lock(); defer a.mu.Unlock(); return a.v }

// A job that is not kept still dies with its run's context, as before.
func TestUnkeptJobDiesWithTheRunContext(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-die")
	runCtx, stop := context.WithCancel(context.Background())
	r, _ := b.start(runCtx, "run_shell", "x", block)
	stop()
	if got := settled(t, b, r.ID); got.State != cancelled {
		t.Fatalf("job = %+v", got)
	}
}

func TestKeptJobIsInvisibleToAnotherConversation(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-a")
	r, _ := b.start(context.Background(), "run_shell", "secret server", block)
	keepJob(t, b, r.ID)
	b.CloseRun()

	other, _ := run(t, s, "conv-b")
	defer other.CloseRun()
	if _, err := other.get(r.ID); !errors.Is(err, errNotFound) {
		t.Fatalf("another conversation read the job: %v", err)
	}
	if _, err := (jobCancelTool{jobTool{other, "job_cancel"}}).Run(context.Background(), `{"id":"`+r.ID+`"}`); !errors.Is(err, errNotFound) {
		t.Fatalf("another conversation cancelled the job: %v", err)
	}
	if len(other.list()) != 0 || len(other.drain()) != 0 {
		t.Fatal("another conversation sees or is told about the job")
	}
	home, _ := run(t, s, "conv-a")
	defer home.CloseRun()
	if got, _ := home.get(r.ID); got.State != running {
		t.Fatalf("the owner lost its job: %+v", got)
	}
}

// A job that ends while a run of its conversation is attached is announced by
// that run, once, and never through the callback.
func TestAnnouncedByTheAttachedRunOnly(t *testing.T) {
	c := newEnds()
	s := &Survival{Hold: NewHold(c.record)}
	b, _ := run(t, s, "conv-att")
	r, _ := b.start(context.Background(), "run_shell", "short", block)
	keepJob(t, b, r.ID)
	b.CloseRun()

	next, _ := run(t, s, "conv-att")
	defer next.CloseRun()
	cancelJob(t, next, r.ID)
	settled(t, next, r.ID)
	if n := len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts); n != 1 {
		t.Fatalf("attached run announced %d notices, want 1", n)
	}
	if n := len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts); n != 0 {
		t.Fatal("re-announced")
	}
	if c.count() != 0 {
		t.Fatal("the callback fired for a job an attached run owns")
	}
}

func cancelJob(t *testing.T, b *board, id string) {
	t.Helper()
	if _, err := (jobCancelTool{jobTool{b, "job_cancel"}}).Run(context.Background(), `{"id":"`+id+`"}`); err != nil {
		t.Fatal(err)
	}
}

// A job that ends with no run attached is announced through the callback, once,
// and the next run does not announce it again, though it can still read it.
func TestAnnouncedByTheCallbackOnly(t *testing.T) {
	c := newEnds()
	s := &Survival{Hold: NewHold(c.record)}
	b, _ := run(t, s, "conv-cb")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "build", func(ctx context.Context) (string, error) {
		<-finish
		return "build ok", nil
	})
	keepJob(t, b, r.ID)
	b.CloseRun()

	close(finish)
	got := c.next(t)
	if got.Conversation != "conv-cb" || got.JobID != r.ID || got.State != "succeeded" || got.Tool != "run_shell" ||
		!strings.Contains(got.Notice, "build ok") || !strings.Contains(got.Notice, "Background job update") {
		t.Fatalf("Ended = %+v", got)
	}
	next, _ := run(t, s, "conv-cb")
	defer next.CloseRun()
	if n := len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts); n != 0 {
		t.Fatalf("the next run re-announced a job the callback carried (%d)", n)
	}
	if out := status(t, next, `{"id":"`+r.ID+`"}`); !strings.Contains(out, "build ok") {
		t.Fatalf("the finished result is unreadable: %q", out)
	}
	time.Sleep(20 * time.Millisecond)
	if c.count() != 1 {
		t.Fatalf("callback fired %d times", c.count())
	}
}

// A kept job that had already ended, unannounced, when its run closed is told
// to the host then: no run is left to carry the notice.
func TestEndedJobUnannouncedAtCloseGoesThroughTheCallback(t *testing.T) {
	c := newEnds()
	s := &Survival{Hold: NewHold(c.record)}
	b, _ := run(t, s, "conv-late")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "build", func(context.Context) (string, error) { <-finish; return "done", nil })
	keepJob(t, b, r.ID)
	close(finish)
	settled(t, b, r.ID)
	b.CloseRun()
	if got := c.next(t); got.JobID != r.ID {
		t.Fatalf("Ended = %+v", got)
	}
	next, _ := run(t, s, "conv-late")
	defer next.CloseRun()
	if n := len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts); n != 0 {
		t.Fatal("announced twice")
	}
}

// However a job's end races a run attaching, exactly one of the two announces
// it.
func TestExactlyOnceUnderRaces(t *testing.T) {
	for range 200 {
		c := newEnds()
		s := &Survival{Hold: NewHold(c.record)}
		b, _ := run(t, s, "conv-race")
		finish := make(chan struct{})
		r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { <-finish; return "ok", nil })
		keepJob(t, b, r.ID)
		b.CloseRun()

		var wg sync.WaitGroup
		var notices int
		wg.Add(2)
		go func() { defer wg.Done(); close(finish) }()
		go func() {
			defer wg.Done()
			next, _ := run(t, s, "conv-race")
			defer next.CloseRun()
			settled(t, next, r.ID)
			notices += len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts)
		}()
		wg.Wait()
		// The callback runs on the job goroutine, a moment after it ends.
		time.Sleep(time.Millisecond)
		if total := notices + c.count(); total != 1 {
			t.Fatalf("announced %d times (run %d, callback %d)", total, notices, c.count())
		}
	}
}

func TestMaxJobsIsEnforcedPerConversation(t *testing.T) {
	s := &Survival{Hold: NewHold(nil), MaxJobs: 2}
	b, _ := run(t, s, "conv-max")
	defer b.CloseRun()
	var ids []string
	for range 3 {
		r, _ := b.start(context.Background(), "run_shell", "svc", block)
		ids = append(ids, r.ID)
	}
	keepJob(t, b, ids[0])
	keepJob(t, b, ids[1])
	_, err := (jobKeepTool{jobTool{b, "job_keep"}}).Run(context.Background(), `{"id":"`+ids[2]+`"}`)
	if err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("third keep = %v", err)
	}
	// The ceiling counts running kept jobs, so cancelling one frees a place, and
	// another conversation has its own.
	cancelJob(t, b, ids[0])
	settled(t, b, ids[0])
	keepJob(t, b, ids[2])
	other, _ := run(t, s, "conv-other")
	defer other.CloseRun()
	r, _ := other.start(context.Background(), "run_shell", "svc", block)
	keepJob(t, other, r.ID)
}

func TestMaxAgeCancelsAKeptJob(t *testing.T) {
	c := newEnds()
	s := &Survival{Hold: NewHold(c.record), MaxAge: 80 * time.Millisecond}
	b, _ := run(t, s, "conv-age")
	r, _ := b.start(context.Background(), "run_shell", "forever", block)
	keepJob(t, b, r.ID)
	b.CloseRun()
	got := c.next(t)
	if got.State != "cancelled" || !strings.Contains(got.Notice, "maximum age") {
		t.Fatalf("Ended = %+v", got)
	}
	next, _ := run(t, s, "conv-age")
	defer next.CloseRun()
	if j, _ := next.get(r.ID); j.State != cancelled || !strings.Contains(j.Err, "maximum age") {
		t.Fatalf("job = %+v", j)
	}
	// An age already used up counts from the job's start, not from the keep.
	old, _ := next.start(context.Background(), "run_shell", "old", block)
	time.Sleep(100 * time.Millisecond)
	keepJob(t, next, old.ID)
	if j := settled(t, next, old.ID); j.State != cancelled {
		t.Fatalf("an over-age job was kept: %+v", j)
	}
}

func TestFinishedKeptRecordsExpire(t *testing.T) {
	s := &Survival{Hold: NewHold(func(Ended) {}), MaxAge: 50 * time.Millisecond}
	b, _ := run(t, s, "conv-exp")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { <-finish; return "ok", nil })
	keepJob(t, b, r.ID)
	b.CloseRun()
	close(finish)
	time.Sleep(200 * time.Millisecond)
	next, _ := run(t, s, "conv-exp")
	defer next.CloseRun()
	if _, err := next.get(r.ID); !errors.Is(err, errNotFound) {
		t.Fatalf("an expired finished record was handed to a later run: %v", err)
	}
}

func TestKeepRefusals(t *testing.T) {
	s := &Survival{Hold: NewHold(nil), Exclude: []string{"agent_call"}}
	b, _ := run(t, s, "conv-ref")
	defer b.CloseRun()
	keep := jobKeepTool{jobTool{b, "job_keep"}}
	sub, _ := b.start(context.Background(), "spawn_subagent", "child", block)
	host, _ := b.start(context.Background(), "agent_call", "model", block)
	done, _ := b.start(context.Background(), "run_shell", "quick", func(context.Context) (string, error) { return "", nil })
	settled(t, b, done.ID)
	for name, tc := range map[string]struct{ id, want string }{
		"sub-agent":     {sub.ID, "usage"},
		"excluded tool": {host.ID, "model-backed"},
		"finished job":  {done.ID, "already finished"},
		"unknown job":   {"job_99", "not found"},
		"missing id":    {"", "id is required"},
	} {
		_, err := keep.Run(context.Background(), `{"id":"`+tc.id+`"}`)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %v, want %q", name, err, tc.want)
		}
	}
	if got, _ := b.get(sub.ID); got.Keep {
		t.Fatal("a sub-agent was marked kept")
	}
	b.CloseRun()
	if got, _ := b.get(sub.ID); got.State != cancelled {
		t.Fatalf("the sub-agent outlived its run: %+v", got)
	}
}

func TestKeepNeedsAConversation(t *testing.T) {
	ext, err := (Plugin{Survive: &Survival{Hold: NewHold(nil)}}).BeginRun(context.Background(), agentcore.RunInfo{Limits: agentcore.DefaultLimits()})
	if err != nil {
		t.Fatal(err)
	}
	b := ext.(*board)
	r, _ := b.start(context.Background(), "run_shell", "x", block)
	if _, err := (jobKeepTool{jobTool{b, "job_keep"}}).Run(context.Background(), `{"id":"`+r.ID+`"}`); err == nil || !strings.Contains(err.Error(), "conversation") {
		t.Fatalf("keep without identity = %v", err)
	}
	b.CloseRun()
	if got, _ := b.get(r.ID); got.State != cancelled {
		t.Fatalf("job = %+v", got)
	}
	// A durable session id is an identity too.
	durable, err := (Plugin{Survive: &Survival{Hold: NewHold(nil)}}).BeginRun(context.Background(), agentcore.RunInfo{SessionID: "sess-1", Limits: agentcore.DefaultLimits()})
	if err != nil || durable.(*board).conv != "sess-1" {
		t.Fatalf("session identity: %v", err)
	}
}

func TestTwoRunsOfOneConversationCannotBeAttached(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	first, _ := run(t, s, "conv-dup")
	_, err := (Plugin{Survive: s}).BeginRun(WithConversation(context.Background(), "conv-dup"), agentcore.RunInfo{})
	if err == nil {
		t.Fatal("a second attached run was accepted")
	}
	first.CloseRun()
	next, _ := run(t, s, "conv-dup")
	next.CloseRun()
}

// A kept job does not hold up the final answer the way an unkept one does.
func TestKeptJobDoesNotBlockTheFinalAnswer(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-stop")
	defer b.CloseRun()
	b.stopWait = 20 * time.Millisecond
	kept, _ := b.start(context.Background(), "run_shell", "server", block)
	keepJob(t, b, kept.ID)
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue || d.StopReason != "" {
		t.Fatalf("a kept job blocked the answer: %+v", d)
	}
	forgotten, _ := b.start(context.Background(), "run_shell", "forgotten", block)
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); !d.Continue || !strings.Contains(d.Inject[0].Content, "pending") {
		t.Fatalf("an unkept job no longer blocks the answer: %+v", d)
	}
	_ = forgotten
}

// The process dies: the Hold is gone, and the restored checkpoint reports the
// kept job as interrupted, once.
func TestRestartReportsAKeptJobInterrupted(t *testing.T) {
	first := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, first, "conv-restart")
	r, _ := b.start(context.Background(), "run_shell", "dev server", block)
	keepJob(t, b, r.ID)
	state, err := b.NativeState() // taken before CloseRun, as the loop does
	if err != nil {
		t.Fatal(err)
	}
	b.CloseRun()

	restarted := &Survival{Hold: NewHold(nil)} // a new process: nothing held
	next, _ := run(t, restarted, "conv-restart")
	defer next.CloseRun()
	if err := next.RestoreNativeState(state); err != nil {
		t.Fatal(err)
	}
	got, _ := next.get(r.ID)
	if got.State != cancelled || !strings.Contains(got.Err, "interrupted") {
		t.Fatalf("job = %+v", got)
	}
	notice := next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts
	if len(notice) != 1 || !strings.Contains(notice[0].Content, "interrupted") {
		t.Fatalf("notice = %+v", notice)
	}
	if len(next.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts) != 0 {
		t.Fatal("announced twice")
	}
}

// The process lives on: restoring the checkpoint of the previous run finds the
// kept job running, not interrupted, and does not duplicate it.
func TestRestoreAdoptsTheLiveKeptJob(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-live")
	r, _ := b.start(context.Background(), "run_shell", "dev server", block)
	other, _ := b.start(context.Background(), "run_shell", "forgotten", block)
	keepJob(t, b, r.ID)
	state, _ := b.NativeState()
	b.CloseRun()

	next, _ := run(t, s, "conv-live")
	defer next.CloseRun()
	if err := next.RestoreNativeState(state); err != nil {
		t.Fatal(err)
	}
	if got, _ := next.get(r.ID); got.State != running || !got.Keep {
		t.Fatalf("kept job = %+v", got)
	}
	if got, _ := next.get(other.ID); got.State != cancelled {
		t.Fatalf("forgotten job = %+v", got)
	}
	if n := len(next.list()); n != 2 {
		t.Fatalf("restored board holds %d records, want 2", n)
	}
	// New ids continue after everything the transcript names.
	fresh, _ := next.start(context.Background(), "t", "new", func(context.Context) (string, error) { return "", nil })
	if fresh.ID == r.ID || fresh.ID == other.ID {
		t.Fatalf("id %s reused", fresh.ID)
	}
	settled(t, next, fresh.ID)
}

// Without a checkpoint to restore, ids still do not collide with a kept job.
func TestIDsContinueAfterAdoptedJobs(t *testing.T) {
	s := &Survival{Hold: NewHold(nil)}
	b, _ := run(t, s, "conv-ids")
	var last string
	for range 3 {
		r, _ := b.start(context.Background(), "t", "x", block)
		last = r.ID
	}
	keepJob(t, b, last)
	b.CloseRun()
	next, _ := run(t, s, "conv-ids")
	defer next.CloseRun()
	fresh, _ := next.start(context.Background(), "t", "new", block)
	if fresh.ID == "job_1" || fresh.ID == "job_2" || fresh.ID == "job_3" {
		t.Fatalf("id %s collides", fresh.ID)
	}
}

func TestHoldCancelAll(t *testing.T) {
	h := NewHold(func(Ended) {})
	s := &Survival{Hold: h}
	b, _ := run(t, s, "conv-shut")
	r, _ := b.start(context.Background(), "run_shell", "x", block)
	keepJob(t, b, r.ID)
	b.CloseRun()
	h.CancelAll()
	next, _ := run(t, s, "conv-shut")
	defer next.CloseRun()
	if got := settled(t, next, r.ID); got.State != cancelled {
		t.Fatalf("job = %+v", got)
	}
}

// A panicking host callback does not take the job goroutine down.
func TestCallbackPanicIsContained(t *testing.T) {
	called := make(chan struct{})
	s := &Survival{Hold: NewHold(func(Ended) { close(called); panic("host bug") })}
	b, _ := run(t, s, "conv-panic")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { <-finish; return "", nil })
	keepJob(t, b, r.ID)
	b.CloseRun()
	close(finish)
	<-called
}

// heldConversations is how many conversations the Hold still keeps anything for.
func heldConversations(h *Hold) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.convs)
}

// Once the run the host started for a kept job's notice has read it and closed,
// the Hold keeps nothing for that conversation: the run's checkpoint carries the
// record. Before, the finished record and the conversation's entry stayed in the
// Hold until the process exited unless a later run came back after MaxAge.
func TestHoldForgetsAKeptJobOnceItsNoticeRunCloses(t *testing.T) {
	c := newEnds()
	h := NewHold(c.record)
	s := &Survival{Hold: h}
	b, _ := run(t, s, "conv-forget")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "build", func(context.Context) (string, error) {
		<-finish
		return "build ok", nil
	})
	keepJob(t, b, r.ID)
	b.CloseRun()
	close(finish)
	c.next(t)

	notice, _ := run(t, s, "conv-forget")
	if out := status(t, notice, `{"id":"`+r.ID+`"}`); !strings.Contains(out, "build ok") {
		t.Fatalf("the notice run cannot read the result: %q", out)
	}
	notice.CloseRun()
	if n := heldConversations(h); n != 0 {
		t.Fatalf("the Hold still keeps %d conversation(s) after the notice run closed", n)
	}
}

// A kept job that ends while its own run is attached is announced by that run
// and not handed to the Hold when the run closes.
func TestHoldDoesNotKeepAJobItsRunAnnounced(t *testing.T) {
	h := NewHold(func(Ended) { t.Error("the callback fired for a job its run announced") })
	s := &Survival{Hold: h}
	b, _ := run(t, s, "conv-own")
	r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { return "ok", nil })
	keepJob(t, b, r.ID)
	if _, err := b.wait(context.Background(), r.ID); err != nil {
		t.Fatal(err)
	}
	if got := b.drain(); len(got) != 1 {
		t.Fatalf("drain = %d records", len(got))
	}
	b.CloseRun()
	if n := heldConversations(h); n != 0 {
		t.Fatalf("the Hold keeps %d conversation(s) for a job its run announced", n)
	}
}

// A kept job announced through the callback that no run ever comes back for
// (the host could not deliver the notice, the conversation is gone) leaves the
// Hold once it is older than MaxAge.
func TestHoldExpiresAFinishedJobNoRunCollects(t *testing.T) {
	c := newEnds()
	h := NewHold(c.record)
	s := &Survival{Hold: h, MaxAge: 50 * time.Millisecond}
	for _, conv := range []string{"conv-a", "conv-b", "conv-c"} {
		b, _ := run(t, s, conv)
		finish := make(chan struct{})
		r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { <-finish; return "ok", nil })
		keepJob(t, b, r.ID)
		b.CloseRun()
		close(finish)
		c.next(t)
	}
	if n := heldConversations(h); n != 3 {
		t.Fatalf("held = %d right after the callbacks, want 3 (the notice run must still find them)", n)
	}
	deadline := time.Now().Add(2 * time.Second)
	for heldConversations(h) != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("the Hold still keeps %d expired conversation(s)", heldConversations(h))
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A run that adopts a finished job before its expiry keeps it: the expiry does
// not pull the record out from under the attached run.
func TestExpiryLeavesAnAdoptedRecordAlone(t *testing.T) {
	c := newEnds()
	h := NewHold(c.record)
	s := &Survival{Hold: h, MaxAge: 40 * time.Millisecond}
	b, _ := run(t, s, "conv-adopt")
	finish := make(chan struct{})
	r, _ := b.start(context.Background(), "run_shell", "x", func(context.Context) (string, error) { <-finish; return "kept result", nil })
	keepJob(t, b, r.ID)
	b.CloseRun()
	close(finish)
	c.next(t)
	next, _ := run(t, s, "conv-adopt")
	defer next.CloseRun()
	time.Sleep(150 * time.Millisecond)
	if out := status(t, next, `{"id":"`+r.ID+`"}`); !strings.Contains(out, "kept result") {
		t.Fatalf("the adopted record vanished: %q", out)
	}
}
