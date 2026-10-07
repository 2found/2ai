package jobs

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func beginJobs(t *testing.T, p Plugin, owner string) *jobsRun {
	t.Helper()
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{Owner: owner, Limits: agentcore.DefaultLimits()})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	if ext == nil {
		t.Fatal("jobs must accept every run — it needs nothing from the session")
	}
	return ext.(*jobsRun)
}

func testJobsPolicy(t *testing.T, owner string) *jobsRun {
	t.Helper()
	p := beginJobs(t, Plugin{}, owner)
	if p == nil {
		t.Fatal("expected a jobs policy")
	}
	return p
}

func TestConfiguredConcurrencyAndUnlimited(t *testing.T) {
	for _, limit := range []int{2, -1} {
		p := beginJobs(t, Plugin{MaxConcurrent: limit}, "configured")
		ctx, cancel := context.WithCancel(context.Background())
		l := launcherFor(p, ctx)
		count := 20
		if limit > 0 {
			count = limit
		}
		for range count {
			if _, err := l.Start("work", "wait", func(ctx context.Context) (string, error) { <-ctx.Done(); return "", ctx.Err() }); err != nil {
				cancel()
				t.Fatal(err)
			}
		}
		if limit > 0 {
			if _, err := l.Start("extra", "wait", func(context.Context) (string, error) { return "", nil }); err == nil {
				t.Error("concurrency ceiling ignored")
			}
		}
		cancel()
		waitFor(t, func() bool {
			for _, j := range p.store.List(p.owner) {
				if !j.State.Done() {
					return false
				}
			}
			return true
		})
	}
}

func launcherFor(p *jobsRun, ctx context.Context) Launcher {
	return Launcher{store: p.store, owner: p.owner, base: ctx, maxBytes: p.maxBytes}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition not met within 2s")
}

func TestJobs_StartReturnsImmediatelyAndCompletes(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	l := launcherFor(p, context.Background())

	release := make(chan struct{})
	started := time.Now()
	job, err := l.Start("export", "big export", func(ctx context.Context) (string, error) {
		<-release
		return "42 rows", nil
	})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	// The whole point: Start does not block on the work.
	if time.Since(started) > 500*time.Millisecond {
		t.Fatal("Start blocked on the job")
	}
	if job.State != JobRunning {
		t.Fatalf("state = %s, want running", job.State)
	}

	close(release)
	waitFor(t, func() bool {
		j, _ := p.store.Get(p.owner, job.ID)
		return j.State == JobSucceeded
	})
	j, _ := p.store.Get(p.owner, job.ID)
	if j.Result != "42 rows" {
		t.Fatalf("result = %q", j.Result)
	}
	if j.EndedAt.IsZero() {
		t.Fatal("a finished job must record EndedAt")
	}
}

func TestJobs_FailureAndPanicBothBecomeFailedJobs(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	l := launcherFor(p, context.Background())

	bad, _ := l.Start("t", "boom", func(context.Context) (string, error) {
		return "", errors.New("query exploded")
	})
	// A panicking job must not take the process down with it.
	panicky, _ := l.Start("t", "panic", func(context.Context) (string, error) {
		panic("nil map write")
	})

	waitFor(t, func() bool {
		a, _ := p.store.Get(p.owner, bad.ID)
		b, _ := p.store.Get(p.owner, panicky.ID)
		return a.State.Done() && b.State.Done()
	})
	a, _ := p.store.Get(p.owner, bad.ID)
	if a.State != JobFailed || !strings.Contains(a.Err, "query exploded") {
		t.Fatalf("error job = %+v", a)
	}
	b, _ := p.store.Get(p.owner, panicky.ID)
	if b.State != JobFailed || !strings.Contains(b.Err, "panicked") {
		t.Fatalf("panicking job = %+v", b)
	}
}

func TestJobs_FencedByOwner(t *testing.T) {
	store := NewLocalJobStore()
	mine := &jobsRun{store: store, owner: "sess-1", maxWait: time.Second, maxBytes: 1024}
	theirs := &jobsRun{store: store, owner: "sess-2", maxWait: time.Second, maxBytes: 1024}

	job, err := launcherFor(mine, context.Background()).Start("t", "secret", func(context.Context) (string, error) {
		return "sensitive output", nil
	})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	waitFor(t, func() bool { j, _ := store.Get("sess-1", job.ID); return j.State.Done() })

	// Another session must not be able to read, wait on, or cancel it.
	if _, err := store.Get(theirs.owner, job.ID); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("cross-owner Get = %v, want ErrJobNotFound", err)
	}
	if err := store.Cancel(theirs.owner, job.ID); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("cross-owner Cancel = %v", err)
	}
	if _, err := store.Wait(context.Background(), theirs.owner, job.ID); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("cross-owner Wait = %v", err)
	}
	if got := store.List(theirs.owner); len(got) != 0 {
		t.Fatalf("cross-owner List returned %d jobs", len(got))
	}
	// And the notice drain is per-owner too.
	if got := store.DrainCompleted(theirs.owner); len(got) != 0 {
		t.Fatalf("cross-owner drain returned %d jobs", len(got))
	}
	if got := store.DrainCompleted("sess-1"); len(got) != 1 {
		t.Fatalf("owner drain returned %d jobs, want 1", len(got))
	}
}

func TestJobs_CancelStopsTheWork(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	l := launcherFor(p, context.Background())

	job, _ := l.Start("t", "long", func(ctx context.Context) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	})
	if err := p.store.Cancel(p.owner, job.ID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	waitFor(t, func() bool { j, _ := p.store.Get(p.owner, job.ID); return j.State.Done() })
	j, _ := p.store.Get(p.owner, job.ID)
	if j.State != JobCancelled {
		t.Fatalf("state = %s, want cancelled", j.State)
	}
	// Cancelling again is a no-op, not an error.
	if err := p.store.Cancel(p.owner, job.ID); err != nil {
		t.Fatalf("second Cancel: %v", err)
	}
}

func TestJobs_CancelAllStopsEverythingForAnOwner(t *testing.T) {
	store := NewLocalJobStore()
	mine := &jobsRun{store: store, owner: "sess-1", maxWait: time.Second, maxBytes: 1024}
	theirs := &jobsRun{store: store, owner: "sess-2", maxWait: time.Second, maxBytes: 1024}

	block := func(ctx context.Context) (string, error) { <-ctx.Done(); return "", ctx.Err() }
	a, _ := launcherFor(mine, context.Background()).Start("t", "a", block)
	b, _ := launcherFor(theirs, context.Background()).Start("t", "b", block)

	store.CancelAll("sess-1")
	waitFor(t, func() bool { j, _ := store.Get("sess-1", a.ID); return j.State.Done() })

	// The other owner's job is untouched — CancelAll is a run boundary, not a
	// process-wide kill switch.
	other, _ := store.Get("sess-2", b.ID)
	if other.State.Done() {
		t.Fatal("CancelAll leaked across owners")
	}
	store.CancelAll("sess-2")
}

func TestJobs_DrainCompletedIsOneShot(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	l := launcherFor(p, context.Background())
	job, _ := l.Start("t", "quick", func(context.Context) (string, error) { return "done", nil })
	waitFor(t, func() bool { j, _ := p.store.Get(p.owner, job.ID); return j.State.Done() })

	first := p.store.DrainCompleted(p.owner)
	if len(first) != 1 {
		t.Fatalf("first drain returned %d, want 1", len(first))
	}
	// A notice must be delivered exactly once, or every later turn re-announces
	// the same finished job.
	if second := p.store.DrainCompleted(p.owner); len(second) != 0 {
		t.Fatalf("second drain returned %d, want 0", len(second))
	}
}

func TestJobs_ResultIsBoundedBeforeItReachesTheModel(t *testing.T) {
	p := beginJobs(t, Plugin{MaxResultBytes: 200}, "sess-1")
	l := launcherFor(p, context.Background())
	job, _ := l.Start("t", "huge", func(context.Context) (string, error) {
		return strings.Repeat("x", 100_000), nil
	})
	waitFor(t, func() bool { j, _ := p.store.Get(p.owner, job.ID); return j.State.Done() })
	j, _ := p.store.Get(p.owner, job.ID)
	if len(j.Result) > 200 {
		t.Fatalf("stored result is %d bytes — an unbounded job result blows the context", len(j.Result))
	}
}

func TestJobs_WaitReturnsWhileStillRunningWithoutCancelling(t *testing.T) {
	p := beginJobs(t, Plugin{MaxWaitSeconds: 1}, "sess-1")
	l := launcherFor(p, context.Background())
	release := make(chan struct{})
	job, _ := l.Start("t", "slow", func(ctx context.Context) (string, error) {
		select {
		case <-release:
			return "eventually", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})

	tool := jobWaitTool{jobTool{policy: p, name: "job_wait"}}
	out, err := tool.Run(context.Background(), `{"id":"`+job.ID+`","timeout_seconds":1}`)
	if err != nil {
		t.Fatalf("job_wait: %v", err)
	}
	if !strings.Contains(out, "still running") {
		t.Fatalf("expected a still-running answer, got %q", out)
	}
	// Timing out the wait must NOT cancel the job — that would silently destroy
	// work the model merely got impatient about.
	j, _ := p.store.Get(p.owner, job.ID)
	if j.State != JobRunning {
		t.Fatalf("wait timeout cancelled the job (state=%s)", j.State)
	}
	close(release)
	waitFor(t, func() bool { j, _ := p.store.Get(p.owner, job.ID); return j.State == JobSucceeded })
}

func TestJobs_ToolsRenderStateAndRejectUnknownIDs(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	l := launcherFor(p, context.Background())
	job, _ := l.Start("run_sql", "count rows", func(context.Context) (string, error) { return "1234", nil })
	waitFor(t, func() bool { j, _ := p.store.Get(p.owner, job.ID); return j.State.Done() })

	list := jobListTool{jobTool{policy: p, name: "job_list"}}
	out, err := list.Run(context.Background(), `{}`)
	if err != nil || !strings.Contains(out, job.ID) || !strings.Contains(out, "succeeded") {
		t.Fatalf("job_list = %q, %v", out, err)
	}

	status := jobStatusTool{jobTool{policy: p, name: "job_status"}}
	out, err = status.Run(context.Background(), `{"id":"`+job.ID+`"}`)
	if err != nil || !strings.Contains(out, "1234") {
		t.Fatalf("job_status = %q, %v", out, err)
	}
	if _, err := status.Run(context.Background(), `{"id":"job_nope"}`); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("unknown id = %v, want ErrJobNotFound", err)
	}
	if _, err := status.Run(context.Background(), `{}`); err == nil {
		t.Fatal("missing id must be an error")
	}

	cancel := jobCancelTool{jobTool{policy: p, name: "job_cancel"}}
	if _, err := cancel.Run(context.Background(), `{"id":"job_nope"}`); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("cancel unknown = %v", err)
	}
}

func TestJobs_EmptyListIsExplicit(t *testing.T) {
	p := testJobsPolicy(t, "sess-1")
	out, err := jobListTool{jobTool{policy: p, name: "job_list"}}.Run(context.Background(), `{}`)
	if err != nil || !strings.Contains(out, "No background jobs") {
		t.Fatalf("job_list on empty = %q, %v", out, err)
	}
}

func TestJobs_CompletionNoticeInlinesSmallResultsAndDefersLargeOnes(t *testing.T) {
	small := Job{ID: "job_1", Tool: "run_sql", State: JobSucceeded, Result: "7 rows",
		StartedAt: time.Now().Add(-time.Second), EndedAt: time.Now()}
	large := Job{ID: "job_2", Tool: "export", State: JobSucceeded, Result: strings.Repeat("y", 5000),
		StartedAt: time.Now().Add(-time.Second), EndedAt: time.Now()}
	failed := Job{ID: "job_3", Tool: "crawl", State: JobFailed, Err: "host unreachable",
		StartedAt: time.Now().Add(-time.Second), EndedAt: time.Now()}

	notice := completionNotice([]Job{small, large, failed}, 2048)
	if !strings.Contains(notice, "7 rows") {
		t.Fatal("a small result should ride inline — otherwise the model spends a turn fetching it")
	}
	if strings.Contains(notice, strings.Repeat("y", 100)) {
		t.Fatal("a large result must not ride inline")
	}
	if !strings.Contains(notice, "job_status") {
		t.Fatal("a deferred result must tell the model how to fetch it")
	}
	if !strings.Contains(notice, "host unreachable") {
		t.Fatal("a failure must state why")
	}
	if completionNotice(nil, 2048) != "" {
		t.Fatal("no finished jobs means no notice")
	}
}

func TestJobs_LauncherWithoutStoreFailsLoud(t *testing.T) {
	var l Launcher
	if _, err := l.Start("t", "x", func(context.Context) (string, error) { return "", nil }); err == nil {
		t.Fatal("a tool starting a job on a run without the jobs seam must get an error")
	}
}

func TestJobs_SpecWithoutWorkIsRejected(t *testing.T) {
	store := NewLocalJobStore()
	if _, err := store.Start(context.Background(), "o", JobSpec{Tool: "t"}); err == nil {
		t.Fatal("a job with no work must be rejected")
	}
}

func TestJobs_FromContext(t *testing.T) {
	if _, ok := From(context.Background()); ok {
		t.Fatal("a run without jobs must not expose a launcher")
	}
	p := testJobsPolicy(t, "sess-1")
	ctx := withJobs(context.Background(), launcherFor(p, context.Background()))
	l, ok := From(ctx)
	if !ok {
		t.Fatal("expected a launcher on the context")
	}
	if l.owner != "sess-1" {
		t.Fatalf("launcher owner = %q — a tool must not be able to address another session", l.owner)
	}
}

func TestJobs_NilSettingsDisableTheSeam(t *testing.T) {
	if p := beginJobs(t, Plugin{}, "sess-1"); p == nil {
		t.Fatal("nil settings must leave jobs off")
	}
}

func TestJobs_NoticeOnByDefault(t *testing.T) {
	if p := beginJobs(t, Plugin{}, "s"); !p.notify {
		t.Fatal("completion notices must be on by default — polling is a turn tax")
	}
	if p := beginJobs(t, Plugin{SuppressCompletionNotice: true}, "s"); p.notify {
		t.Fatal("SuppressCompletionNotice must turn notices off")
	}
}

func TestLocalJobsBoundConcurrentWorkAndFenceOwners(t *testing.T) {
	store := NewLocalJobStore()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	work := JobSpec{Tool: "test", Run: func(ctx context.Context) (string, error) { <-ctx.Done(); return "", ctx.Err() }}
	for range 15 {
		if _, err := store.Start(ctx, "owner", work); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Start(ctx, "owner", work); err == nil {
		t.Fatal("concurrent capacity not enforced")
	}
	other, err := store.Start(ctx, "other", work)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get("owner", other.ID); err == nil {
		t.Fatal("foreign job visible")
	}
	store.CancelAll("owner")
	for _, job := range store.List("owner") {
		if _, err := store.Wait(context.Background(), "owner", job.ID); err != nil {
			t.Fatal(err)
		}
	}
	got, err := store.Get("other", other.ID)
	if err != nil || got.State != JobRunning {
		t.Fatal("cancellation crossed ownership")
	}
	store.CancelAll("other")
	_, _ = store.Wait(context.Background(), "other", other.ID)
}

func TestFinishWithPendingJobReturnsControlAfterBoundedWait(t *testing.T) {
	run := beginJobs(t, Plugin{}, "owner")
	run.maxWait = time.Millisecond
	defer run.CloseRun()
	_, err := run.store.Start(context.Background(), run.owner, JobSpec{Run: func(ctx context.Context) (string, error) { <-ctx.Done(); return "", ctx.Err() }})
	if err != nil {
		t.Fatal(err)
	}
	decision := run.TurnStopping(context.Background(), agentcore.StopInfo{})
	if !decision.Continue || decision.StopReason != "" || len(decision.Inject) != 1 {
		t.Fatalf("pending work accepted as final: %+v", decision)
	}
}
