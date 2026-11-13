package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

func begin(t *testing.T, p Plugin) *board {
	t.Helper()
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{Limits: agentcore.DefaultLimits()})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	return ext.(*board)
}

func settled(t *testing.T, b *board, id string) record {
	t.Helper()
	r, err := b.wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func block(ctx context.Context) (string, error) { <-ctx.Done(); return "", ctx.Err() }

func TestRejectsNegativeConcurrency(t *testing.T) {
	if _, err := (Plugin{MaxConcurrent: -2}).BeginRun(context.Background(), agentcore.RunInfo{}); err == nil {
		t.Fatal("MaxConcurrent below -1 accepted")
	}
}

func TestConfiguredConcurrencyAndUnlimited(t *testing.T) {
	for _, limit := range []int{0, 2, -1} {
		b := begin(t, Plugin{MaxConcurrent: limit})
		ctx, cancel := context.WithCancel(context.Background())
		count := map[int]int{0: defaultMaxConcurrent, 2: 2, -1: 20}[limit]
		for range count {
			if _, err := b.start(ctx, "work", "wait", block); err != nil {
				cancel()
				t.Fatal(err)
			}
		}
		_, err := b.start(ctx, "extra", "wait", block)
		if limit >= 0 && err == nil {
			t.Errorf("limit %d: concurrency ceiling ignored", limit)
		}
		if limit < 0 && err != nil {
			t.Errorf("unlimited board refused a job: %v", err)
		}
		cancel()
		for _, r := range b.list() {
			settled(t, b, r.ID)
		}
	}
}

func TestStartReturnsImmediatelyAndCompletes(t *testing.T) {
	b := begin(t, Plugin{})
	release := make(chan struct{})
	started := time.Now()
	r, err := b.start(context.Background(), "export", "big export", func(context.Context) (string, error) {
		<-release
		return "42 rows", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(started) > 500*time.Millisecond || r.State != running {
		t.Fatalf("start blocked on the work or returned %s", r.State)
	}
	close(release)
	got := settled(t, b, r.ID)
	if got.State != succeeded || got.Result != "42 rows" || got.EndedAt.IsZero() {
		t.Fatalf("finished job = %+v", got)
	}
}

func TestFailureAndPanicBothBecomeFailedJobs(t *testing.T) {
	b := begin(t, Plugin{})
	bad, _ := b.start(context.Background(), "t", "boom", func(context.Context) (string, error) { return "", errors.New("query exploded") })
	panicky, _ := b.start(context.Background(), "t", "panic", func(context.Context) (string, error) { panic("nil map write") })
	if r := settled(t, b, bad.ID); r.State != failed || !strings.Contains(r.Err, "query exploded") {
		t.Fatalf("error job = %+v", r)
	}
	if r := settled(t, b, panicky.ID); r.State != failed || !strings.Contains(r.Err, "panicked") {
		t.Fatalf("panicking job = %+v", r)
	}
}

func TestStartWithoutWorkOrOnEndedRunIsRejected(t *testing.T) {
	b := begin(t, Plugin{})
	if _, err := b.start(context.Background(), "t", "x", nil); err == nil {
		t.Fatal("a job with no work was accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := b.start(ctx, "t", "x", block); err == nil {
		t.Fatal("a job was started on an ended run")
	}
}

// Each run has its own board, so one run's ids mean nothing to another.
func TestBoardsAreIsolatedPerRun(t *testing.T) {
	mine, theirs := begin(t, Plugin{}), begin(t, Plugin{})
	r, _ := mine.start(context.Background(), "t", "secret", func(context.Context) (string, error) { return "sensitive output", nil })
	settled(t, mine, r.ID)
	if _, err := theirs.get(r.ID); !errors.Is(err, errNotFound) {
		t.Fatalf("other run read the job: %v", err)
	}
	if _, err := (jobCancelTool{jobTool{theirs, "job_cancel"}}).Run(context.Background(), `{"id":"`+r.ID+`"}`); !errors.Is(err, errNotFound) {
		t.Fatalf("other run cancelled the job: %v", err)
	}
	if got := theirs.drain(); len(got) != 0 {
		t.Fatalf("other run was notified of %d jobs", len(got))
	}
}

func TestCancelStopsTheWorkAndIsIdempotent(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "t", "long", block)
	cancel := jobCancelTool{jobTool{b, "job_cancel"}}
	if _, err := cancel.Run(context.Background(), `{"id":"`+r.ID+`"}`); err != nil {
		t.Fatal(err)
	}
	if got := settled(t, b, r.ID); got.State != cancelled {
		t.Fatalf("state = %s, want cancelled", got.State)
	}
	if _, err := cancel.Run(context.Background(), `{"id":"`+r.ID+`"}`); err != nil {
		t.Fatalf("second cancel: %v", err)
	}
}

func TestCloseRunCancelsRunningWork(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "t", "long", block)
	b.CloseRun()
	if got, _ := b.get(r.ID); got.State != cancelled {
		t.Fatalf("job outlived its run: %+v", got)
	}
}

func TestDrainIsOneShot(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "t", "quick", func(context.Context) (string, error) { return "done", nil })
	settled(t, b, r.ID)
	if first := b.drain(); len(first) != 1 {
		t.Fatalf("first drain returned %d, want 1", len(first))
	}
	// A notice must be delivered exactly once, or every later turn re-announces
	// the same finished job.
	if second := b.drain(); len(second) != 0 {
		t.Fatalf("second drain returned %d, want 0", len(second))
	}
	if d := b.BeforeStep(context.Background(), agentcore.StepInfo{}); len(d.AdditionalContexts) != 0 {
		t.Fatalf("announced job re-announced: %+v", d)
	}
}

func TestResultIsBoundedBeforeItReachesTheModel(t *testing.T) {
	ext, err := (Plugin{}).BeginRun(context.Background(), agentcore.RunInfo{Limits: agentcore.Limits{MaxToolResultLen: 200}})
	if err != nil {
		t.Fatal(err)
	}
	b := ext.(*board)
	r, _ := b.start(context.Background(), "t", "huge", func(context.Context) (string, error) { return strings.Repeat("x", 100_000), nil })
	if got := settled(t, b, r.ID); len(got.Result) > 200 {
		t.Fatalf("stored result is %d bytes", len(got.Result))
	}
}

// A long run that starts many jobs must not keep every record forever: finished
// and announced ones beyond the cap are evicted oldest first, and an
// unannounced one is kept until a notice carries it.
func TestFinishedRecordsAreBounded(t *testing.T) {
	b := begin(t, Plugin{MaxConcurrent: -1})
	quick := func(context.Context) (string, error) { return "ok", nil }
	var ids []string
	for range keepFinished * 3 {
		r, err := b.start(context.Background(), "t", "quick", quick)
		if err != nil {
			t.Fatal(err)
		}
		settled(t, b, r.ID)
		ids = append(ids, r.ID)
		b.drain()
	}
	unannounced, _ := b.start(context.Background(), "t", "late", quick)
	settled(t, b, unannounced.ID)
	if n := len(b.list()); n > keepFinished+1 {
		t.Fatalf("board holds %d records, want at most %d", n, keepFinished+1)
	}
	if _, err := b.get(ids[0]); !errors.Is(err, errNotFound) {
		t.Fatal("oldest announced record was not evicted")
	}
	if _, err := b.get(ids[len(ids)-1]); err != nil {
		t.Fatal("newest announced record was evicted")
	}
	if _, err := b.get(unannounced.ID); err != nil {
		t.Fatal("an unannounced record was evicted before a notice carried it")
	}
	if got := b.drain(); len(got) != 1 || got[0].ID != unannounced.ID {
		t.Fatalf("drain after eviction = %+v", got)
	}
}

func TestWaitReturnsWhileStillRunningWithoutCancelling(t *testing.T) {
	b := begin(t, Plugin{})
	release := make(chan struct{})
	r, _ := b.start(context.Background(), "t", "slow", func(ctx context.Context) (string, error) {
		select {
		case <-release:
			return "eventually", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	out, err := (jobWaitTool{jobTool{b, "job_wait"}}).Run(context.Background(), `{"id":"`+r.ID+`","timeout_seconds":1}`)
	if err != nil || !strings.Contains(out, "still running") {
		t.Fatalf("job_wait = %q, %v", out, err)
	}
	// Timing out the wait must NOT cancel the job.
	if got, _ := b.get(r.ID); got.State != running {
		t.Fatalf("wait timeout cancelled the job (state=%s)", got.State)
	}
	close(release)
	if got := settled(t, b, r.ID); got.State != succeeded {
		t.Fatalf("state = %s", got.State)
	}
}

func TestToolsRenderStateAndRejectUnknownIDs(t *testing.T) {
	b := begin(t, Plugin{})
	list := jobListTool{jobTool{b, "job_list"}}
	if out, err := list.Run(context.Background(), `{}`); err != nil || !strings.Contains(out, "No background jobs") {
		t.Fatalf("job_list on empty = %q, %v", out, err)
	}
	r, _ := b.start(context.Background(), "run_sql", "count rows", func(context.Context) (string, error) { return "1234", nil })
	settled(t, b, r.ID)
	if out, err := list.Run(context.Background(), `{}`); err != nil || !strings.Contains(out, r.ID) || !strings.Contains(out, "succeeded") {
		t.Fatalf("job_list = %q, %v", out, err)
	}
	status := jobStatusTool{jobTool{b, "job_status"}}
	if out, err := status.Run(context.Background(), `{"id":"`+r.ID+`"}`); err != nil || !strings.Contains(out, "1234") {
		t.Fatalf("job_status = %q, %v", out, err)
	}
	if _, err := status.Run(context.Background(), `{"id":"job_nope"}`); !errors.Is(err, errNotFound) {
		t.Fatalf("unknown id = %v", err)
	}
	if _, err := status.Run(context.Background(), `{}`); err == nil {
		t.Fatal("missing id must be an error")
	}
	if _, err := (jobWaitTool{jobTool{b, "job_wait"}}).Run(context.Background(), `{"id":"job_nope"}`); !errors.Is(err, errNotFound) {
		t.Fatalf("wait unknown = %v", err)
	}
}

func TestCompletionNoticeInlinesSmallResultsAndDefersLargeOnes(t *testing.T) {
	now := time.Now()
	notice := completionNotice([]record{
		{ID: "job_1", Tool: "run_sql", State: succeeded, Result: "7 rows", StartedAt: now.Add(-time.Second), EndedAt: now},
		{ID: "job_2", Tool: "export", State: succeeded, Result: strings.Repeat("y", 5000), StartedAt: now.Add(-time.Second), EndedAt: now},
		{ID: "job_3", Tool: "crawl", State: failed, Err: "host unreachable", StartedAt: now.Add(-time.Second), EndedAt: now},
		{ID: "job_4", Tool: "crawl", State: cancelled, Err: interrupted, StartedAt: now.Add(-time.Second), EndedAt: now},
	}, 2048)
	for _, want := range []string{"7 rows", "job_status", "host unreachable", interrupted} {
		if !strings.Contains(notice, want) {
			t.Fatalf("notice missing %q:\n%s", want, notice)
		}
	}
	if strings.Contains(notice, strings.Repeat("y", 100)) {
		t.Fatal("a large result must not ride inline")
	}
	if completionNotice(nil, 2048) != "" {
		t.Fatal("no finished jobs means no notice")
	}
}

func TestFinishWithPendingJobReturnsControlAfterBoundedWait(t *testing.T) {
	b := begin(t, Plugin{})
	defer b.CloseRun()
	if _, err := b.start(context.Background(), "t", "pending", block); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	// The parent ctx ending means the run is aborting, not that the model may
	// claim the work done.
	if d := b.TurnStopping(ctx, agentcore.StopInfo{}); d.Continue || d.StopReason != "aborted" {
		t.Fatalf("aborted finish = %+v", d)
	}
	// A live run gets control back with the work still pending.
	b.stopWait = time.Millisecond
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); !d.Continue || d.StopReason != "" || len(d.Inject) != 1 || !strings.Contains(d.Inject[0].Content, "still pending") {
		t.Fatalf("pending work accepted as final: %+v", d)
	}
}

func TestFinishAnnouncesWorkThatSettled(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "t", "quick", func(context.Context) (string, error) { return "found it", nil })
	settled(t, b, r.ID)
	d := b.TurnStopping(context.Background(), agentcore.StopInfo{})
	if !d.Continue || len(d.Inject) != 1 || !strings.Contains(d.Inject[0].Content, "found it") {
		t.Fatalf("settled work not announced before the finish: %+v", d)
	}
	if again := b.TurnStopping(context.Background(), agentcore.StopInfo{}); again.Continue {
		t.Fatalf("announced work re-opened the finish: %+v", again)
	}
}

// The board survives resume through the native plugin checkpoint: finished
// results stay readable, an unannounced one is announced on the first resumed
// step, a job the end of the run interrupted is reported as interrupted rather
// than vanishing, and new ids never collide with ids the transcript names.
func TestBoardSurvivesResume(t *testing.T) {
	first := begin(t, Plugin{})
	announced, _ := first.start(context.Background(), "run_sql", "count", func(context.Context) (string, error) { return "1234", nil })
	settled(t, first, announced.ID)
	first.drain()
	unannounced, _ := first.start(context.Background(), "export", "export", func(context.Context) (string, error) { return "exported", nil })
	settled(t, first, unannounced.ID)
	interruptedJob, _ := first.start(context.Background(), "crawl", "crawl", block)
	raw, err := first.NativeState()
	if err != nil {
		t.Fatal(err)
	}
	first.CloseRun()

	second := begin(t, Plugin{})
	if err := second.RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
	if out, err := (jobStatusTool{jobTool{second, "job_status"}}).Run(context.Background(), `{"id":"`+announced.ID+`"}`); err != nil || !strings.Contains(out, "1234") {
		t.Fatalf("finished result lost across resume: %q %v", out, err)
	}
	notice := second.BeforeStep(context.Background(), agentcore.StepInfo{}).AdditionalContexts
	if len(notice) != 1 {
		t.Fatalf("resumed run announced %d messages, want 1", len(notice))
	}
	text := notice[0].Content
	if !strings.Contains(text, unannounced.ID) || !strings.Contains(text, "exported") || !strings.Contains(text, interruptedJob.ID) || !strings.Contains(text, "interrupted") || strings.Contains(text, announced.ID) {
		t.Fatalf("resumed notice = %q", text)
	}
	// Restored records have no goroutine: waiting returns at once.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if got, err := second.wait(ctx, interruptedJob.ID); err != nil || got.State != cancelled {
		t.Fatalf("restored interrupted job = %+v %v", got, err)
	}
	next, err := second.start(context.Background(), "t", "new", func(context.Context) (string, error) { return "", nil })
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{announced.ID, unannounced.ID, interruptedJob.ID} {
		if next.ID == id {
			t.Fatalf("new job reused checkpointed id %s", id)
		}
	}
	settled(t, second, next.ID)
}

func TestRestoreRejectsInvalidCheckpoints(t *testing.T) {
	for name, raw := range map[string]string{
		"not json":           `{`,
		"negative seq":       `{"seq":-1}`,
		"more jobs than seq": `{"seq":0,"jobs":[{"id":"job_1","state":"succeeded"}]}`,
		"running job":        `{"seq":1,"jobs":[{"id":"job_1","state":"running"}]}`,
		"unknown state":      `{"seq":1,"jobs":[{"id":"job_1","state":"paused"}]}`,
		"missing id":         `{"seq":1,"jobs":[{"state":"failed"}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := begin(t, Plugin{}).RestoreNativeState(json.RawMessage(raw)); err == nil {
				t.Fatal("invalid checkpoint accepted")
			}
		})
	}
	// A checkpoint is bounded on the way in too: an oversized result is clipped.
	b := begin(t, Plugin{})
	if err := b.RestoreNativeState(json.RawMessage(fmt.Sprintf(`{"seq":1,"jobs":[{"id":"job_1","state":"succeeded","result":%q}]}`, strings.Repeat("z", 1_000_000)))); err != nil {
		t.Fatal(err)
	}
	if r, _ := b.get("job_1"); len(r.Result) > agentcore.DefaultLimits().MaxToolResultLen {
		t.Fatalf("restored result is %d bytes", len(r.Result))
	}
}

// The receipt a tool gets back from agentcore.LaunchBackground names the job.
func TestLaunchBackgroundReceipt(t *testing.T) {
	b := begin(t, Plugin{})
	ctx := b.RunContext(context.Background())
	raw, err := agentcore.LaunchBackground(ctx, "shell", "make", func(context.Context) (string, error) { return "built", nil })
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct{ ID, Tool, Label, State string }
	if err := json.Unmarshal(raw, &receipt); err != nil || receipt.ID == "" || receipt.Tool != "shell" || receipt.State != string(running) {
		t.Fatalf("receipt = %s %v", raw, err)
	}
	settled(t, b, receipt.ID)
	if _, err := agentcore.LaunchBackground(context.Background(), "shell", "make", block); err == nil {
		t.Fatal("a run without the plugin launched background work")
	}
}
