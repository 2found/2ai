package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/2found/2ai/core/agentcore"
)

// publisher starts a job that publishes each chunk, then parks until released.
func publisher(t *testing.T, b *board, chunks ...string) (id string, release func()) {
	t.Helper()
	published := make(chan struct{})
	done := make(chan struct{})
	r, err := b.start(context.Background(), "run_shell", "build", func(ctx context.Context) (string, error) {
		for _, c := range chunks {
			agentcore.PublishJobOutput(ctx, c)
		}
		close(published)
		select {
		case <-done:
			return "built", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	<-published
	var once sync.Once
	return r.ID, func() { once.Do(func() { close(done) }) }
}

func status(t *testing.T, b *board, args string) string {
	t.Helper()
	out, err := (jobStatusTool{jobTool{b, "job_status"}}).Run(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// A job that prints far more than the buffer holds keeps its head and its tail
// and costs bounded memory, however long it runs.
func TestLiveOutputIsBounded(t *testing.T) {
	b := begin(t, Plugin{})
	var chunks []string
	chunks = append(chunks, "HEAD-MARKER\n")
	for i := range 20_000 {
		chunks = append(chunks, fmt.Sprintf("line %06d padding padding padding\n", i))
	}
	chunks = append(chunks, "TAIL-MARKER\n")
	id, release := publisher(t, b, chunks...)
	defer release()
	j, _ := b.find(id)
	if held := len(j.out.head) + len(j.out.tail); held > liveHead+2*liveTail {
		t.Fatalf("job retains %d bytes of output, bound is %d", held, liveHead+2*liveTail)
	}
	if j.out.size() < 600_000 {
		t.Fatalf("total counted %d bytes", j.out.size())
	}
	full := j.out.text(0)
	for _, want := range []string{"HEAD-MARKER", "TAIL-MARKER", "bytes omitted"} {
		if !strings.Contains(full, want) {
			t.Fatalf("retained output lacks %q", want)
		}
	}
	if len(full) > liveHead+liveTail+64 {
		t.Fatalf("rendered output is %d bytes", len(full))
	}
	if !strings.Contains(full, "line 019999") || strings.Contains(full, "line 010000") {
		t.Fatal("tail is not the newest output, or the middle was kept")
	}
}

func TestLiveOutputNeverSplitsARune(t *testing.T) {
	b := begin(t, Plugin{})
	// Three-byte runes written one byte at a time cross every chunk boundary.
	var chunks []string
	for range 6_000 {
		for _, c := range []byte("日本語") {
			chunks = append(chunks, string([]byte{c}))
		}
	}
	id, release := publisher(t, b, chunks...)
	defer release()
	j, _ := b.find(id)
	for _, n := range []int{0, 1, 2, 3, 100, 2048} {
		if got := j.out.text(n); !utf8.ValidString(got) {
			t.Fatalf("text(%d) is not valid UTF-8", n)
		}
	}
	if line := j.out.lastLine(); !utf8.ValidString(line) {
		t.Fatal("lastLine is not valid UTF-8")
	}
}

func TestJobStatusOfRunningJobIsATailByDefault(t *testing.T) {
	b := begin(t, Plugin{})
	var chunks []string
	for i := range 2_000 {
		chunks = append(chunks, fmt.Sprintf("step %04d\n", i))
	}
	id, release := publisher(t, b, chunks...)
	defer release()

	short := status(t, b, `{"id":"`+id+`"}`)
	if !strings.Contains(short, "running") || !strings.Contains(short, "step 1999") {
		t.Fatalf("status lacks the state or the newest output:\n%s", short)
	}
	if strings.Contains(short, "step 1500") {
		t.Fatal("default status returned more than a tail")
	}
	if len(short) > defaultStatusTail+300 {
		t.Fatalf("default status is %d bytes", len(short))
	}
	more := status(t, b, `{"id":"`+id+`","output_bytes":6000}`)
	if !strings.Contains(more, "step 1500") || len(more) <= len(short) {
		t.Fatal("output_bytes did not widen the tail")
	}
	// The ask is clipped to what a tool result may carry.
	ext, _ := (Plugin{}).BeginRun(context.Background(), agentcore.RunInfo{Limits: agentcore.Limits{MaxToolResultLen: 500}})
	small := ext.(*board)
	sid, srelease := publisher(t, small, chunks...)
	defer srelease()
	if got := status(t, small, `{"id":"`+sid+`","output_bytes":100000}`); len(got) > 500+300 {
		t.Fatalf("status exceeded the result limit: %d bytes", len(got))
	}
	if _, err := (jobStatusTool{jobTool{b, "job_status"}}).Run(context.Background(), `{"id":"`+id+`","output_bytes":-1}`); err == nil {
		t.Fatal("negative output_bytes accepted")
	}
}

// A running job that published nothing renders exactly as before, so a host
// that never publishes sees no change.
func TestRunningJobWithoutOutputRendersAsBefore(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "t", "quiet", block)
	defer b.CloseRun()
	got := status(t, b, `{"id":"`+r.ID+`"}`)
	if strings.Count(got, "\n") != 0 || !strings.HasPrefix(got, "job "+r.ID+" (t) — running after") {
		t.Fatalf("status = %q", got)
	}
	list, _ := (jobListTool{jobTool{b, "job_list"}}).Run(context.Background(), "")
	if strings.Contains(list, ">") || strings.Count(list, "\n") != 0 {
		t.Fatalf("list = %q", list)
	}
}

func TestJobListShowsElapsedAndTheLastLine(t *testing.T) {
	b := begin(t, Plugin{})
	id, release := publisher(t, b, "compiling a\n", "compiling b\n\n", "  linking "+strings.Repeat("x", 300)+"\n")
	defer release()
	out, err := (jobListTool{jobTool{b, "job_list"}}).Run(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(out, "\n")
	if len(lines) != 2 || !strings.HasPrefix(lines[0], id+"\trunning\trun_shell\t") || !strings.HasSuffix(lines[0], "\tbuild") {
		t.Fatalf("list = %q", out)
	}
	if !strings.HasPrefix(lines[1], "\t> linking xxx") || len(lines[1]) > lastLineBytes+16 {
		t.Fatalf("last line = %q", lines[1])
	}
	release()
	settled(t, b, id)
	if out, _ := (jobListTool{jobTool{b, "job_list"}}).Run(context.Background(), ""); strings.Contains(out, ">") {
		t.Fatalf("a finished job still lists its output: %q", out)
	}
}

func TestWaitTimeoutCarriesTheNewestOutput(t *testing.T) {
	b := begin(t, Plugin{})
	id, release := publisher(t, b, "server listening on :8080\n")
	defer release()
	out, err := (jobWaitTool{jobTool{b, "job_wait"}}).Run(context.Background(), `{"id":"`+id+`","timeout_seconds":1}`)
	if err != nil || !strings.Contains(out, "still running") || !strings.Contains(out, "listening on :8080") {
		t.Fatalf("wait = %q %v", out, err)
	}
}

// A job that fails or is cancelled has no result, so its last output is what
// explains it; the checkpoint carries it too.
func TestFailedAndInterruptedJobsKeepTheirLastOutput(t *testing.T) {
	b := begin(t, Plugin{})
	failing, _ := b.start(context.Background(), "t", "fail", func(ctx context.Context) (string, error) {
		agentcore.PublishJobOutput(ctx, "error: undefined symbol foo\n")
		return "", fmt.Errorf("exit status 2")
	})
	settled(t, b, failing.ID)
	if got := status(t, b, `{"id":"`+failing.ID+`"}`); !strings.Contains(got, "exit status 2") || !strings.Contains(got, "undefined symbol foo") {
		t.Fatalf("failed status = %q", got)
	}
	running, _ := b.start(context.Background(), "t", "long", func(ctx context.Context) (string, error) {
		agentcore.PublishJobOutput(ctx, "downloading 40%\n")
		<-ctx.Done()
		return "", ctx.Err()
	})
	for b.lastLine(running.ID) == "" {
		time.Sleep(time.Millisecond)
	}
	raw, err := b.NativeState()
	if err != nil {
		t.Fatal(err)
	}
	b.CloseRun()
	next := begin(t, Plugin{})
	if err := next.RestoreNativeState(raw); err != nil {
		t.Fatal(err)
	}
	if got := status(t, next, `{"id":"`+running.ID+`"}`); !strings.Contains(got, "interrupted") || !strings.Contains(got, "downloading 40%") {
		t.Fatalf("interrupted status = %q", got)
	}
	var saved checkpoint
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Jobs) != 2 {
		t.Fatalf("checkpoint = %s", raw)
	}
}

// Output publishing is called from many goroutines (a shell's two pipes) while
// the model reads the board.
func TestPublishingIsRaceFree(t *testing.T) {
	b := begin(t, Plugin{})
	var wg sync.WaitGroup
	ready := make(chan struct{})
	release := make(chan struct{})
	r, _ := b.start(context.Background(), "t", "noisy", func(ctx context.Context) (string, error) {
		var inner sync.WaitGroup
		for w := range 4 {
			inner.Add(1)
			go func() {
				defer inner.Done()
				for i := range 500 {
					agentcore.PublishJobOutput(ctx, fmt.Sprintf("w%d line %d\n", w, i))
				}
			}()
		}
		inner.Wait()
		close(ready)
		<-release
		return "", nil
	})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-ready:
				return
			default:
				status(t, b, `{"id":"`+r.ID+`"}`)
				b.lastLine(r.ID)
			}
		}
	}()
	wg.Wait()
	close(release)
	settled(t, b, r.ID)
}

// A wrapper that runs a tool under the values of the call that started the job
// keeps the job's output handle through agentcore.BackgroundValues.
func TestBackgroundValuesKeepsTheJobHandle(t *testing.T) {
	b := begin(t, Plugin{})
	type key struct{}
	call := context.WithValue(context.Background(), key{}, "call value")
	r, _ := b.start(call, "t", "wrapped", func(jobCtx context.Context) (string, error) {
		ctx := agentcore.BackgroundValues(jobCtx, context.WithoutCancel(call))
		if ctx.Value(key{}) != "call value" {
			return "", fmt.Errorf("lost the call's values")
		}
		agentcore.PublishJobOutput(ctx, "through the wrapper\n")
		<-ctx.Done()
		return "", ctx.Err()
	})
	for b.lastLine(r.ID) == "" {
		time.Sleep(time.Millisecond)
	}
	b.CloseRun()
	if got := settled(t, b, r.ID); !strings.Contains(got.Output, "through the wrapper") {
		t.Fatalf("job = %+v", got)
	}
}
