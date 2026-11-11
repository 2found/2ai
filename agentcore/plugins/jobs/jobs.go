// Package jobs lets a tool go asynchronous.
//
// Without it every tool is synchronous: a long crawl, build, render, or export
// blocks the whole run until it finishes, and a run that has to wait cannot
// also think. With it a tool calls agentcore.LaunchBackground, returns a job
// receipt immediately, and the model keeps working; the built-in job_list /
// job_status / job_wait / job_cancel tools observe and stop the work, and jobs
// that finished since the last turn are announced at the top of the next one.
//
// Every run gets its own board, so a job can be neither observed nor cancelled
// by another run, and every job is cancelled when its run ends. The board's
// records are checkpointed: a resumed run can still read the results of jobs
// that finished, and learns which ones the end of the previous run interrupted.
package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

type state string

const (
	running   state = "running"
	succeeded state = "succeeded"
	failed    state = "failed"
	cancelled state = "cancelled"
)

// record is one unit of background work as the model, the receipt and the
// checkpoint see it.
type record struct {
	ID    string `json:"id"`
	Tool  string `json:"tool"`
	Label string `json:"label"`
	State state  `json:"state"`
	// Result is the work's output once it succeeds, bounded by the board.
	Result string `json:"result,omitempty"`
	// Err is the failure message once it fails or is cancelled.
	Err       string    `json:"error,omitempty"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitzero"`
	// Announced is set once a completion notice carried the job.
	Announced bool `json:"announced,omitempty"`
}

func (r record) done() bool { return r.State != running }

// duration is how long the job ran (so far, if still running).
func (r record) duration() time.Duration {
	if r.EndedAt.IsZero() {
		return time.Since(r.StartedAt)
	}
	return r.EndedAt.Sub(r.StartedAt)
}

// errNotFound is returned for an id this run's board does not hold.
var errNotFound = errors.New("job not found")

// Plugin installs background jobs.
type Plugin struct {
	// MaxConcurrent bounds running jobs per run. Zero uses 15; -1 disables the ceiling.
	MaxConcurrent int
}

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "jobs" }

// Register adds the plugin as a run extension.
func (p Plugin) Register(r *agentcore.Registry) error {
	r.AddExtension(p)
	return nil
}

const (
	defaultMaxConcurrent = 15
	// maxWait bounds one job_wait call so the model cannot park the run forever
	// on a job that never finishes.
	maxWait = 120 * time.Second
	// defaultResultBytes bounds a stored result when the run sets no tool
	// result limit, so a completed job's output cannot blow the context.
	defaultResultBytes = 16 * 1024
	// keepFinished bounds the finished records a board holds. Announced ones
	// beyond it are evicted oldest first, so a long run that starts many jobs
	// neither grows its memory nor its checkpoint without bound.
	keepFinished = 16
	// stopWait bounds how long a finish waits for pending work to settle.
	stopWait = 5 * time.Second
	// noticeInlineLimit bounds how much of a finished job's output rides into
	// the completion notice before the model is told to fetch it explicitly.
	noticeInlineLimit = 2 * 1024
)

// interrupted is the error a resumed run reads for a job the end of the
// previous run cancelled.
const interrupted = "interrupted: the run ended before this job finished; start it again if it is still needed"

// BeginRun gives the run its own board.
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if p.MaxConcurrent < -1 {
		return nil, fmt.Errorf("jobs: MaxConcurrent must be nonnegative or -1")
	}
	limit := p.MaxConcurrent
	if limit == 0 {
		limit = defaultMaxConcurrent
	}
	maxBytes := info.Limits.MaxToolResultLen
	if maxBytes <= 0 {
		maxBytes = defaultResultBytes
	}
	return &board{maxConcurrent: limit, maxBytes: maxBytes, stopWait: stopWait}, nil
}

// board is one run's background work and the agentcore.Extension the loop
// talks to.
type board struct {
	maxConcurrent int
	maxBytes      int
	stopWait      time.Duration

	mu   sync.Mutex
	seq  int
	jobs []*job // oldest first
}

type job struct {
	record
	cancel   context.CancelFunc
	finished chan struct{} // closed once the work has returned
}

// Name identifies the extension in composition diagnostics.
func (*board) Name() string { return "jobs" }

// Tools contributes the four observation tools.
func (b *board) Tools() []agentcore.Tool {
	return []agentcore.Tool{
		jobListTool{jobTool{b, "job_list"}},
		jobStatusTool{jobTool{b, "job_status"}},
		jobWaitTool{jobTool{b, "job_wait"}},
		jobCancelTool{jobTool{b, "job_cancel"}},
	}
}

// RunContext binds the launcher to the run's context, which is how a tool finds
// it. The context is the run's, not the calling tool's, so work survives the
// call that started it and still dies with the run.
func (b *board) RunContext(ctx context.Context) context.Context {
	return agentcore.WithBackgroundLauncher(ctx, func(tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error) {
		r, err := b.start(ctx, tool, label, run)
		if err != nil {
			return nil, err
		}
		return json.Marshal(r)
	})
}

// start launches the work in a goroutine and records it as running.
func (b *board) start(ctx context.Context, tool, label string, run func(context.Context) (string, error)) (record, error) {
	if run == nil {
		return record{}, errors.New("background job has no work to run")
	}
	if err := ctx.Err(); err != nil {
		return record{}, err
	}
	b.mu.Lock()
	active := 0
	for _, j := range b.jobs {
		if !j.done() {
			active++
		}
	}
	if b.maxConcurrent > 0 && active >= b.maxConcurrent {
		b.mu.Unlock()
		return record{}, fmt.Errorf("concurrent job capacity reached (%d)", b.maxConcurrent)
	}
	b.seq++
	jobCtx, cancel := context.WithCancel(ctx)
	j := &job{record: record{ID: fmt.Sprintf("job_%d", b.seq), Tool: tool, Label: label, State: running, StartedAt: time.Now()}, cancel: cancel, finished: make(chan struct{})}
	b.jobs = append(b.jobs, j)
	snapshot := j.record
	b.mu.Unlock()

	go func() {
		defer close(j.finished)
		defer cancel()
		var out string
		var err error
		// A panicking job must not take the process down: it becomes a failed
		// job, exactly like one that returned an error.
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("job panicked: %v", r)
				}
			}()
			out, err = run(jobCtx)
		}()
		b.mu.Lock()
		defer b.mu.Unlock()
		j.EndedAt = time.Now()
		switch {
		case jobCtx.Err() != nil && err != nil:
			j.State = cancelled
			j.Err = jobCtx.Err().Error()
		case err != nil:
			j.State = failed
			j.Err = err.Error()
		default:
			j.State = succeeded
			j.Result = agentcore.TruncateMiddle(out, b.maxBytes)
		}
		b.evictLocked()
	}()
	return snapshot, nil
}

// evictLocked drops the oldest announced finished records beyond keepFinished.
// Unannounced ones stay until a notice has carried them.
func (b *board) evictLocked() {
	finished := 0
	for _, j := range b.jobs {
		if j.done() {
			finished++
		}
	}
	kept := b.jobs[:0]
	for _, j := range b.jobs {
		if finished > keepFinished && j.done() && j.Announced {
			finished--
			continue
		}
		kept = append(kept, j)
	}
	clear(b.jobs[len(kept):])
	b.jobs = kept
}

func (b *board) find(id string) (*job, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, j := range b.jobs {
		if j.ID == id {
			return j, nil
		}
	}
	return nil, errNotFound
}

func (b *board) get(id string) (record, error) {
	j, err := b.find(id)
	if err != nil {
		return record{}, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return j.record, nil
}

// list returns the board's records, newest first.
func (b *board) list() []record {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]record, 0, len(b.jobs))
	for i := len(b.jobs) - 1; i >= 0; i-- {
		out = append(out, b.jobs[i].record)
	}
	return out
}

// wait blocks until the job is terminal or ctx ends.
func (b *board) wait(ctx context.Context, id string) (record, error) {
	j, err := b.find(id)
	if err != nil {
		return record{}, err
	}
	select {
	case <-j.finished:
	case <-ctx.Done():
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return j.record, nil
}

// drain returns the finished jobs no notice has carried yet, and marks them
// announced.
func (b *board) drain() []record {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []record
	for _, j := range b.jobs {
		if j.done() && !j.Announced {
			j.Announced = true
			out = append(out, j.record)
		}
	}
	b.evictLocked()
	return out
}

// CloseRun cancels everything this run started. A job outliving its run would
// be unobservable and uncancellable — nothing left to look at it.
func (b *board) CloseRun() {
	b.mu.Lock()
	pending := make([]*job, 0, len(b.jobs))
	for _, j := range b.jobs {
		if !j.done() {
			pending = append(pending, j)
		}
	}
	b.mu.Unlock()
	for _, j := range pending {
		j.cancel()
	}
	// Cooperative workers (including native model streams) settle before usage
	// is folded. An arbitrary host job cannot hold shutdown forever.
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for _, j := range pending {
		select {
		case <-j.finished:
		case <-timeout.C:
			return
		}
	}
}

// TurnStopping makes a normal final answer account for the work the run
// launched. It waits briefly instead of letting a confident finish silently
// cancel pending work.
func (b *board) TurnStopping(ctx context.Context, _ agentcore.StopInfo) agentcore.StopDecision {
	waitCtx, cancel := context.WithTimeout(ctx, b.stopWait)
	defer cancel()
	for _, r := range b.list() {
		if r.done() {
			continue
		}
		if settled, err := b.wait(waitCtx, r.ID); err != nil || !settled.done() {
			if ctx.Err() != nil {
				return agentcore.StopDecision{StopReason: "aborted"}
			}
			return agentcore.StopDecision{Continue: true, Inject: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Background work is still pending. Use job_wait/status to collect it, or job_cancel if it is no longer needed; do not claim it completed."}}}
		}
	}
	completed := b.drain()
	if len(completed) == 0 {
		return agentcore.StopDecision{}
	}
	return agentcore.StopDecision{Continue: true, Inject: []agentcore.Message{{Role: agentcore.RoleUser, Content: completionNotice(completed, b.maxBytes)}}}
}

// BeforeStep announces jobs that finished since the last turn.
//
// The notice is a synthetic user message the LOOP appends and persists, so the
// model learns its background work is done without spending a turn on a poll —
// and a resumed run replays the same conversation.
func (b *board) BeforeStep(context.Context, agentcore.StepInfo) agentcore.StepDecision {
	text := completionNotice(b.drain(), noticeInlineLimit)
	if text == "" {
		return agentcore.StepDecision{}
	}
	return agentcore.StepDecision{AdditionalContexts: []agentcore.Message{{Role: agentcore.RoleUser, Content: text}}}
}

type checkpoint struct {
	Seq  int      `json:"seq"`
	Jobs []record `json:"jobs,omitempty"`
}

// NativeState checkpoints the board. It is taken before CloseRun cancels the
// work, so a job still running here is recorded as interrupted and not yet
// announced: the resumed run's first step tells the model, instead of the job
// silently vanishing. No goroutine survives a run; only its record does.
func (b *board) NativeState() (json.RawMessage, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := checkpoint{Seq: b.seq, Jobs: make([]record, 0, len(b.jobs))}
	for _, j := range b.jobs {
		r := j.record
		if !r.done() {
			r.State, r.Err, r.EndedAt, r.Announced = cancelled, interrupted, time.Now(), false
		}
		out.Jobs = append(out.Jobs, r)
	}
	return json.Marshal(out)
}

// RestoreNativeState loads a checkpointed board. Ids keep counting from the
// checkpoint so a new job never reuses the id of one the transcript names.
func (b *board) RestoreNativeState(raw json.RawMessage) error {
	var saved checkpoint
	if err := json.Unmarshal(raw, &saved); err != nil {
		return err
	}
	if saved.Seq < 0 || len(saved.Jobs) > saved.Seq {
		return errors.New("jobs: invalid checkpoint")
	}
	restored := make([]*job, 0, len(saved.Jobs))
	for _, r := range saved.Jobs {
		switch r.State {
		case succeeded, failed, cancelled:
		default:
			return fmt.Errorf("jobs: invalid checkpointed state %q", r.State)
		}
		if r.ID == "" {
			return errors.New("jobs: checkpointed job has no id")
		}
		r.Result = agentcore.TruncateMiddle(r.Result, b.maxBytes)
		finished := make(chan struct{})
		close(finished)
		restored = append(restored, &job{record: r, cancel: func() {}, finished: finished})
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.seq, b.jobs = saved.Seq, restored
	b.evictLocked()
	return nil
}

// completionNotice renders the synthetic user message announcing finished jobs.
// It carries the result inline for a small success so the common case costs the
// model no extra tool call, and points at job_status when the payload is large.
func completionNotice(jobs []record, inlineLimit int) string {
	if len(jobs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Background job update:\n")
	for _, j := range jobs {
		fmt.Fprintf(&b, "- %s (%s, %s, %s)", j.ID, j.Tool, j.State, j.duration().Round(time.Millisecond))
		switch {
		case j.State == failed || j.State == cancelled:
			fmt.Fprintf(&b, ": %s", j.Err)
		case j.State == succeeded && len(j.Result) <= inlineLimit:
			fmt.Fprintf(&b, ":\n%s", j.Result)
		case j.State == succeeded:
			fmt.Fprintf(&b, ": %d bytes of output — call job_status with id %q to read it.", len(j.Result), j.ID)
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// ---- model-facing tools -------------------------------------------------

// jobTool is the shared base for the built-in job_* tools.
type jobTool struct {
	board *board
	name  string
}

func (t jobTool) Name() string { return t.name }

// SelfGated exempts the job_* tools from the permission gate. They observe and
// cancel work THIS run launched — the board is per run — so they grant no
// capability the agent did not already exercise by starting the job.
func (jobTool) SelfGated() bool { return true }

// Parallel: every job tool is a bounded read or a cancel request, safe to run
// alongside other parallel-eligible calls in the same batch.
func (t jobTool) Parallel() bool { return true }

// Bookkeeping: observing background work is administration, not progress toward
// the task. Two things follow, and both matter for a run that goes async:
//
//   - A turn spent only on job_* calls is refunded against MaxTurns, so polling
//     a slow job cannot starve the turn budget it is waiting to spend.
//   - A repeat guard asks the run whether a tool is bookkeeping, so the
//     legitimate "poll the same job id until it finishes" chain is transparent
//     to it instead of being nudged as a stuck loop.
//
// The MaxToolCalls budget still backstops a runaway polling loop.
func (t jobTool) Bookkeeping() bool { return true }

func jobIDSchema(name, description string, extra map[string]any) agentcore.ToolSchema {
	props := map[string]any{
		"id": map[string]any{"type": "string", "description": "The job id."},
	}
	for k, v := range extra {
		props[k] = v
	}
	return agentcore.ToolSchema{
		Name:        name,
		Description: description,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   []string{"id"},
		},
	}
}

type jobListTool struct{ jobTool }

func (t jobListTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   "job_list",
		Strict: agentcore.ToolStrictEnabled,
		Description: "List the background jobs this session started, newest first, with their state and duration. " +
			"Jobs still running when you finish your answer are cancelled — wait for the ones whose results you need.",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false},
	}
}

func (t jobListTool) Run(_ context.Context, _ string) (string, error) {
	jobs := t.board.list()
	if len(jobs) == 0 {
		return "No background jobs.", nil
	}
	var b strings.Builder
	for _, j := range jobs {
		fmt.Fprintf(&b, "%s\t%s\t%s\t%s\t%s\n", j.ID, j.State, j.Tool, j.duration().Round(time.Millisecond), j.Label)
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

type jobStatusTool struct{ jobTool }

func (t jobStatusTool) Schema() agentcore.ToolSchema {
	return jobIDSchema("job_status",
		"Read one background job: its state, and its full output once it has finished.", nil)
}

func (t jobStatusTool) Run(_ context.Context, args string) (string, error) {
	id, err := jobIDArg(args)
	if err != nil {
		return "", err
	}
	j, err := t.board.get(id)
	if err != nil {
		return "", err
	}
	return renderJob(j), nil
}

type jobWaitTool struct{ jobTool }

func (t jobWaitTool) Schema() agentcore.ToolSchema {
	return jobIDSchema("job_wait",
		"Block until a background job finishes, then return its result. Bounded by a timeout — "+
			"if it returns while the job is still running, the job keeps going and you can wait again or continue with other work.",
		map[string]any{
			"timeout_seconds": map[string]any{
				"type":        "integer",
				"description": "How long to wait before returning with the job still running.",
			},
		})
}

func (t jobWaitTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		ID      string `json:"id"`
		Timeout int    `json:"timeout_seconds"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.ID) == "" {
		return "", errors.New("id is required")
	}
	wait := maxWait
	if in.Timeout > 0 && time.Duration(in.Timeout)*time.Second < wait {
		wait = time.Duration(in.Timeout) * time.Second
	}
	waitCtx, cancel := context.WithTimeout(ctx, wait)
	defer cancel()
	j, err := t.board.wait(waitCtx, in.ID)
	if err != nil {
		return "", err
	}
	if !j.done() {
		return fmt.Sprintf("Job %s is still running after %s. It was not cancelled — continue with other work and check job_status later.", j.ID, wait), nil
	}
	return renderJob(j), nil
}

type jobCancelTool struct{ jobTool }

func (t jobCancelTool) Schema() agentcore.ToolSchema {
	return jobIDSchema("job_cancel", "Cancel a running background job.", nil)
}

func (t jobCancelTool) Run(_ context.Context, args string) (string, error) {
	id, err := jobIDArg(args)
	if err != nil {
		return "", err
	}
	j, err := t.board.find(id)
	if err != nil {
		return "", err
	}
	j.cancel()
	return fmt.Sprintf("Cancellation requested for %s.", id), nil
}

// jobIDArg pulls the required id out of a job tool's arguments.
func jobIDArg(args string) (string, error) {
	var in struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	id := strings.TrimSpace(in.ID)
	if id == "" {
		return "", errors.New("id is required")
	}
	return id, nil
}

// renderJob formats a job for the model.
func renderJob(j record) string {
	head := fmt.Sprintf("job %s (%s) — %s after %s", j.ID, j.Tool, j.State, j.duration().Round(time.Millisecond))
	switch j.State {
	case succeeded:
		if j.Result == "" {
			return head + "\n(no output)"
		}
		return head + "\n" + j.Result
	case failed, cancelled:
		return head + "\n" + j.Err
	default:
		return head
	}
}
