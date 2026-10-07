// Package jobs lets a tool go asynchronous.
//
// Without it every tool is synchronous: a long crawl, build, render, or export
// blocks the whole run until it finishes, and a run that has to wait cannot
// also think. With it a tool calls jobs.From(ctx).Start(...), returns a job id
// immediately, and the model keeps working; the built-in job_list / job_status
// / job_wait / job_cancel tools observe and stop the work, and jobs that
// finished since the last turn are announced at the top of the next one.
//
// Every job is fenced to the run that started it and cancelled when that run
// ends, so a job can neither outlive its owner nor be observed by another run.
package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

type JobState string

const (
	JobRunning   JobState = "running"
	JobSucceeded JobState = "succeeded"
	JobFailed    JobState = "failed"
	JobCancelled JobState = "cancelled"
)

// Done reports whether the state is terminal.
func (s JobState) Done() bool { return s != JobRunning }

// Job is one unit of background work.
type Job struct {
	ID    string   `json:"id"`
	Tool  string   `json:"tool"`
	Label string   `json:"label"`
	State JobState `json:"state"`
	// Result is the work's output once it succeeds, bounded by the store.
	Result string `json:"result,omitempty"`
	// Err is the failure message once it fails.
	Err       string    `json:"error,omitempty"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`
}

// Duration is how long the job ran (so far, if still running).
func (j Job) Duration() time.Duration {
	if j.EndedAt.IsZero() {
		return time.Since(j.StartedAt)
	}
	return j.EndedAt.Sub(j.StartedAt)
}

// ErrJobNotFound is returned for an unknown job, and for a job owned by another
// session — the two are deliberately indistinguishable so a job id is not an
// existence oracle across sessions.
var ErrJobNotFound = errors.New("agentcore: job not found")

// JobStore owns background work for a run. Implementations must fence every
// operation by owner: an id from another owner behaves exactly like an unknown
// id. LocalJobStore is the in-process default.
type JobStore interface {
	// Start launches fn and returns immediately with a Running job.
	Start(ctx context.Context, owner string, spec JobSpec) (Job, error)
	// Get returns one job, or ErrJobNotFound.
	Get(owner, id string) (Job, error)
	// List returns the owner's jobs, newest first.
	List(owner string) []Job
	// Cancel requests cancellation. Cancelling a finished job is a no-op.
	Cancel(owner, id string) error
	// Wait blocks until the job is terminal or the context ends, returning the
	// job as of that moment.
	Wait(ctx context.Context, owner, id string) (Job, error)
	// DrainCompleted returns jobs that reached a terminal state since the last
	// drain for this owner, clearing the pending-notice set.
	DrainCompleted(owner string) []Job
	// CancelAll stops every running job for an owner. The loop calls it when a
	// run ends so background work cannot outlive the run that started it.
	CancelAll(owner string)
}

// JobSpec describes work to launch.
type JobSpec struct {
	// Tool is the tool that started the job (descriptive; used in notices).
	Tool string
	// Label is a short human description ("export 2.3M rows").
	Label string
	// Run performs the work. Its context is cancelled when the job is cancelled
	// or the run ends. The returned string is the job's result.
	Run func(ctx context.Context) (string, error)
}

// Plugin installs background jobs.
type Plugin struct {
	// MaxConcurrent bounds running jobs per owner in the default store. Zero uses 15; -1 disables the ceiling.
	MaxConcurrent int
	// Store owns the work. nil uses a fresh LocalJobStore for the run.
	Store JobStore
	// MaxResultBytes bounds a job result held in the store, so a completed job's
	// output cannot blow the context when the model reads it. 0 uses the run's
	// limits.MaxToolResultLen.
	MaxResultBytes int
	// MaxWaitSeconds bounds one job_wait call so the model cannot park the run
	// forever on a job that never finishes. 0 uses defaultJobMaxWait.
	MaxWaitSeconds int
	// SuppressCompletionNotice turns OFF the synthetic user message that lists
	// jobs finished since the last turn. The notice is on by default because
	// without it the model must poll job_status, burning a turn per check — and
	// a model that forgets to poll never learns its work finished at all.
	SuppressCompletionNotice bool
}

// Local installs the in-process job store — the right default for a single
// server. A custom store must preserve run ownership and cancellation; durable
// host tasks belong on the consumer's existing task queue.
func Local() Plugin { return Plugin{} }

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "jobs" }

// Register adds the plugin as a run extension.
func (p Plugin) Register(r *agentcore.Registry) error {
	r.AddExtension(p)
	return nil
}

// BeginRun resolves the policy and binds a launcher to THIS run.
//
// info.Owner is the fence: it is the session id on a durable run and a unique
// token otherwise, so a job started here is invisible to every other run and a
// tool cannot forge another run's owner — it never sees one.
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if p.MaxConcurrent < -1 {
		return nil, fmt.Errorf("jobs: MaxConcurrent must be nonnegative or -1")
	}
	store := p.Store
	if store == nil {
		local := NewLocalJobStore()
		local.maxConcurrent = p.MaxConcurrent
		store = local
	}
	maxBytes := p.MaxResultBytes
	if maxBytes <= 0 {
		maxBytes = info.Limits.MaxToolResultLen
	}
	if maxBytes <= 0 {
		maxBytes = defaultJobResultBytes
	}
	wait := time.Duration(p.MaxWaitSeconds) * time.Second
	if wait <= 0 {
		wait = defaultJobMaxWait
	}
	return &jobsRun{
		store:    store,
		owner:    info.Owner,
		maxWait:  wait,
		notify:   !p.SuppressCompletionNotice,
		maxBytes: maxBytes,
	}, nil
}

// Name identifies the extension in composition diagnostics.
func (*jobsRun) Name() string { return "jobs" }

// Tools contributes the four observation tools.
func (p *jobsRun) Tools() []agentcore.Tool {
	return []agentcore.Tool{
		jobListTool{jobTool{policy: p, name: "job_list"}},
		jobStatusTool{jobTool{policy: p, name: "job_status"}},
		jobWaitTool{jobTool{policy: p, name: "job_wait"}},
		jobCancelTool{jobTool{policy: p, name: "job_cancel"}},
	}
}

// SelfGated exempts the job_* tools from the permission gate. They observe and
// cancel work THIS run launched — the owner fence makes that literal — so they
// grant no capability the agent did not already exercise by starting the job.
func (jobTool) SelfGated() bool { return true }

// RunContext binds the launcher to the run's context, which is how a tool finds
// it. The context is the run's, not the calling tool's, so work survives the
// call that started it and still dies with the run.
func (p *jobsRun) RunContext(ctx context.Context) context.Context {
	launcher := Launcher{store: p.store, owner: p.owner, base: ctx, maxBytes: p.maxBytes}
	ctx = withJobs(ctx, launcher)
	return agentcore.WithBackgroundLauncher(ctx, func(tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error) {
		job, err := launcher.Start(tool, label, run)
		if err != nil {
			return nil, err
		}
		return json.Marshal(job)
	})
}

// CloseRun cancels everything this run started. A job outliving its run would
// be unobservable and uncancellable — nothing left to look at it.
func (p *jobsRun) CloseRun() {
	p.store.CancelAll(p.owner)
	// Cooperative workers (including native model streams) settle before usage
	// is folded. An arbitrary host job cannot hold shutdown forever.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, job := range p.store.List(p.owner) {
		if !job.State.Done() {
			_, _ = p.store.Wait(ctx, p.owner, job.ID)
		}
	}
}

// A normal final answer must account for the work the run launched. Wait here
// instead of allowing a confident finish to silently cancel pending children.
func (p *jobsRun) TurnStopping(ctx context.Context, _ agentcore.StopInfo) agentcore.StopDecision {
	waitCtx, cancel := context.WithTimeout(ctx, min(p.maxWait, 5*time.Second))
	defer cancel()
	for _, job := range p.store.List(p.owner) {
		if !job.State.Done() {
			if settled, err := p.store.Wait(waitCtx, p.owner, job.ID); err != nil || !settled.State.Done() {
				if ctx.Err() != nil {
					return agentcore.StopDecision{StopReason: "aborted"}
				}
				return agentcore.StopDecision{Continue: true, Inject: []agentcore.Message{{Role: agentcore.RoleUser, Content: "Background work is still pending. Use job_wait/status to collect it, or job_cancel if it is no longer needed; do not claim it completed."}}}
			}
		}
	}
	completed := p.store.DrainCompleted(p.owner)
	if len(completed) == 0 {
		return agentcore.StopDecision{}
	}
	return agentcore.StopDecision{Continue: true, Inject: []agentcore.Message{{Role: agentcore.RoleUser, Content: completionNotice(completed, p.maxBytes)}}}
}

// BeforeStep announces jobs that finished since the last turn.
//
// The notice is a synthetic user message the LOOP appends and persists, so the
// model learns its background work is done without spending a turn on a poll —
// and a resumed run replays the same conversation.
func (p *jobsRun) BeforeStep(context.Context, agentcore.StepInfo) agentcore.StepDecision {
	if !p.notify {
		return agentcore.StepDecision{}
	}
	drainer, ok := p.store.(interface{ DrainCompleted(owner string) []Job })
	if !ok {
		return agentcore.StepDecision{}
	}
	done := drainer.DrainCompleted(p.owner)
	text := completionNotice(done, jobNoticeInlineLimit)
	if text == "" {
		return agentcore.StepDecision{}
	}
	return agentcore.StepDecision{
		AdditionalContexts: []agentcore.Message{{Role: agentcore.RoleUser, Content: text}},
	}
}

const (
	defaultJobMaxWait     = 120 * time.Second
	defaultJobResultBytes = 16 * 1024
)

// jobsRun is one run's resolved jobs configuration — the agentcore.Extension
// the loop talks to.
type jobsRun struct {
	store    JobStore
	owner    string
	maxWait  time.Duration
	notify   bool
	maxBytes int
}

// Launcher is what a tool receives from the run context: a store already
// bound to this run's owner, so a tool can start background work without being
// handed — or being able to forge — another session's owner token.
type Launcher struct {
	store    JobStore
	owner    string
	base     context.Context
	maxBytes int
}

// Start launches background work and returns immediately. The job's context is
// derived from the RUN's context, not the calling tool's, so the work is not
// cancelled when the tool call that started it returns — that is the entire
// point — but still dies with the run.
func (l Launcher) Start(tool, label string, run func(ctx context.Context) (string, error)) (Job, error) {
	if l.store == nil {
		return Job{}, errors.New("agentcore: background jobs are not enabled for this run")
	}
	bounded := func(ctx context.Context) (string, error) {
		out, err := run(ctx)
		return agentcore.TruncateMiddle(out, l.maxBytes), err
	}
	return l.store.Start(l.base, l.owner, JobSpec{Tool: tool, Label: label, Run: bounded})
}

type jobsCtxKey struct{}

// withJobs puts a bound launcher on the context for tools to find.
func withJobs(ctx context.Context, l Launcher) context.Context {
	return context.WithValue(ctx, jobsCtxKey{}, l)
}

// From returns the background-job launcher for the current run, if this plugin
// is installed. A tool that wants to go asynchronous calls this and degrades
// gracefully when it is absent; a tool that does not is unaffected. That is the
// whole ejection contract for jobs: the ok result is the plugin's presence.
func From(ctx context.Context) (Launcher, bool) {
	l, ok := ctx.Value(jobsCtxKey{}).(Launcher)
	return l, ok
}

// completionNotice renders the synthetic user message announcing finished jobs.
// It carries the result inline for a small success so the common case costs the
// model no extra tool call, and points at job_status when the payload is large.
func completionNotice(jobs []Job, inlineLimit int) string {
	if len(jobs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Background job update:\n")
	for _, j := range jobs {
		fmt.Fprintf(&b, "- %s (%s, %s, %s)", j.ID, j.Tool, j.State, j.Duration().Round(time.Millisecond))
		switch {
		case j.State == JobFailed:
			fmt.Fprintf(&b, ": %s", j.Err)
		case j.State == JobSucceeded && len(j.Result) <= inlineLimit:
			fmt.Fprintf(&b, ":\n%s", j.Result)
		case j.State == JobSucceeded:
			fmt.Fprintf(&b, ": %d bytes of output — call job_status with id %q to read it.", len(j.Result), j.ID)
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// jobNoticeInlineLimit bounds how much of a finished job's output rides into the
// completion notice before the model is told to fetch it explicitly.
const jobNoticeInlineLimit = 2 * 1024

// LocalJobStore runs jobs as goroutines in this process, fenced by owner.
type LocalJobStore struct {
	maxConcurrent int
	mu            sync.Mutex
	jobs          map[string]*localJob // keyed by owner + "/" + id
	pending       map[string][]string  // owner -> ids finished but not yet drained
	seq           int
}

type localJob struct {
	job    Job
	cancel context.CancelFunc
	done   chan struct{}
	owner  string
}

// NewLocalJobStore builds an empty in-process store.
func NewLocalJobStore() *LocalJobStore {
	return &LocalJobStore{jobs: make(map[string]*localJob), pending: make(map[string][]string)}
}

func jobKey(owner, id string) string { return owner + "/" + id }

// Start launches the work in a goroutine and records it as Running.
func (s *LocalJobStore) Start(ctx context.Context, owner string, spec JobSpec) (Job, error) {
	if spec.Run == nil {
		return Job{}, errors.New("agentcore: job spec has no work to run")
	}
	if err := ctx.Err(); err != nil {
		return Job{}, err
	}
	s.mu.Lock()
	active := 0
	for _, job := range s.jobs {
		if job.owner == owner && !job.job.State.Done() {
			active++
		}
	}
	limit := s.maxConcurrent
	if limit == 0 {
		limit = 15
	}
	if limit > 0 && active >= limit {
		s.mu.Unlock()
		return Job{}, fmt.Errorf("agentcore: concurrent job capacity reached (%d)", limit)
	}
	s.seq++
	id := fmt.Sprintf("job_%d", s.seq)
	jobCtx, cancel := context.WithCancel(ctx)
	lj := &localJob{
		job:    Job{ID: id, Tool: spec.Tool, Label: spec.Label, State: JobRunning, StartedAt: time.Now()},
		cancel: cancel,
		done:   make(chan struct{}),
		owner:  owner,
	}
	s.jobs[jobKey(owner, id)] = lj
	snapshot := lj.job
	s.mu.Unlock()

	go func() {
		defer close(lj.done)
		// A panicking job must not take the process down: it becomes a failed
		// job, exactly like one that returned an error.
		var out string
		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("job panicked: %v", r)
				}
			}()
			out, err = spec.Run(jobCtx)
		}()

		s.mu.Lock()
		defer s.mu.Unlock()
		lj.job.EndedAt = time.Now()
		switch {
		case jobCtx.Err() != nil && err != nil:
			lj.job.State = JobCancelled
			lj.job.Err = jobCtx.Err().Error()
		case err != nil:
			lj.job.State = JobFailed
			lj.job.Err = err.Error()
		default:
			lj.job.State = JobSucceeded
			lj.job.Result = out
		}
		s.pending[owner] = append(s.pending[owner], id)
	}()

	return snapshot, nil
}

// Get returns a snapshot of one job, fenced by owner.
func (s *LocalJobStore) Get(owner, id string) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lj, ok := s.jobs[jobKey(owner, id)]
	if !ok {
		return Job{}, ErrJobNotFound
	}
	return lj.job, nil
}

// List returns the owner's jobs, newest first.
func (s *LocalJobStore) List(owner string) []Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Job, 0, 8)
	for _, lj := range s.jobs {
		if lj.owner == owner {
			out = append(out, lj.job)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out
}

// Cancel requests cancellation; cancelling a finished job is a no-op.
func (s *LocalJobStore) Cancel(owner, id string) error {
	s.mu.Lock()
	lj, ok := s.jobs[jobKey(owner, id)]
	s.mu.Unlock()
	if !ok {
		return ErrJobNotFound
	}
	lj.cancel()
	return nil
}

// Wait blocks until the job is terminal or ctx ends.
func (s *LocalJobStore) Wait(ctx context.Context, owner, id string) (Job, error) {
	s.mu.Lock()
	lj, ok := s.jobs[jobKey(owner, id)]
	s.mu.Unlock()
	if !ok {
		return Job{}, ErrJobNotFound
	}
	select {
	case <-lj.done:
	case <-ctx.Done():
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return lj.job, nil
}

// DrainCompleted returns and clears the owner's finished-since-last-drain set.
func (s *LocalJobStore) DrainCompleted(owner string) []Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := s.pending[owner]
	if len(ids) == 0 {
		return nil
	}
	delete(s.pending, owner)
	out := make([]Job, 0, len(ids))
	for _, id := range ids {
		if lj, ok := s.jobs[jobKey(owner, id)]; ok {
			out = append(out, lj.job)
		}
	}
	return out
}

// CancelAll stops every running job for an owner.
func (s *LocalJobStore) CancelAll(owner string) {
	s.mu.Lock()
	cancels := make([]context.CancelFunc, 0, 4)
	for _, lj := range s.jobs {
		if lj.owner == owner && !lj.job.State.Done() {
			cancels = append(cancels, lj.cancel)
		}
	}
	s.mu.Unlock()
	for _, c := range cancels {
		c()
	}
}

// ---- model-facing tools -------------------------------------------------

// jobTool is the shared base for the built-in job_* tools.
type jobTool struct {
	policy *jobsRun
	name   string
}

func (t jobTool) Name() string { return t.name }

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
	jobs := t.policy.store.List(t.policy.owner)
	if len(jobs) == 0 {
		return "No background jobs.", nil
	}
	var b strings.Builder
	for _, j := range jobs {
		fmt.Fprintf(&b, "%s\t%s\t%s\t%s\t%s\n", j.ID, j.State, j.Tool, j.Duration().Round(time.Millisecond), j.Label)
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
	j, err := t.policy.store.Get(t.policy.owner, id)
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
	wait := t.policy.maxWait
	if in.Timeout > 0 && time.Duration(in.Timeout)*time.Second < wait {
		wait = time.Duration(in.Timeout) * time.Second
	}
	waitCtx, cancel := context.WithTimeout(ctx, wait)
	defer cancel()
	j, err := t.policy.store.Wait(waitCtx, t.policy.owner, in.ID)
	if err != nil {
		return "", err
	}
	if !j.State.Done() {
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
	if err := t.policy.store.Cancel(t.policy.owner, id); err != nil {
		return "", err
	}
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
func renderJob(j Job) string {
	head := fmt.Sprintf("job %s (%s) — %s after %s", j.ID, j.Tool, j.State, j.Duration().Round(time.Millisecond))
	switch j.State {
	case JobSucceeded:
		if j.Result == "" {
			return head + "\n(no output)"
		}
		return head + "\n" + j.Result
	case JobFailed, JobCancelled:
		return head + "\n" + j.Err
	default:
		return head
	}
}

// ToolNames are the tools this plugin contributes, for a consumer that wants to
// name them in a policy or a UI.
var ToolNames = []string{"job_list", "job_status", "job_wait", "job_cancel"}
