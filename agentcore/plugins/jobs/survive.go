package jobs

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

// Survival is the opt-in that lets a background job outlive the run that
// started it. Set it as Plugin.Survive.
//
// The model chooses: only a job it marks with job_keep survives; every other
// job is cancelled when the run ends, exactly as without this option. A kept
// job keeps running after the run's final answer. It is listed, readable and
// cancellable (job_list, job_status, job_wait, job_cancel) from the next run of
// the SAME conversation and from no other. Its result is announced exactly
// once: in that next run's first step, or, when it ends while no run of the
// conversation is attached, through Hold's callback.
//
// What the host must do:
//
//   - Create one Hold for the process (NewHold) and give every run the same one.
//   - Tell the library which conversation a run belongs to: WithConversation on
//     the context passed to RunNative, or a durable session id (RunInfo.SessionID).
//     A run with neither cannot keep jobs.
//   - Implement the Hold's callback to start a new run of that conversation
//     carrying Ended.Notice as a user message.
//   - Persist the native checkpoint as it already does. A kept job lives in the
//     process: when the process restarts the Hold is empty, and the restored
//     checkpoint reports the job as interrupted.
//
// A job started under one run keeps that run's context values, so its policy,
// credentials and session stay those of the run that started it; the next run
// only observes and cancels it.
//
// Model-backed work cannot be kept. spawn_subagent folds its child's token
// usage into the Agent that spawned it, and that Agent's run has ended by the
// time a surviving child would finish: the usage would be lost or charged to
// the wrong run. job_keep refuses it with that reason. Shell commands and host
// tools can be kept; Exclude refuses more tools that call a model.
type Survival struct {
	// Hold is the host-owned home of kept jobs between runs. Required.
	Hold *Hold
	// MaxAge bounds a kept job's lifetime, counted from its start; at that age it
	// is cancelled and reported cancelled. Zero uses one hour.
	MaxAge time.Duration
	// MaxJobs bounds how many kept jobs one conversation may have running. Zero
	// uses four.
	MaxJobs int
	// Exclude names more tools whose jobs job_keep refuses, besides
	// spawn_subagent.
	Exclude []string
}

const (
	defaultMaxAge  = time.Hour
	defaultMaxKept = 4
	// subagentToolName is spawn_subagent's tool name, spelled out because
	// plugins do not import each other.
	subagentToolName = "spawn_subagent"
	// sendWait is how long job_send waits for a just-started job to register
	// its message handler.
	sendWait    = 250 * time.Millisecond
	maxSendSize = 4 * 1024
)

type survival struct {
	h       *Hold
	maxAge  time.Duration
	maxJobs int
	exclude map[string]bool
}

func (s *survival) hold() *Hold {
	if s == nil {
		return nil
	}
	return s.h
}

// Ended tells the host that a kept job ended while no run of its conversation
// was attached.
type Ended struct {
	Conversation string
	JobID        string
	Tool         string
	Label        string
	// State is "succeeded", "failed" or "cancelled".
	State string
	// Notice is the completion notice (the text the next run's first step would
	// have carried), ready to be sent as a user message of a new run.
	Notice string
}

// Hold is the host-owned home of kept jobs between runs, keyed by conversation.
// It owns no storage and starts nothing: it only keeps the goroutines of jobs
// the model chose to keep reachable. Create one per process with NewHold.
type Hold struct {
	mu    sync.Mutex
	onEnd func(Ended)
	convs map[string]*heldConv
}

type heldConv struct {
	jobs     []*job // detached, oldest first
	attached *board
	seq      int
}

// NewHold creates a Hold. onEnd, when set, is called once for each kept job that
// ends while no run of its conversation is attached, and for a kept job that
// had ended unannounced when its run closed. It runs on the job's goroutine:
// return quickly and start the run elsewhere. It is never called for a job a
// run is attached to; that run announces the job itself.
func NewHold(onEnd func(Ended)) *Hold {
	return &Hold{onEnd: onEnd, convs: map[string]*heldConv{}}
}

// CancelAll cancels every detached job still running, for host shutdown.
func (h *Hold) CancelAll() {
	h.mu.Lock()
	var running []*job
	for _, c := range h.convs {
		for _, j := range c.jobs {
			if !j.done() {
				running = append(running, j)
			}
		}
	}
	h.mu.Unlock()
	for _, j := range running {
		j.cancel()
	}
}

func (h *Hold) notify(e *Ended) {
	if h == nil || e == nil || h.onEnd == nil {
		return
	}
	defer func() { _ = recover() }() // a host callback must not kill the process
	h.onEnd(*e)
}

type conversationKey struct{}

// WithConversation names the conversation a run belongs to, for Survive. The
// host sets it on the context it passes to RunNative; it is the identity under
// which kept jobs are held and found again. A durable session id is used when
// the context carries none.
func WithConversation(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, conversationKey{}, id)
}

func conversation(ctx context.Context, info agentcore.RunInfo) string {
	if id, _ := ctx.Value(conversationKey{}).(string); id != "" {
		return id
	}
	return info.SessionID
}

// attach resolves the config, joins the Hold's mutex and adopts the
// conversation's kept jobs.
func (b *board) attach(ctx context.Context, info agentcore.RunInfo, cfg *Survival) error {
	if cfg.Hold == nil {
		return errors.New("jobs: Survival.Hold is required")
	}
	if cfg.MaxAge < 0 || cfg.MaxJobs < 0 {
		return errors.New("jobs: Survival.MaxAge and MaxJobs must not be negative")
	}
	s := &survival{h: cfg.Hold, maxAge: cfg.MaxAge, maxJobs: cfg.MaxJobs, exclude: map[string]bool{subagentToolName: true}}
	if s.maxAge == 0 {
		s.maxAge = defaultMaxAge
	}
	if s.maxJobs == 0 {
		s.maxJobs = defaultMaxKept
	}
	for _, tool := range cfg.Exclude {
		s.exclude[tool] = true
	}
	b.survive, b.mu = s, &s.h.mu
	b.conv = conversation(ctx, info)
	if b.conv == "" {
		return nil
	}
	h := s.h
	h.mu.Lock()
	defer h.mu.Unlock()
	c := h.convs[b.conv]
	if c == nil {
		c = &heldConv{}
		h.convs[b.conv] = c
	}
	if c.attached != nil {
		return fmt.Errorf("jobs: conversation %q already has a run attached", b.conv)
	}
	now := time.Now()
	for _, j := range c.jobs {
		if j.done() && j.Announced && now.Sub(j.EndedAt) > s.maxAge {
			continue
		}
		j.owner = b
		b.jobs = append(b.jobs, j)
	}
	c.jobs, c.attached, b.seq = nil, b, c.seq
	return nil
}

// detach hands the run's kept jobs to the Hold when the run closes. A kept job
// that already ended unannounced is announced through the callback now: no run
// remains to carry it.
func (b *board) detach() {
	if b.survive == nil || b.conv == "" {
		return
	}
	h := b.survive.h
	h.mu.Lock()
	c := h.convs[b.conv]
	if c == nil || c.attached != b {
		h.mu.Unlock()
		return
	}
	var ends []*Ended
	for _, j := range b.jobs {
		if !j.Keep {
			continue
		}
		j.owner = nil
		c.jobs = append(c.jobs, j)
		if j.done() && !j.Announced {
			ends = append(ends, j.endedLocked())
		}
	}
	c.attached, c.seq = nil, max(c.seq, b.seq)
	h.evictLocked(b.conv, c)
	h.mu.Unlock()
	for _, e := range ends {
		h.notify(e)
	}
}

// endedDetachedLocked is the job's end as the Hold sees it: a callback payload
// when no run owns the job, nil when a run does (that run announces it).
func (j *job) endedDetachedLocked() *Ended {
	if j.owner != nil || j.hold == nil {
		return nil
	}
	e := j.endedLocked()
	if c := j.hold.convs[j.conv]; c != nil {
		j.hold.evictLocked(j.conv, c)
	}
	return e
}

// endedLocked marks the job announced and describes it for the host.
func (j *job) endedLocked() *Ended {
	j.Announced = true
	return &Ended{Conversation: j.conv, JobID: j.ID, Tool: j.Tool, Label: j.Label, State: string(j.State), Notice: completionNotice([]record{j.record}, noticeInlineLimit)}
}

// evictLocked drops announced finished detached jobs beyond keepFinished, and
// the conversation entry once nothing is left in it.
func (h *Hold) evictLocked(conv string, c *heldConv) {
	finished := 0
	for _, j := range c.jobs {
		if j.done() {
			finished++
		}
	}
	kept := c.jobs[:0]
	for _, j := range c.jobs {
		if finished > keepFinished && j.done() && j.Announced {
			finished--
			continue
		}
		kept = append(kept, j)
	}
	clear(c.jobs[len(kept):])
	c.jobs = kept
	if len(c.jobs) == 0 && c.attached == nil {
		delete(h.convs, conv)
	}
}

// keep marks a running job to outlive the run.
func (b *board) keep(id string) (string, error) {
	if b.survive == nil {
		return "", errors.New("jobs cannot outlive the run here")
	}
	if b.conv == "" {
		return "", errors.New("this run has no conversation identity, so a job cannot outlive it")
	}
	j, err := b.find(id)
	if err != nil {
		return "", err
	}
	s := b.survive
	b.mu.Lock()
	defer b.mu.Unlock()
	switch {
	case j.done():
		return "", fmt.Errorf("job %s already finished (%s); there is nothing to keep", id, j.State)
	case j.Keep:
		return fmt.Sprintf("Job %s is already kept.", id), nil
	case s.exclude[j.Tool]:
		if j.Tool == subagentToolName {
			return "", fmt.Errorf("job %s is a sub-agent: its token usage is accounted to this run, so it cannot outlive it; use job_wait to collect it", id)
		}
		return "", fmt.Errorf("job %s (%s) cannot outlive the run: it is model-backed work whose usage this run accounts for", id, j.Tool)
	}
	kept := 0
	for _, o := range b.jobs {
		if o.Keep && !o.done() {
			kept++
		}
	}
	if kept >= s.maxJobs {
		return "", fmt.Errorf("this conversation already keeps %d running jobs (the maximum); cancel one with job_cancel first", kept)
	}
	j.Keep = true
	wait := max(s.maxAge-time.Since(j.StartedAt), 0)
	j.timer = time.AfterFunc(wait, func() {
		b.mu.Lock()
		if !j.done() {
			j.expired = fmt.Sprintf("cancelled: the kept job reached its maximum age of %s", s.maxAge)
		}
		b.mu.Unlock()
		j.cancel()
	})
	return fmt.Sprintf("Job %s will keep running after you answer, for at most %s from its start. Its result is announced when it ends; list or cancel it with job_list / job_cancel on a later turn.", id, s.maxAge), nil
}

// send delivers a message to a running job that accepts them.
func (b *board) send(id, message string) error {
	j, err := b.find(id)
	if err != nil {
		return err
	}
	deadline := time.Now().Add(sendWait)
	for {
		b.mu.Lock()
		handle, done, state, tool, young := j.onMsg, j.done(), j.State, j.Tool, time.Since(j.StartedAt) < time.Second
		b.mu.Unlock()
		switch {
		case done:
			return fmt.Errorf("job %s already finished (%s); it can no longer receive messages", id, state)
		case handle != nil:
			return handle(message)
		case !young || time.Now().After(deadline):
			return fmt.Errorf("job %s (%s) does not accept messages; only a running background sub-agent does", id, tool)
		}
		// A job that has only just started may not have registered yet.
		time.Sleep(5 * time.Millisecond)
	}
}

type jobKeepTool struct{ jobTool }

func (t jobKeepTool) Schema() agentcore.ToolSchema {
	s := t.board.survive
	return jobIDSchema("job_keep",
		fmt.Sprintf("Let a running background job outlive this run: a dev server, a long build. Call it right after starting the job, "+
			"then answer the person; the job keeps running and its result is announced when it ends. A job you do not keep is cancelled when you answer. "+
			"A kept job lives at most %s and each conversation keeps at most %d. Sub-agents cannot be kept. "+
			"Never keep work whose output you need for this answer: wait for it instead.", s.maxAge, s.maxJobs), nil)
}

func (t jobKeepTool) Run(_ context.Context, args string) (string, error) {
	id, err := jobIDArg(args)
	if err != nil {
		return "", err
	}
	return t.board.keep(id)
}

type jobSendTool struct{ jobTool }

func (t jobSendTool) Schema() agentcore.ToolSchema {
	return jobIDSchema("job_send",
		"Send a follow-up message to a running background sub-agent. It reads the message as a user message at its next step, "+
			"so use it to correct course or add a constraint without cancelling the work. Refused when the job already finished, "+
			"does not exist, or is not a sub-agent.",
		map[string]any{
			"message": map[string]any{"type": "string", "description": "What to tell the sub-agent."},
		})
}

func (t jobSendTool) Run(_ context.Context, args string) (string, error) {
	var in struct {
		ID      string `json:"id"`
		Message string `json:"message"`
	}
	if err := unmarshalArgs(args, &in); err != nil {
		return "", err
	}
	id, msg := strings.TrimSpace(in.ID), strings.TrimSpace(in.Message)
	switch {
	case id == "":
		return "", errors.New("id is required")
	case msg == "":
		return "", errors.New("message is required")
	case len(msg) > maxSendSize:
		return "", fmt.Errorf("message is %d bytes; at most %d", len(msg), maxSendSize)
	}
	if err := t.board.send(id, msg); err != nil {
		return "", err
	}
	return fmt.Sprintf("Message sent to %s. It reads it at its next step.", id), nil
}
