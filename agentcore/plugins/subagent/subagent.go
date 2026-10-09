// Package subagent installs spawn_subagent: self-forking and cross-agent
// delegation under shared depth and budget caps.
//
// What it buys the parent is CONTEXT, not compute: a child explores in its own
// isolated history and returns only its final answer, so the tool churn, dead
// ends, and large intermediate results never enter the parent's window.
//
// A spawned child is an ordinary governed tool call — same policy gate, same
// credential boundary, same trace — not a privileged side channel. Depth rides
// the context, so an A -> B -> A cycle is bounded by the same MaxDepth as a
// straight chain, and the plugin declines a run already at the cap rather than
// advertising a tool that would only refuse.
package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/telemetry"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

// ToolSpawnSubagent is the stable name of the built-in delegation tool
// (ARCHITECT-AGENT-TEAM P1): the model forks an ephemeral child agent for one
// self-contained task and receives only its final answer, keeping the child's
// exploration (tool churn, dead ends, large intermediate results) out of the
// parent's context window.
const ToolSpawnSubagent = "spawn_subagent"

// Sub-agent defaults (ARCHITECT-AGENT-TEAM suggested caps).
const (
	defaultSubagentMaxDepth       = 1         // only the top-level agent may spawn
	defaultSubagentMaxOutputBytes = 48 * 1024 // model-visible answer per child
)

// The default spawn budget SCALES WITH THE RUN, because a fixed number cannot
// be right for both shapes of run this package serves.
//
// A spawn is an ordinary tool call, so MaxToolCalls is already the hard ceiling
// on how many are even reachable; MaxPerRun is the second, tighter bound that
// exists because a child is not a tool call's worth of work but a whole run's.
// Expressing it as a SHARE of the tool budget is what makes it transferable: it
// says "at most a third of this run's actions may be delegation", which is a
// claim about the shape of the run rather than about its length.
//
// At DefaultLimits (24 tool calls) the share reproduces the previous fixed
// default of 8 exactly, so nothing changes for a short run. A long autonomous
// run that authorizes thousands of tool calls gets a proportionate delegation
// budget instead of running out of children on its ninth spawn and being told
// to "finish the remaining work yourself" for the rest of the task — which is
// how a fixed 8 failed: it was calibrated against a 12-turn run and silently
// became a hard stop on delegation for anything longer.
//
// The floor keeps a deliberately tool-starved run (a small MaxToolCalls set for
// some other reason) from losing delegation entirely.
//
// Cost note: the product MaxPerRun × the child's own MaxTurns is what a run can
// ultimately spend, and children inherit the parent's Limits, so raising
// MaxToolCalls raises the ceiling superlinearly. That is why this is a share
// and not simply "unbounded": a consumer that wants a specific number should
// set MaxPerRun explicitly rather than reach it by widening the tool budget.
const (
	defaultSubagentSpawnShare = 3 // one child per N tool calls of the run's budget
	minSubagentMaxPerRun      = 8 // floor, and the historical fixed default
)

// Plugin caps the delegation surface. A child inherits the parent's provider,
// model ladder, tools, policy, hooks, memory, and definition — it can never
// widen access — and runs with isolated history, so only its final answer
// (truncated to MaxOutputBytes) returns to the parent.
type Plugin struct {
	// AllowAsync offers run-owned jobs for ephemeral forks. Durable delegation
	// must continue using its host journal and reattachment protocol.
	AllowAsync bool
	// MaxDepth is how many nesting levels may spawn: 1 (the default) lets the
	// top-level agent spawn children but forbids grandchildren.
	MaxDepth int
	// MaxPerRun caps how many children one run may spawn in total. Zero derives
	// it from the run's own tool budget (a third of Limits.MaxToolCalls, floor
	// 8), so a long run gets a proportionate delegation budget without the
	// consumer having to restate one.
	MaxPerRun int
	// MaxOutputBytes caps the child answer surfaced to the parent model.
	MaxOutputBytes int
	// Delegates names the other agents this one may hand a task to. Each Run is
	// an opaque closure the consumer injects — agentcore never loads another
	// agent itself. Empty leaves only self-delegation.
	Delegates []Delegate
	// RunFork selects the consumer's runtime for self-delegation. The child is
	// always created by Agent.Fork first, so runtime selection cannot widen its
	// inherited capabilities. Nil uses the native engine with consumer checkpoints
	// for ephemeral runs; durable self-forks require a native session host.
	RunFork ForkRunner
}

// ForkRequest describes one isolated task or one corrective retry. Previous is
// the original result, including native state; adapters must not reconstruct a
// native transcript from its display-only Messages projection.
type ForkRequest struct {
	SessionID string
	Prompt    string
	Task      string
	Previous  *agentcore.RunResult
}

type ForkRunner func(context.Context, *agentcore.Agent, ForkRequest, agentcore.StreamSink) (agentcore.RunResult, error)

// SelfOnly enables delegation to ephemeral forks of this agent, with no
// cross-agent roster.
func SelfOnly() Plugin { return Plugin{} }

// To enables cross-agent delegation to the named teammates, plus self-forking.
func To(delegates ...Delegate) Plugin { return Plugin{Delegates: delegates} }

// Name identifies the plugin and the extension it installs.
func (Plugin) Name() string { return "subagent" }

// Register adds the plugin as a run extension.
func (p Plugin) Register(r *agentcore.Registry) error {
	r.AddExtension(p)
	return nil
}

// BeginRun installs the tool for a run still above the nesting cap, and
// DECLINES one that is already at it.
//
// Declining is how delegation bottoms out structurally: a run at MaxDepth is
// never offered the tool, so the cap is enforced by absence rather than by a
// refusal the model would have to read, reason about, and work around. The
// depth rides the context, so it survives crossing into another agent's run.
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	settings := p.normalized(info.Limits)
	if info.Depth >= settings.MaxDepth {
		return nil, nil
	}
	return &subagentTool{parent: info.Agent, settings: settings, durable: info.Durable}, nil
}

// normalized fills zero fields with the defaults, sizing the spawn budget to
// the run's own limits.
func (s Plugin) normalized(limits agentcore.Limits) Plugin {
	if s.MaxDepth <= 0 {
		s.MaxDepth = defaultSubagentMaxDepth
	}
	if s.MaxPerRun <= 0 {
		s.MaxPerRun = max(limits.MaxToolCalls/defaultSubagentSpawnShare, minSubagentMaxPerRun)
	}
	if s.MaxOutputBytes <= 0 {
		s.MaxOutputBytes = defaultSubagentMaxOutputBytes
	}
	return s
}

// Delegate is one named other agent this agent may hand a task to (cross-agent
// delegation). agentcore never builds the target itself: Run is an opaque
// closure the consumer injects, executing the target agent under its own
// identity — its persona, tools, policy, and secrets, not the caller's. The
// closure receives the caller's ctx (so cancelling the parent cancels the
// delegate, and the delegation depth carried on ctx caps recursion across
// agents) and an optional sink for live tool-activity notes; it returns the
// target's final answer plus its token usage for parent-run accounting.
type Delegate struct {
	// Name is the stable identifier the model selects with (the target agent's
	// human name). Matched case-insensitively.
	Name string
	// Description is a one-line hint helping the model pick the right teammate.
	Description string
	// Run executes the delegated task on the target agent.
	Run func(ctx context.Context, task string, sink agentcore.StreamSink) (string, agentcore.Usage, error)
}

// subagentTool is both the run's agentcore.Extension and the model-facing tool.
// One object because the per-run spawn counter belongs to both: the extension
// exists for exactly one run, and the counter is what makes MaxPerRun mean
// anything.
//
// It is subject to the normal permission gate (unlike read_skill or read_spill):
// spawning reaches capability the agent has NOT already exercised, so the
// consumer must permit spawn_subagent in its policy.
type subagentTool struct {
	parent   *agentcore.Agent
	settings Plugin
	durable  bool
	spawned  int32 // atomic; children spawned this run
}

// Name identifies both the extension and the tool — they are the same thing.
func (t *subagentTool) Tools() []agentcore.Tool { return []agentcore.Tool{t} }

func (t *subagentTool) Name() string { return ToolSpawnSubagent }

// Parallel opts spawn calls into concurrent batch execution: each child is a
// fresh Agent instance driving its own isolated history, so a fan-out turn
// ("spawn three children, then synthesize") runs them concurrently.
func (t *subagentTool) Parallel() bool { return true }

// RetrySafeCall marks a spawn call safe to re-issue after a crash only when it
// self-forks through a durable native host: the child session ID is tied to
// the recorded physical invocation, so a replayed self-fork reattaches — returning a completed child's
// recorded answer, or resuming an interrupted child from its own durable log —
// instead of running a duplicate child. A delegate-routed spawn has no such
// wiring (Delegate.Run is an opaque closure under the target agent's identity),
// so replaying it would re-run the teammate's entire task; those calls are left
// dangling for the model to decide. (On a storeless run recovery never happens,
// so the declaration is moot.)
func (t *subagentTool) RetrySafeCall(call agentcore.ToolCall) bool {
	if !t.durable || t.settings.RunFork == nil {
		return false
	}
	var in subagentArgs
	if err := json.Unmarshal([]byte(call.Arguments), &in); err != nil {
		return false
	}
	who := strings.TrimSpace(in.Agent)
	return strings.TrimSpace(in.Task) != "" && !in.Async && (who == "" || strings.EqualFold(who, "self"))
}

func (t *subagentTool) Schema() agentcore.ToolSchema {
	props := map[string]any{
		"task": map[string]any{
			"type":        "string",
			"description": "A bounded, self-contained objective with constraints and the evidence or artifact references to return. Supply only context authorized for the destination; the host may further restrict what is forwarded.",
		},
		"context": map[string]any{
			"type":        "string",
			"description": "Optional authorized background (identifiers, constraints, prior findings). Self-forks see only the supplied task/context; named targets receive the input and history their host permits. Never assume a target has seen this conversation.",
		},
		"output_schema": map[string]any{
			"type":        "object",
			"description": "Optional JSON Schema the sub-agent's final answer must satisfy. When set, the sub-agent is instructed to end with a bare JSON value, the answer is validated against the schema, and on a violation the sub-agent is re-opened once with the error. You receive the validated JSON — or the raw answer marked as validation-failed if the retry also fails.",
		},
	}
	if t.settings.AllowAsync && !t.durable {
		props["async"] = map[string]any{"type": "boolean", "description": "Return a job id immediately. Use job_wait/status/cancel to manage it; the job is cancelled when this run ends."}
	}
	// Tool advertisement already applies policy and run-depth gates. Keeping the
	// orchestration contract here avoids unconditional system instructions for a
	// capability that is absent, and keeps it in sync with the actual parameters.
	desc := "Delegate one bounded task and collect its final answer. " +
		"Use delegation when isolated exploration or a matching specialist helps; handle ordinary reasoning, writing and quick tool lookups directly when your own capabilities suffice. Respect explicit no-tool requests. " +
		"A self-fork inherits your capabilities subject to host restrictions, with fresh isolated history; its intermediate work does not enter your conversation. " +
		"Give a clear objective, constraints and expected evidence. Collect required results, reconcile gaps, and synthesize one coherent answer for the caller. " +
		"Report missing access, failures and unresolved work honestly; a partial result is not completed work, and an uncertain external effect must not be repeated. " +
		"Delegation never grants new authority. Pass output_schema when a validated structured answer is needed."
	if roster := t.settings.Delegates; len(roster) > 0 {
		var lines []string
		for _, d := range roster {
			line := d.Name
			if d.Description != "" {
				line += " — " + d.Description
			}
			lines = append(lines, line)
		}
		props["agent"] = map[string]any{
			"type": "string",
			"description": "Which agent runs the task. Omit (or \"self\") to fork a clone of yourself. " +
				"Available teammates (each runs under its OWN persona, tools, and permissions — pick the one whose specialty matches the task): " +
				strings.Join(lines, "; "),
		}
		desc += " Select a named target through agent only when its stated purpose covers the requested task, project or resource. " +
			"The roster is not a general search pool: do not probe unrelated or previously used targets as a substitute for an unavailable destination. " +
			"If no target covers the request and you lack the needed access, report the missing source or capability; if the target is ambiguous, ask for concrete task context, not roster management. " +
			"Each named target uses its own host-defined identity and permissions; its description is a selection hint, not an authorization grant."
	}
	if t.settings.AllowAsync && !t.durable {
		desc += " For independent work, async:true returns a job receipt. A receipt is not completion: collect the result with job_wait/status, or cancel unneeded work with job_cancel, before claiming success."
	}
	return agentcore.ToolSchema{
		Name:        ToolSpawnSubagent,
		Description: desc,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   []string{"task"},
		},
	}
}

// subagentArgs is the decoded argument shape.
type subagentArgs struct {
	Task         string          `json:"task"`
	Async        bool            `json:"async"`
	Context      string          `json:"context"`
	Agent        string          `json:"agent"`
	OutputSchema json.RawMessage `json:"output_schema"`
}

func (t *subagentTool) Run(ctx context.Context, args string) (string, error) {
	return t.RunStreaming(ctx, args, nil)
}

// RunStreaming spawns and drives one child run — a fork of the parent by
// default, or a named delegate agent when the agent argument selects one. When
// emit is set (a streamed parent run), the child's tool activity is forwarded
// as brief partial notes so a live viewer (chat, Lab) sees the delegation
// working rather than a silent long call. The child shares the parent's ctx,
// so cancelling the parent run cancels in-flight children (team-architecture
// safety rule), and the ctx carries the incremented delegation depth so a
// delegate cannot re-delegate past MaxDepth (A→B→A recursion stops).
func (t *subagentTool) RunStreaming(ctx context.Context, args string, emit func(partial string)) (string, error) {
	var in subagentArgs
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	task := strings.TrimSpace(in.Task)
	if task == "" {
		return "", fmt.Errorf("task is required")
	}
	// Compile the schema before any child runs: an invalid schema is a caller
	// bug, and failing here costs nothing while failing after the child ran
	// costs a whole delegation.
	schema, err := compileOutputSchema(in.OutputSchema)
	if err != nil {
		return "", err
	}
	// Defense in depth: the loop already stops advertising the tool at MaxDepth,
	// but the ctx check also protects consumers that seed a depth floor.
	depth := agentcore.DelegationDepth(ctx)
	if depth >= t.settings.MaxDepth {
		return "", fmt.Errorf("delegation depth exhausted (max %d) — finish the work yourself", t.settings.MaxDepth)
	}
	if n := atomic.AddInt32(&t.spawned, 1); int(n) > t.settings.MaxPerRun {
		return "", fmt.Errorf("sub-agent budget exhausted (%d per run) — finish the remaining work yourself", t.settings.MaxPerRun)
	}
	if in.Async {
		if !t.settings.AllowAsync || t.parent.IsDurable() {
			return "", errors.New("async delegation is not available for this run")
		}
		values := context.WithoutCancel(ctx)
		receipt, err := agentcore.LaunchBackground(ctx, ToolSpawnSubagent, task, func(runCtx context.Context) (string, error) {
			childCtx := agentcore.WithDelegationDepth(asyncContext{Context: runCtx, values: values}, depth+1)
			return t.execute(childCtx, in, task, schema, func(note string) {
				agentcore.ReportProgress(childCtx, note)
			})
		})
		if err != nil {
			return "", err
		}
		return string(receipt), nil
	}
	ctx = agentcore.WithDelegationDepth(ctx, depth+1)
	return t.execute(ctx, in, task, schema, emit)
}

type asyncContext struct {
	context.Context
	values context.Context
}

func (c asyncContext) Value(key any) any { return c.values.Value(key) }

func (t *subagentTool) execute(ctx context.Context, in subagentArgs, task string, schema *jsonschema.Schema, emit func(string)) (string, error) {
	ctx = telemetry.DetachSafeExecution(ctx, telemetry.CategoryChild)
	var err error

	prompt := task
	if c := strings.TrimSpace(in.Context); c != "" {
		prompt = task + "\n\nContext:\n" + c
	}
	if schema != nil {
		prompt += outputSchemaInstruction(in.OutputSchema)
	}

	// Resolve the target: self-fork by default, a granted teammate when named.
	label := "sub-agent"
	var delegate *Delegate
	if who := strings.TrimSpace(in.Agent); who != "" && !strings.EqualFold(who, "self") {
		for i := range t.settings.Delegates {
			if strings.EqualFold(t.settings.Delegates[i].Name, who) {
				delegate = &t.settings.Delegates[i]
				break
			}
		}
		if delegate == nil {
			return "", fmt.Errorf("unknown agent %q — use \"self\" or one of the teammates listed in the tool description", who)
		}
		label = delegate.Name
	}

	// Forward the child's tool activity as partial notes on a streamed run.
	var sink agentcore.StreamSink
	if emit != nil {
		sink = func(ev agentcore.StreamEvent) {
			if ev.Type == agentcore.StreamToolExecStart && ev.Tool != nil {
				emit(fmt.Sprintf("[%s] running %s", label, ev.Tool.Tool))
			}
			// Surface the child's own progress notes (reattach / resume / budget)
			// so the parent's viewer sees why a spawn returned instantly.
			if ev.Type == agentcore.StreamProgress && ev.Note != "" {
				emit(fmt.Sprintf("[%s] %s", label, ev.Note))
			}
		}
	}

	var final string
	var res agentcore.RunResult
	if delegate != nil {
		var usage agentcore.Usage
		final, usage, err = delegate.Run(ctx, prompt, sink)
		res.Final = final
		// Fold the delegate's spend into the parent run before handling the
		// error, so even a failed delegate's tokens/cost are accounted.
		t.parent.AddChildUsage(usage)
		if err != nil {
			return "", fmt.Errorf("agent %s failed: %w", delegate.Name, err)
		}
	} else {
		// A durable parent gives the child a durable session of its own, with an
		// ID derived deterministically from the recorded invocation: the same
		// logical spawn always maps to the same child session (pi's deterministic
		// child-session IDs). A replayed spawn call therefore REATTACHES instead
		// of duplicating: a child that already completed returns its recorded
		// answer without re-running (no duplicate spend or side effects), and a
		// child that crashed mid-run resumes from its own log. This is what makes
		// spawn_subagent safe to declare RetrySafe.
		childSession, sessionErr := t.childSession(ctx)
		if sessionErr != nil {
			return "", sessionErr
		}
		child := t.parent.Fork(childSession)
		if t.settings.RunFork != nil {
			res, err = t.settings.RunFork(ctx, child, ForkRequest{SessionID: childSession, Prompt: prompt, Task: task}, sink)
		} else {
			seed := []agentcore.Message{{Role: agentcore.RoleUser, Content: prompt}}
			res, err = child.RunNative(ctx, agentcore.NativeRun{Input: seed, Task: task, Sink: sink})
		}
		// Fold the child's spend before handling the error (a child's own
		// children are already folded into res.Usage by its native run, recursively).
		t.parent.AddChildUsage(res.Usage)
		if err != nil {
			return "", fmt.Errorf("sub-agent failed: %w", err)
		}
		// A cancelled run is not an error to its caller — the loop stops between
		// turns and returns what it has with StopReason "aborted" and a nil error,
		// which is right for a viewer that walked away. It is wrong here. res.Final
		// is then whatever the child last happened to say, mid-task, and returning
		// it hands the parent a killed child's partial state as its ANSWER: the
		// parent records the shard as reconciled, the batch looks complete, and the
		// work that was actually interrupted is never redone. Failing the spawn
		// instead puts an interrupted note in the transcript the model can act on,
		// and leaves the call replayable — the child session id is deterministic,
		// so a resume reattaches to the child's own log and finishes it.
		if res.StopReason == "aborted" {
			return "", fmt.Errorf("sub-agent was interrupted before it finished (stop reason: %s)", res.StopReason)
		}
		final = res.Final
		if strings.TrimSpace(final) == "" {
			return "", fmt.Errorf("sub-agent produced no answer (stop reason: %s)", res.StopReason)
		}
	}
	final = strings.TrimSpace(final)
	if final == "" {
		return "", fmt.Errorf("agent %s produced no answer", label)
	}
	// Validate BEFORE TruncateMiddle: a truncated answer must fail honestly
	// rather than pass on a prefix.
	if schema != nil {
		final, err = t.validateWithRetry(ctx, final, res, delegate, prompt, schema, sink)
		if err != nil {
			return "", err
		}
	}
	return agentcore.TruncateMiddle(final, t.settings.MaxOutputBytes), nil
}

// validateWithRetry enforces the output_schema contract after the child
// produced an answer: validate, and on a violation re-open the child exactly
// once with the error. A second failure returns the raw answer marked as
// validation-failed rather than erroring the spawn — the parent's model can
// still read the answer, and the note tells it the JSON is not trustworthy.
//
// An ephemeral correction resumes the native checkpoint. A durable correction
// uses a separate "<childSession>/retry" session seeded from that checkpoint;
// the original completed child remains immutable. The deterministic id keeps the whole spawn
// replay-safe: a re-issued spawn reattaches to whichever child log completed.
// A delegate has no transcript to re-open — Run is an opaque closure — so its
// retry is a single re-invocation carrying the error in the task.
func (t *subagentTool) validateWithRetry(ctx context.Context, final string, res agentcore.RunResult, delegate *Delegate, prompt string, schema *jsonschema.Schema, sink agentcore.StreamSink) (string, error) {
	verr := validateOutput(final, schema)
	if verr == nil {
		return final, nil
	}
	retryFinal, retryRes, retryErr := t.retryOnce(ctx, delegate, prompt, res, verr, sink)
	t.parent.AddChildUsage(retryRes.Usage)
	var question *agentcore.ChildQuestionError
	if errors.As(retryErr, &question) {
		return "", retryErr
	}
	// Same guard as the main spawn path: an aborted retry must not hand the
	// parent a killed child's partial answer — fall back to the first answer.
	if retryErr == nil && retryRes.StopReason == "aborted" {
		retryErr = fmt.Errorf("sub-agent retry was interrupted before it finished")
	}
	if retryErr == nil && strings.TrimSpace(retryFinal) != "" {
		rerr := validateOutput(retryFinal, schema)
		if rerr == nil {
			return strings.TrimSpace(retryFinal), nil
		}
		return strings.TrimSpace(retryFinal) + "\n\n[validation failed: " + rerr.Error() + "]", nil
	}
	return final + "\n\n[validation failed: " + verr.Error() + "]", nil
}

// retryOnce runs the single corrective attempt. Self-forks re-open the child's
// transcript in a fresh fork; delegates re-run the task with the error and the
// rejected answer appended.
func (t *subagentTool) retryOnce(ctx context.Context, delegate *Delegate, prompt string, res agentcore.RunResult, validationErr error, sink agentcore.StreamSink) (string, agentcore.RunResult, error) {
	if delegate != nil {
		retryTask := prompt + "\n\nYour previous answer failed output_schema validation: " + validationErr.Error() +
			"\nRejected answer:\n" + res.Final +
			"\nProduce ONLY a corrected JSON value matching the schema — no prose, no markdown fence."
		final, usage, err := delegate.Run(ctx, retryTask, sink)
		return final, agentcore.RunResult{Usage: usage}, err
	}
	retrySession, sessionErr := t.childSession(ctx)
	if sessionErr != nil {
		return "", agentcore.RunResult{}, sessionErr
	}
	if retrySession != "" {
		retrySession += "/retry"
	}
	child := t.parent.Fork(retrySession)
	if t.settings.RunFork != nil {
		correction := correctionMessage(validationErr).Content
		r, err := t.settings.RunFork(ctx, child, ForkRequest{SessionID: retrySession, Prompt: correction, Task: "correct the invalid final answer", Previous: &res}, sink)
		return r.Final, r, err
	}
	if len(res.NativeState) > 0 {
		r, err := child.RunNative(ctx, agentcore.NativeRun{State: res.NativeState, Input: []agentcore.Message{correctionMessage(validationErr)}, Task: "correct the invalid final answer", Sink: sink})
		return r.Final, r, err
	}
	return "", agentcore.RunResult{}, errors.New("native sub-agent correction requires its checkpoint")
}

// Durable child identity comes from the recorded physical invocation. Provider
// call IDs can repeat on later turns and cannot identify a durable child.
func (t *subagentTool) childSession(ctx context.Context) (string, error) {
	if !t.durable {
		return "", nil
	}
	if t.settings.RunFork == nil {
		return "", errors.New("durable sub-agent requires a native session host")
	}
	key, ok := agentcore.IdempotencyKey(ctx)
	if !ok {
		return "", errors.New("durable sub-agent requires a recorded native invocation")
	}
	return t.parent.SessionID() + "/" + key, nil
}
