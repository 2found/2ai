// Package todo contributes a live run plan: a checklist the model writes for
// itself and that the loop pins into every request.
//
// The plan is deliberately NOT part of the transcript. It lives in a Store the
// tool writes and a context hook reads, so compaction can never trim it — even
// after the original task and all early turns are summarized away, the freshly
// rendered checklist is right there in front of the model. That is the whole
// point of the capability: it is what keeps a long autonomous run on its
// original objective.
//
// Tools and request hooks share a run-bound store. Native checkpoints preserve
// its bounded plan state; forks get a fresh store so child plans cannot replace
// the parent's checklist.
package todo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

// ToolName is the stable name the model calls, and the name a policy must
// permit.
const ToolName = "update_plan"

// Todo status values. A well-formed plan has at most one in_progress item — the
// single thing the agent is doing right now — mirroring a focused worklist.
const (
	StatusPending    = "pending"
	StatusInProgress = "in_progress"
	StatusCompleted  = "completed"
	StatusBlocked    = "blocked"
	StatusAbandoned  = "abandoned"
)

// Item is one step in the run's plan: a short imperative description and its
// current status.
type Item struct {
	ID      string `json:"id,omitempty"`
	Phase   string `json:"phase,omitempty"`
	Content string `json:"content"`
	Status  string `json:"status"`
}

// The plan's ceilings.
//
// Everything pinned into every request needs one, and the plan needs it more
// than most: it is immune to compaction BY CONSTRUCTION (that is the feature),
// so nothing else in the system will ever bring it back down. An agent that
// decomposes as it discovers — which is what a long run does — appends steps and
// leaves the finished ones behind as the record that the work happened. Three
// hundred turns of that is 25 KB in the prefix of every call for the rest of the
// run, crowding out the window the plan exists to keep the model oriented in.
//
// The numbers are chosen from what each part is FOR, not from a uniform budget:
//
//   - keepCompleted — finished steps are a progress signal, not a working set.
//     The most recent few say "here is where you are"; the two hundredth-most
//     recent says nothing the running count does not.
//   - maxItemBytes — a plan item is a short imperative. A model that writes a
//     paragraph into one is the other way this block bloats, and it is not
//     bounded by item count at all.
//   - maxRenderBytes — the last-resort ceiling, so the invariant "the pinned
//     plan is bounded" holds no matter what shape the plan takes. At the loop's
//     ~4-bytes-per-token estimate this is ~500 tokens.
const (
	keepCompleted  = 5
	maxItemBytes   = 160
	maxRenderBytes = 2000
)

// Store holds a single run's live plan. It is the out-of-band state that
// makes a long run goal-stable: the plan is owned here, not in the transcript,
// so compaction can never trim it. The loop re-injects a rendering of it before
// every native provider request (PiContextHook), so the model always sees its own
// up-to-date checklist regardless of how much history was summarized away.
//
// A Store is safe for concurrent use; the tool writes it while a context
// hook reads it.
type Store struct {
	mutation sync.Mutex
	mu       sync.RWMutex
	items    []Item
	// retired counts completed steps folded away to keep the pinned plan bounded.
	// It is a floor, not an exact tally: the model sends the full list each time,
	// so a step it drops from its own list on its own initiative is never counted
	// here. That is the right direction to be wrong in — the number never claims
	// progress that did not happen.
	retired int
}

// NewStore returns an empty plan store.
func NewStore() *Store { return &Store{} }

// Set replaces the whole plan. The model always sends the full list (not a
// delta), so a replace is the correct semantics and keeps the store trivially
// consistent. Items are copied so a later caller mutation cannot alias the store.
//
// Set also folds: completed steps beyond the most recent few are dropped from
// the list and added to a running count. This is what keeps a long run's plan
// from becoming a transcript. It is done HERE rather than only in Render so that
// what the model reads and what the store holds are the same thing — a model
// shown a folded list sends the folded list back, and a store that quietly held
// more would re-expand the render on the next write.
func (s *Store) Set(items []Item) {
	s.mutation.Lock()
	defer s.mutation.Unlock()
	s.set(items)
}

func (s *Store) set(items []Item) {
	cp := make([]Item, 0, len(items))
	completed := 0
	for _, it := range items {
		it.Content = clampItem(it.Content)
		it.Phase = clampItem(it.Phase)
		if it.Status == StatusCompleted {
			completed++
		}
		cp = append(cp, it)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if drop := completed - keepCompleted; drop > 0 {
		kept := make([]Item, 0, len(cp)-drop)
		for _, it := range cp {
			// Oldest first: the list is ordered, so the ones at the front are the
			// ones furthest from what the agent is doing now.
			if it.Status == StatusCompleted && drop > 0 {
				drop--
				s.retired++
				continue
			}
			kept = append(kept, it)
		}
		cp = kept
	}
	s.items = cp
}

// Retired reports how many completed steps have been folded into the count.
func (s *Store) Retired() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.retired
}

// clampItem bounds one step's text. A plan item is a short imperative; anything
// longer is prose that belongs in the answer, not in a block reprinted on every
// turn for the rest of the run.
func clampItem(content string) string {
	content = strings.TrimSpace(content)
	if len(content) <= maxItemBytes {
		return content
	}
	return agentcore.TruncateBytes(content, maxItemBytes)
}

// List returns a copy of the current plan.
func (s *Store) List() []Item {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Item, len(s.items))
	copy(out, s.items)
	return out
}

// Render formats the plan as a compact checklist for the model. It returns the
// empty string when there is no plan yet, so the context hook injects nothing
// until the agent has actually written one.
func (s *Store) Render() string {
	s.mu.RLock()
	items := make([]Item, len(s.items))
	copy(items, s.items)
	retired := s.retired
	s.mu.RUnlock()

	if len(items) == 0 && retired == 0 {
		return ""
	}

	head := "Current plan (your live todo list — keep it updated with update_plan):\n"
	if retired > 0 {
		head += fmt.Sprintf("[x] (%d earlier steps completed)\n", retired)
	}

	lines := make([]string, len(items))
	for i, it := range items {
		label := it.Content
		if it.Phase != "" {
			label = "[" + it.Phase + "] " + label
		}
		if it.ID != "" {
			label = it.ID + ": " + label
		}
		lines[i] = statusBox(it.Status) + " " + label
	}
	return strings.TrimRight(head+fitLines(lines, items, maxRenderBytes-len(head)), "\n")
}

// fitLines is the last-resort ceiling on the rendered plan, and it drops with a
// priority rather than a rule of thumb: whatever else goes, the step the agent
// is ON stays. That is the one line the model cannot choose its next action
// without, and it is not necessarily near the front of the list.
//
// Everything else is kept in the list's own order from the front, which favors
// the steps nearest the current work over a long pending tail, and the remainder
// is accounted for rather than silently vanished.
func fitLines(lines []string, items []Item, budget int) string {
	total := 0
	for _, l := range lines {
		total += len(l) + 1
	}
	if total <= budget {
		return strings.Join(lines, "\n") + "\n"
	}

	keep := make([]bool, len(lines))
	used := 0
	// Reserve the active step first, before anything can crowd it out.
	for i, it := range items {
		if it.Status == StatusInProgress {
			keep[i] = true
			used += len(lines[i]) + 1
		}
	}
	for i := range lines {
		if keep[i] {
			continue
		}
		if used+len(lines[i])+1 > budget {
			continue
		}
		keep[i] = true
		used += len(lines[i]) + 1
	}

	var b strings.Builder
	dropped := 0
	for i, l := range lines {
		if !keep[i] {
			dropped++
			continue
		}
		b.WriteString(l)
		b.WriteByte('\n')
	}
	if dropped > 0 {
		fmt.Fprintf(&b, "… (%d further steps in the plan, not shown here)\n", dropped)
	}
	return b.String()
}

func statusBox(status string) string {
	switch status {
	case StatusCompleted:
		return "[x]"
	case StatusInProgress:
		return "[~]"
	case StatusBlocked:
		return "[!]"
	case StatusAbandoned:
		return "[-]"
	default:
		return "[ ]"
	}
}

// ContextPrefix marks the injected plan reminder so it is recognizable in a
// transcript and never confused with model-authored content.
const ContextPrefix = "[run plan]"

// PiContextHook pins the bounded plan into Pi's outgoing native view.
// Raw provider messages pass through untouched, and the reminder never becomes
// another persisted conversation message on each turn.
func PiContextHook(store *Store) agentcore.PiContextHook {
	return func(ctx context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
		rendered := store.Render()
		if rendered == "" {
			return messages, nil
		}
		reminder, err := json.Marshal(map[string]any{"role": "system", "content": ContextPrefix + "\n" + rendered, "timestamp": time.Now().UnixMilli()})
		if err != nil {
			return nil, err
		}
		out := make([]json.RawMessage, 0, len(messages)+1)
		out = append(out, messages...)
		return append(out, reminder), nil
	}
}

// planTool is the model-facing tool that writes the run plan.
type planTool struct {
	store *Store
}

// NewTool returns the built-in update_plan tool bound to a run's plan store.
// The model calls it to record and revise its checklist for a multi-step task;
// the stored plan is then pinned into every later native request by PiContextHook.
func NewTool(store *Store) agentcore.Tool { return &planTool{store: store} }

func (t *planTool) Name() string { return ToolName }

func (t *planTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolName,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Record or update your plan as a todo list for a multi-step task. " +
			"Send the FULL list every time (it replaces the previous plan). Mark exactly one " +
			"item in_progress (the step you are doing now), completed for finished steps, and " +
			"pending for the rest; use blocked for waiting on a dependency and abandoned for explicitly dropped work. " +
			"Give steps stable IDs and optional phase labels; patch_plan can update identified steps atomically. The plan is pinned into your context and survives summarization, " +
			"so use it to stay on the original goal across a long run. Because it is pinned, keep it " +
			"a plan and not a log: one short line per step. Older completed steps are folded into a " +
			"running count automatically — that count is not an item, and you do not need to restate " +
			"the steps behind it.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"items": map[string]any{
					"type":        "array",
					"description": "The full ordered todo list.",
					"maxItems":    128,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"id":      map[string]any{"type": "string", "maxLength": 64, "description": "Stable item identifier for patch_plan."},
							"phase":   map[string]any{"type": "string", "maxLength": 160, "description": "Optional phase, such as investigate, implement, verify."},
							"content": map[string]any{"type": "string", "description": "Short imperative description of the step."},
							"status": map[string]any{
								"type":        "string",
								"enum":        []string{StatusPending, StatusInProgress, StatusCompleted, StatusBlocked, StatusAbandoned},
								"description": "Step status.",
							},
						},
						"required": []string{"content", "status"},
					},
				},
			},
			"required": []string{"items"},
		},
	}
}

func (t *planTool) Run(ctx context.Context, args string) (string, error) {
	items, err := parseItems(args)
	if err != nil {
		return "", err
	}
	store := t.store
	store.Set(items)
	return "Plan updated.\n" + store.Render(), nil
}

// parseItems decodes and validates one update_plan payload. It is separate from
// Run because resume recovery replays the same arguments out of the log and must
// reach the same verdict — a call the original run rejected must not become a
// plan the recovered run holds.
func parseItems(args string) ([]Item, error) {
	var in struct {
		Items []Item `json:"items"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return nil, fmt.Errorf("update_plan: invalid arguments: %w", err)
	}
	if len(in.Items) > 128 {
		return nil, fmt.Errorf("update_plan: at most 128 items")
	}
	ids := map[string]bool{}
	inProgress := 0
	for i, it := range in.Items {
		if len(it.ID) > 64 || len(it.Phase) > 160 {
			return nil, fmt.Errorf("update_plan: oversized id or phase")
		}
		if it.ID != "" {
			if ids[it.ID] {
				return nil, fmt.Errorf("update_plan: duplicate id %q", it.ID)
			}
			ids[it.ID] = true
		}
		if strings.TrimSpace(it.Content) == "" {
			return nil, fmt.Errorf("update_plan: item %d has empty content", i+1)
		}
		switch it.Status {
		case StatusPending, StatusInProgress, StatusCompleted, StatusBlocked, StatusAbandoned:
		case "":
			in.Items[i].Status = StatusPending
		default:
			return nil, fmt.Errorf("update_plan: item %d has invalid status %q", i+1, it.Status)
		}
		if in.Items[i].Status == StatusInProgress {
			inProgress++
		}
	}
	if inProgress > 1 {
		return nil, fmt.Errorf("update_plan: at most one item may be in_progress (got %d)", inProgress)
	}
	return in.Items, nil
}

// Bookkeeping marks update_plan calls as administrative rather than progress.
//
// A turn spent only on plan updates is refunded against MaxTurns, so an agent
// that keeps its checklist honest is not punished for it — without this, a
// careful planner finishes fewer steps than a careless one on the same budget.
// The MaxToolCalls budget still bounds a runaway planning loop.
func (*planTool) Bookkeeping() bool { return true }

// Plugin contributes the run plan: the update_plan tool and the context hook
// that pins the plan into every request.
type Plugin struct {
	// Store holds this run's plan. Required — a plan with nowhere to live is a
	// tool that silently forgets, which is worse for the model than no tool at
	// all. One Store per run; sharing one across concurrent runs would let two
	// agents overwrite each other's checklist.
	Store *Store
	// CheckCompletion asks the model to resolve pending/in_progress steps before
	// accepting a normal finish. Explicitly blocked or abandoned steps may remain.
	CheckCompletion bool
	// MaxCompletionNudges optionally bounds repair attempts. Nonpositive means
	// unlimited. Exhaustion stops as todo_incomplete, never successful completion.
	MaxCompletionNudges int
}

// With builds the plugin around a store.
func With(s *Store) Plugin { return Plugin{Store: s} }

// Name identifies the plugin.
func (Plugin) Name() string { return "todo" }

// Register installs a run-scoped extension. Its tools, native context transform
// and checkpoint all use the same isolated plan, including on child forks.
func (p Plugin) Register(r *agentcore.Registry) error {
	if p.Store == nil {
		return errNoStore
	}
	r.AddExtension(p)
	return nil
}

type runPlan struct {
	store               *Store
	checkCompletion     bool
	maxCompletionNudges int
}

func (r *runPlan) TurnStopping(_ context.Context, info agentcore.StopInfo) agentcore.StopDecision {
	if !r.checkCompletion {
		return agentcore.StopDecision{}
	}
	unfinished := false
	for _, item := range r.store.List() {
		if item.Status == StatusPending || item.Status == StatusInProgress {
			unfinished = true
			break
		}
	}
	if !unfinished {
		return agentcore.StopDecision{}
	}
	if r.maxCompletionNudges > 0 && info.Attempt >= r.maxCompletionNudges {
		return agentcore.StopDecision{StopReason: "todo_incomplete"}
	}
	return agentcore.StopDecision{
		Continue: true,
		Inject:   []agentcore.Message{{Role: agentcore.RoleUser, Content: "Your plan still has pending or in_progress steps. Continue the remaining work and verify results before marking them completed. If you cannot proceed, mark the affected steps blocked and explain the dependency; use abandoned only for explicitly dropped work. Do not mark steps completed just to finish.\n" + r.store.Render()}},
		Note:     "Resolving unfinished plan steps before finishing.",
	}
}

func (*runPlan) Name() string { return "todo" }
func (r *runPlan) Tools() []agentcore.Tool {
	return []agentcore.Tool{NewTool(r.store), &patchTool{r.store}}
}
func (r *runPlan) TransformNativeContext(ctx context.Context, messages []json.RawMessage) ([]json.RawMessage, error) {
	return PiContextHook(r.store)(ctx, messages)
}
func (r *runPlan) NativeState() (json.RawMessage, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()
	return json.Marshal(struct {
		Items   []Item `json:"items"`
		Retired int    `json:"retired"`
	}{r.store.items, r.store.retired})
}
func (r *runPlan) RestoreNativeState(raw json.RawMessage) error {
	items, err := parseItems(string(raw))
	if err != nil {
		return err
	}
	var state struct {
		Retired int `json:"retired"`
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		return err
	}
	if state.Retired < 0 {
		return fmt.Errorf("todo: negative retired count")
	}
	r.store.mu.Lock()
	r.store.items = nil
	r.store.retired = state.Retired
	r.store.mu.Unlock()
	r.store.Set(items)
	return nil
}

func (p Plugin) BeginRun(ctx context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	store := p.Store
	// A fork inherits capabilities, never the parent's mutable checklist.
	if info.Depth > 0 {
		store = NewStore()
	}
	if store == nil {
		return nil, errNoStore
	}
	if info.Session != nil && len(store.List()) == 0 {
		entries, err := info.Session.Log(ctx, info.SessionID)
		if err != nil {
			return nil, err
		}
		if items, ok := planFromLog(entries); ok {
			store.Set(items)
		}
	}
	return &runPlan{store: store, checkCompletion: p.CheckCompletion, maxCompletionNudges: p.MaxCompletionNudges}, nil
}

// planFromLog folds the log down to the plan in force: the last update_plan call
// wins, and a leaf clears it, mirroring how the loop recovers the goal. A run
// chained onto a finished session starts with a clean checklist rather than
// inheriting the previous task's.
//
// Only successful native effect receipts count. Requested calls and failed or
// denied executions cannot replace the accepted plan. The tool validator also
// rejects malformed receipt arguments.
func planFromLog(entries []agentcore.SessionEntry) ([]Item, bool) {
	var found []Item
	ok := false
	for _, e := range entries {
		apply := func(trace agentcore.ToolTrace, executed bool) {
			if !executed || !trace.Allowed || trace.Error != "" {
				return
			}
			if trace.Tool == ToolName {
				if items, err := parseItems(trace.Args); err == nil {
					found, ok = items, true
				}
			}
			if trace.Tool == PatchToolName {
				if items, err := patchItems(found, trace.Args); err == nil {
					found, ok = items, true
				}
			}
		}
		switch e.Kind {
		case agentcore.EntryPiEffectDone:
			// Only a settled, successful execution can update the native plan.
			// An assistant's requested arguments are not proof of execution.
			var receipt struct {
				Error  string
				Result struct{ Details agentcore.PiToolOutcome }
			}
			if json.Unmarshal([]byte(e.Content), &receipt) != nil || receipt.Error != "" {
				continue
			}
			audit := receipt.Result.Details
			apply(audit.Trace, audit.Executed)
			for _, call := range audit.Invocations {
				apply(call.Trace, call.Executed)
			}
		case agentcore.EntryLeaf:
			found, ok = nil, false
		}
	}
	return found, ok
}

// errNoStore refuses a composition that asks for a plan with nowhere to keep it.
var errNoStore = fmt.Errorf("todo: Plugin requires a Store")
