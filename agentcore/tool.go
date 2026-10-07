package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"unicode/utf8"
)

// Tool is a single capability the agent can invoke. Implementations are
// provided by the consumer (host-injected): the user's Agent Definition may
// reference and permit tools but can never conjure new ones, keeping the
// capability surface under code control.
type Tool interface {
	// Name is the stable identifier the model calls.
	Name() string
	// Schema advertises the tool to the model (JSON Schema parameters).
	Schema() ToolSchema
	// Run executes with validated JSON arguments and returns a result string
	// (already truncated by the loop before reaching the model).
	Run(ctx context.Context, args string) (string, error)
}

// ToolOutput is the optional structured result of a RichTool. Content remains
// the model-visible text result and is processed by the same interceptors,
// hooks, bounds, traces, and spill policy as Tool.Run. Parts are additive rich
// content and are bounded independently at the dispatch boundary; providers
// that cannot carry them degrade explicitly to text.
type ToolOutput struct {
	Content string
	Parts   []ContentPart

	// Invocations records governed tools called from inside this tool. It is
	// audit/control metadata, never provider-visible content. Eval uses it to
	// preserve host-tool bridge traces in the outer call's durable outcome.
	Invocations []ToolInvocation
	// AdditionalContexts and Terminate propagate the same decisions a direct
	// nested call would have produced, but the loop applies them only after the
	// outer tool result so provider tool-call/result adjacency remains valid.
	AdditionalContexts []Message
	Terminate          bool
}

// RichTool is an additive tool capability for outputs such as eval-generated
// images. Implementations still provide Run for direct/legacy callers; the
// agent loop prefers RunRich when it is available (unless a StreamingTool is
// actively streaming). It does not bypass the permission gate or any other
// execution policy.
type RichTool interface {
	RunRich(ctx context.Context, args string) (ToolOutput, error)
}

const (
	maxRichToolParts      = 16
	maxRichToolImages     = 8
	maxRichToolImageBytes = 768 * 1024
)

// boundToolContentParts prevents a trusted-but-buggy tool from bypassing the
// ordinary result bound through structured parts. Eval applies tighter MIME
// validation before this point; the kernel enforces the generic count, decoded
// image-byte, text-byte, and supported-shape limits for every RichTool.
func boundToolContentParts(parts []ContentPart, maxTextBytes int) ([]ContentPart, int, []string) {
	if maxTextBytes <= 0 {
		maxTextBytes = defaultMaxToolResultBytes
	}
	out := make([]ContentPart, 0, min(len(parts), maxRichToolParts))
	textBytes, imageBytes, images, omitted := 0, 0, 0, 0
	var notes []string
	for _, part := range parts {
		if len(out) >= maxRichToolParts {
			omitted++
			continue
		}
		switch part.Type {
		case ContentPartText:
			remaining := maxTextBytes - textBytes
			if part.Text == "" || remaining <= 0 {
				if part.Text != "" {
					omitted++
				}
				continue
			}
			part.Text = truncateMiddle(part.Text, remaining)
			part.Data, part.DataRef, part.MIMEType, part.Detail = "", "", "", ""
			textBytes += len(part.Text)
			out = append(out, part)
		case ContentPartImage:
			if images >= maxRichToolImages || part.Data == "" {
				omitted++
				continue
			}
			normalized, ok := normalizeRichImage(part, maxRichToolImageBytes-imageBytes)
			if !ok {
				omitted++
				continue
			}
			part = normalized.part
			imageBytes += normalized.bytes
			images++
			out = append(out, part)
			if note := normalized.dimensionNote(); note != "" {
				notes = append(notes, note)
			}
		default:
			omitted++
		}
	}
	return out, omitted, notes
}

// ErrParked is the sentinel a tool returns to park the run on a human answer
// (the ask tool): the loop records an EntryQuestion for the call, emits a
// StreamQuestion, and ends the run WITHOUT a tool result — the call stays
// dangling in the durable log until an EntryAnswer resolves it or a resume
// re-issues it. A tool that parks must also be retry-safe (RetrySafeTool /
// CallRetrySafeTool) or a crash-resume would close the call as interrupted
// instead of re-parking.
var ErrParked = errors.New("tool parked the run awaiting a human answer")

// ArgPreparer is an optional Tool capability (pi's prepareArguments): it
// normalizes the raw JSON argument string before validation and execution —
// defaulting fields, coercing shapes the model commonly gets wrong. A tool that
// does not implement it runs with the model's arguments verbatim.
type ArgPreparer interface {
	PrepareArguments(raw string) string
}

// StreamingTool is an optional Tool capability (pi's tool_execution_update): a
// long-running tool may emit partial output as it works via the supplied emit
// callback, which the loop forwards to the stream sink so a viewer sees progress
// before the tool finishes. The returned string is still the authoritative final
// result (identical to Run's), and a tool that does not implement this runs
// through Run unchanged. emit is a no-op on a non-streaming run.
type StreamingTool interface {
	RunStreaming(ctx context.Context, args string, emit func(partial string)) (string, error)
}

// ParallelTool is an optional Tool capability: a tool whose Parallel() returns
// true may be executed concurrently with the other parallel-eligible tool calls
// in the same assistant turn (pi's executionMode). The safe default is
// sequential — tools that mutate state or depend on call ordering must NOT
// implement this, so concurrency is opt-in per read-only tool.
type ParallelTool interface {
	Parallel() bool
}

// ToolSet is an ordered registry of tools keyed by name.
type ToolSet struct {
	order      []string
	byKey      map[string]Tool
	validators *toolValidatorCache
}

// NewToolSet builds a registry from the given tools, preserving order.
func NewToolSet(tools ...Tool) *ToolSet {
	ts := &ToolSet{
		byKey:      make(map[string]Tool, len(tools)),
		validators: newToolValidatorCache(),
	}
	for _, t := range tools {
		ts.Add(t)
	}
	return ts
}

// Add registers a tool, overwriting any existing tool of the same name.
func (ts *ToolSet) Add(t Tool) {
	if _, exists := ts.byKey[t.Name()]; !exists {
		ts.order = append(ts.order, t.Name())
	}
	ts.byKey[t.Name()] = t
}

// With returns a COPY of the set with the given tools added, leaving the
// receiver untouched. The loop assembles a run's effective registry this way —
// host tools plus whichever built-ins the run enables (read_skill,
// spawn_subagent, read_spill, job_*, session_query) — so the shared Agent
// toolset is never mutated per run and two concurrent runs of the same
// definition cannot see each other's built-ins.
func (ts *ToolSet) With(tools ...Tool) *ToolSet {
	clone := &ToolSet{
		order:      append([]string{}, ts.order...),
		byKey:      make(map[string]Tool, len(ts.byKey)+len(tools)),
		validators: ts.validators,
	}
	for k, v := range ts.byKey {
		clone.byKey[k] = v
	}
	for _, t := range tools {
		clone.Add(t)
	}
	return clone
}

// toolValidatorCache is shared by ToolSet.With clones. It stores at most one
// compiled schema per tool name, capped across dynamic extension names, so
// repeated runs avoid recompilation without creating unbounded state in a
// long-lived server. The schema fingerprint prevents a same-named override (or
// a dynamic Schema method) from ever reusing a validator for different
// parameters.
type toolValidatorCache struct {
	mu      sync.Mutex
	entries map[string]toolValidatorEntry
}

const maxToolValidatorCacheEntries = 256

func newToolValidatorCache() *toolValidatorCache {
	return &toolValidatorCache{entries: make(map[string]toolValidatorEntry)}
}

// Get returns the tool of the given name, if registered.
func (ts *ToolSet) Get(name string) (Tool, bool) {
	t, ok := ts.byKey[name]
	return t, ok
}

// Schemas returns the advertised schemas in registration order.
func (ts *ToolSet) Schemas() []ToolSchema {
	out := make([]ToolSchema, 0, len(ts.order))
	for _, name := range ts.order {
		out = append(out, ts.byKey[name].Schema())
	}
	return out
}

// Names returns tool names in registration order.
func (ts *ToolSet) Names() []string {
	out := make([]string, len(ts.order))
	copy(out, ts.order)
	return out
}

// gateExemptTools are the built-ins that bypass the permission gate because
// neither can reach anything the agent does not already have. read_skill only
// returns definition-authored skill bodies from this agent's own definition.
//
// An EXTENSION-contributed tool earns the same bypass by declaring
// SelfGated (see extension.go) — that keeps the claim next to the capability
// that makes it true, and keeps it auditable: the loop can list exactly which
// plugin asked to skip the gate. Anything that CAN reach outside the run (a
// filesystem read, an HTTP fetch, a SQL query) stays gated, plugin or not.
var gateExemptTools = map[string]bool{
	readSkillToolName: true,
}

// defaultMaxToolResultBytes bounds a tool result before it reaches the LLM —
// token + safety guard (pi harness truncate.ts, rebuilt UTF-8-safe).
const defaultMaxToolResultBytes = 24 * 1024

// truncateBytes trims s to at most maxBytes without splitting a UTF-8 rune,
// appending a marker when it cuts. A maxBytes <= 0 disables truncation.
func truncateBytes(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	const marker = "\n…[truncated]"
	budget := maxBytes - len(marker)
	if budget <= 0 {
		budget = maxBytes
	}
	// Back off to a rune boundary.
	for budget > 0 && !utf8.RuneStart(s[budget]) {
		budget--
	}
	return s[:budget] + marker
}

// truncateMiddle trims an oversized string to at most maxBytes by cutting its
// MIDDLE out, keeping the head and the tail verbatim with an omission marker
// between them. Tool results get this shape (not head-only truncateBytes)
// because the end of a long result usually carries the signal — the error after
// pages of build output, a query's final rows, a stack trace's root cause. The
// head gets roughly two thirds of the budget, the tail one third; cuts never
// split a UTF-8 rune. A maxBytes <= 0 disables truncation.
func truncateMiddle(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	// Reserve room for the marker (its digits vary; 64 covers any length).
	const reserve = 64
	budget := maxBytes - reserve
	if budget < 2 {
		return truncateBytes(s, maxBytes)
	}
	head := budget * 2 / 3
	for head > 0 && !utf8.RuneStart(s[head]) {
		head--
	}
	start := len(s) - (budget - head)
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[:head] + fmt.Sprintf("\n…[%d bytes truncated]…\n", start-head) + s[start:]
}

// StringTool declares a callback tool with string-valued JSON arguments. The
// host supplies only its schema and action; AgentCore validates and dispatches
// it through the same governance boundary as any other Tool.
type StringTool struct {
	ToolName, Description string
	Properties            map[string]any
	Required              []string
	Execute               func(context.Context, map[string]string) (string, error)
}

func (t StringTool) Name() string { return t.ToolName }
func (t StringTool) Schema() ToolSchema {
	return ToolSchema{Name: t.ToolName, Description: t.Description, Parameters: map[string]any{"type": "object", "properties": t.Properties, "required": t.Required, "additionalProperties": false}}
}
func (t StringTool) Run(ctx context.Context, args string) (string, error) {
	var input map[string]string
	if err := json.Unmarshal([]byte(args), &input); err != nil {
		return "", err
	}
	if t.Execute == nil {
		return "", fmt.Errorf("tool %s has no executor", t.ToolName)
	}
	return t.Execute(ctx, input)
}

// StringProperties builds bounded, non-empty text arguments for a declaration.
func StringProperties(names ...string) map[string]any {
	properties := make(map[string]any, len(names))
	for _, name := range names {
		properties[name] = map[string]any{"type": "string", "minLength": 1, "maxLength": 2048}
	}
	return properties
}
