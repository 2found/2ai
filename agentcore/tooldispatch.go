package agentcore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Executing one tool call: lookup -> prepare -> validate -> gate -> run ->
// bound -> trace.
//
// This is the trust boundary of the whole runtime. Every rule that decides
// whether a call happens at all, what the model is allowed to see of its
// result, and what the durable trace records about it, is applied here and
// nowhere else — which is what makes the gate unskippable rather than
// merely usually-called.

// maxToolFailures is how many times in a row one tool may error within a single
// run before the loop disables it for the remainder of that run (a per-run
// circuit breaker). A successful execution resets the tool's counter.
const maxToolFailures = 3

// callTool executes a tool — streaming when an emit callback is supplied,
// otherwise plain Run — inside panic recovery. A panicking tool is converted to
// an error so one broken tool degrades to a normal error result instead of
// crashing the run (or, in the parallel dispatch path, the process).
func callTool(ctx context.Context, tool Tool, args string, emit func(partial string)) (out ToolOutput, err error) {
	defer func() {
		if r := recover(); r != nil {
			out, err = ToolOutput{}, fmt.Errorf("tool panicked: %v", r)
		}
	}()
	if st, ok := tool.(StreamingTool); ok && emit != nil {
		text, err := st.RunStreaming(ctx, args, emit)
		return ToolOutput{Content: text}, err
	}
	if rt, ok := tool.(RichTool); ok {
		return rt.RunRich(ctx, args)
	}
	text, err := tool.Run(ctx, args)
	return ToolOutput{Content: text}, err
}

// ToolTrace is a persisted projection of one tool execution (§9
// agent_tool_calls): tool name, validated args, whether it was allowed, and
// result metadata.
type ToolTrace struct {
	// CallID is the provider's id for this specific invocation. It is what makes a
	// trace addressable: two concurrent calls to the same tool differ in nothing
	// else, so a consumer keying on name (or on array position, which shifts as
	// the list grows) reconciles the wrong one onto the other. Empty only for a
	// synthesized trace with no originating model call.
	CallID     string `json:"call_id,omitempty"`
	Tool       string `json:"tool"`
	Args       string `json:"args"`
	Allowed    bool   `json:"allowed"`
	Reason     string `json:"reason,omitempty"`
	Error      string `json:"error,omitempty"`
	ResultMeta string `json:"result_meta,omitempty"`
	LatencyMS  int64  `json:"latency_ms,omitempty"` // wall-clock of the tool execution (0 when not executed)
	// IdempotencyKey is the framework-derived dedupe key handed to the tool
	// (empty on runs without a durable session). Persisting it lets an external
	// side effect be correlated back to the exact logical call that caused it.
	IdempotencyKey string `json:"idempotency_key,omitempty"`
	// SpillLocator is set when the result was too large for the context and its
	// full text was saved to a spill artifact (plugins/spill). It makes the complete
	// output recoverable from the trace long after the run, not just by the model
	// mid-run — an oversized result is no longer lost to observability either.
	SpillLocator string `json:"spill_locator,omitempty"`
}

// ToolDenialReason is why a tool call was refused (ToolTrace.Reason).
// Distinct from RunResult.StopReason — they share the "aborted" spelling
// by coincidence (a cancelled run stops with StopReason "aborted" AND
// remaining tool calls are denied with this reason). The JSON/wire value
// is the existing literal so rows already written keep classifying.
type ToolDenialReason string

const (
	// ToolDenialAborted is the loop's denial reason when a remaining call
	// is short-circuited because the run was cancelled (agentcore/loop.go).
	// ClassifyTool buckets this as ToolAborted; the loop itself also
	// compares against it to decide whether a result is settled enough to
	// persist. A typed constant so those two sites cannot drift from a typo.
	ToolDenialAborted ToolDenialReason = "aborted"
)

// DeniedAborted reports whether this trace is the loop's cancel-denial.
// Historical rows carry the bare string "aborted"; the comparison is on
// that wire value.
func (t ToolTrace) DeniedAborted() bool {
	return t.Reason == string(ToolDenialAborted)
}

// toolOutcome is the result of executing one tool call: the persisted trace, the
// tool-result message fed back to the model, whether the run should terminate,
// and whether it counted against the tool-call budget (only real executions do).
type toolOutcome struct {
	trace     ToolTrace
	message   Message
	terminate bool
	executed  bool
	// invocations are governed calls made from inside this tool (currently eval
	// kernels). They are recorded beside the outer trace and persisted inside
	// its outcome, but do not become standalone provider tool messages.
	invocations []ToolInvocation
	// extra are messages an extension asked to add because of this call. They
	// are buffered and appended by the caller AFTER every result in the batch,
	// so tool-call/result adjacency is never broken.
	extra []Message
	// parked marks a call that ended the run waiting on a human (ErrParked): no
	// tool-result message is produced — the call stays dangling in the durable
	// log behind an EntryQuestion until an EntryAnswer resolves it.
	parked bool
	// A delegated question retains its own session/physical identity. It must
	// not masquerade as a local ask with the spawn arguments as its question.
	childQuestion *ChildQuestionError
}

// runToolCall takes a single model tool call through lookup -> prepareArguments
// -> validate -> beforeToolCall (permission gate + hooks) -> execute ->
// afterToolCall. It touches no shared run state, so it is safe to call
// concurrently for parallel-eligible tools; budget counting and trace recording
// happen in the caller, in order.
func (a *Agent) runToolCall(ctx context.Context, exts *extensionSet, exempt map[string]bool, tools *ToolSet, call ToolCall, limits Limits, emitUpdate func(ToolCall, string)) toolOutcome {
	trace := ToolTrace{CallID: call.ID, Tool: call.Name, Args: call.Arguments}

	tool, ok := tools.Get(call.Name)
	if !ok {
		trace.Allowed = true
		trace.Error = "unknown tool"
		return toolOutcome{trace: trace, message: toolResult(call, "error: unknown tool "+call.Name)}
	}

	// Normalize arguments before validation (optional per-tool prepareArguments).
	args := call.Arguments
	if p, ok := tool.(ArgPreparer); ok {
		args = p.PrepareArguments(args)
	}
	trace.Args = args
	gated := call
	gated.Arguments = args

	// Validate arguments against the tool's schema before any hook runs.
	if err := tools.validateToolArgs(call.Name, args, tool.Schema().Parameters); err != nil {
		trace.Allowed = false
		trace.Error = err.Error()
		return toolOutcome{trace: trace, message: toolResult(call, "invalid arguments: "+err.Error())}
	}

	// beforeToolCall preflight (permission gate + consumer hooks). Two built-ins
	// bypass it because neither can reach anything the agent does not already
	// have: read_skill only returns definition-authored skill bodies, and
	// read_spill only returns output THIS session produced and had truncated
	// away (the store is fenced to the run's session id). Requiring every
	// marketplace preset to enumerate them would be friction with no security
	// value.
	if !gateExemptTools[call.Name] && !exempt[call.Name] {
		if d := a.hooks.runBefore(ctx, gated); !d.Allow {
			trace.Allowed = false
			trace.Reason = d.Reason
			return toolOutcome{trace: trace, message: toolResult(call, "blocked: "+d.Reason)}
		}
	}

	// Resolve credential placeholders at the trust boundary: the call was traced
	// (trace.Args) and gated above in {{cred:NAME}} placeholder form, so the
	// model and the persisted trace never see the literal. Only runArgs — handed
	// straight to the tool — carries the resolved secret. A resolver error fails
	// closed: the call is blocked and the reason fed back to the model.
	runArgs := args
	if a.env.Credentials != nil {
		resolved, err := a.env.Credentials.Resolve(ctx, args)
		if err != nil {
			trace.Allowed = false
			trace.Reason = err.Error()
			return toolOutcome{trace: trace, message: toolResult(call, "blocked: "+err.Error())}
		}
		runArgs = resolved
	}

	// A StreamingTool emits partials as it works (forwarded via emitUpdate); a
	// plain tool runs through Run. Either way out is the authoritative result.
	// callTool wraps execution in panic recovery so a misbehaving tool degrades
	// to an ordinary error result instead of crashing the run (or, on the parallel
	// dispatch path, the whole process).
	var emit func(partial string)
	if emitUpdate != nil {
		emit = func(partial string) { emitUpdate(gated, partial) }
	}
	// Hand the tool its crash-stable idempotency key: (sessionID, call ID) is
	// replayed verbatim by RecoverSession, so a re-run after a crash presents
	// the same key and the external system can dedupe the effect.
	identity := call.ID
	if scope := toolInvocationScope(ctx); scope != "" {
		identity = scope + "/" + call.ID
	}
	ikey := toolIdempotencyKey(a.sessionID, identity)
	trace.IdempotencyKey = ikey
	execStart := time.Now()
	toolCtx := withToolCallID(withIdempotencyKey(ctx, ikey), call.ID)
	if _, ok := ToolInvokerFrom(toolCtx); !ok {
		if budget := toolExecutionBudgetFrom(toolCtx); budget != nil {
			toolCtx = withToolInvoker(toolCtx, &nestedToolInvoker{
				agent: a, exts: exts, exempt: exempt, tools: tools, limits: limits,
				emitUpdate: emitUpdate, budget: budget,
			})
		}
	}
	toolCtx = withToolStack(toolCtx, call.Name)
	richOut, runErr := callTool(toolCtx, tool, runArgs, emit)
	var childQuestion *ChildQuestionError
	if errors.As(runErr, &childQuestion) {
		childQuestion = copyChildQuestion(childQuestion)
	}
	out := richOut.Content
	// A parked call ends the run here: no interceptors, no after hooks, no
	// result message — the caller records the EntryQuestion and stops. The
	// validated args ride the trace so the question lands in the log exactly as
	// the model will see it re-asked on a re-park.
	if errors.Is(runErr, ErrParked) {
		trace.Allowed = true
		trace.ResultMeta = "parked"
		return toolOutcome{trace: trace, parked: true, executed: true, invocations: richOut.Invocations}
	}
	trace.LatencyMS = time.Since(execStart).Milliseconds()
	if runErr == nil {
		var omitted int
		var imageNotes []string
		richOut.Parts, omitted, imageNotes = boundToolContentParts(richOut.Parts, limits.MaxToolResultLen)
		if omitted > 0 {
			note := fmt.Sprintf("[%d rich content part(s) omitted by tool-result limits]", omitted)
			imageNotes = append(imageNotes, note)
		}
		for _, note := range imageNotes {
			if richOut.Content == "" {
				richOut.Content = note
			} else {
				richOut.Content += "\n" + note
			}
		}
		out = richOut.Content
	}
	// Bound the result for the model. Interceptors see the RAW output first,
	// because a lossless bounding strategy (persist the whole thing, hand back a
	// preview plus a locator) cannot be built on top of an already-truncated
	// string. Whichever one replaces the result owns the bound; if none does,
	// the loop falls back to head+tail truncation — keeping the beginning AND
	// end, because the end often carries the signal (a shell error after pages
	// of build output, the final rows of a query, a stack trace's cause) — and
	// the middle is gone for good. The loop does not know which extension, if
	// any, took the job.
	out, meta, resultRef, extra, replaced, extTerm := exts.interceptToolResult(ctx, gated, out, runErr)
	if len(richOut.AdditionalContexts) > 0 {
		extra = append(append([]Message(nil), richOut.AdditionalContexts...), extra...)
	}
	if !replaced {
		out = truncateMiddle(out, limits.MaxToolResultLen)
	}
	trace.SpillLocator = resultRef
	out, term := a.hooks.runAfter(ctx, gated, out, runErr)
	term = term || extTerm || richOut.Terminate

	trace.Allowed = true
	if runErr != nil {
		trace.Error = runErr.Error()
		out = "error: " + runErr.Error()
		richOut.Parts = nil
	}
	trace.ResultMeta = fmt.Sprintf("%d bytes in %dms", len(out), trace.LatencyMS)
	if len(richOut.Parts) > 0 {
		trace.ResultMeta += fmt.Sprintf("; %d rich parts", len(richOut.Parts))
	}
	if len(richOut.Invocations) > 0 {
		trace.ResultMeta += fmt.Sprintf("; %d nested tool calls", len(richOut.Invocations))
	}
	if meta != "" {
		trace.ResultMeta += "; " + meta
	}
	message := toolResult(call, out)
	message.ContentParts = append([]ContentPart(nil), richOut.Parts...)
	message.ResultRef = resultRef
	return toolOutcome{
		trace: trace, message: message, terminate: term, executed: true, extra: extra,
		invocations:   append([]ToolInvocation(nil), richOut.Invocations...),
		childQuestion: childQuestion,
	}
}

// nestedToolInvoker re-enters the exact same dispatch boundary as a direct
// model call. It is intentionally created per outer call: sequence IDs are
// scoped to that call, while the execution budget is shared across the run.
type nestedToolInvoker struct {
	agent      *Agent
	exts       *extensionSet
	exempt     map[string]bool
	tools      *ToolSet
	limits     Limits
	emitUpdate func(ToolCall, string)
	budget     *toolExecutionBudget
	sequence   atomic.Uint64
	nestedMu   sync.RWMutex
}

func (i *nestedToolInvoker) InvokeTool(ctx context.Context, name, args string) (result ToolOutput, runErr error) {
	name = strings.TrimSpace(name)
	parentID, _ := ToolCallID(ctx)
	callID := fmt.Sprintf("%s/bridge-%d", parentID, i.sequence.Add(1))
	if parentID == "" {
		callID = fmt.Sprintf("bridge-%d", i.sequence.Load())
	}
	if events, ok := ctx.Value(nestedToolEventsKey{}).(func(StreamEvent)); ok {
		trace := ToolTrace{CallID: callID, Tool: name, Args: args}
		events(StreamEvent{Type: StreamToolExecStart, Tool: &trace})
		defer func() {
			if len(result.Invocations) > 0 {
				if result.Invocations[0].Trace.CallID == "" {
					result.Invocations[0].Trace.CallID = callID
				}
				trace = result.Invocations[0].Trace
			}
			events(StreamEvent{Type: StreamToolExecEnd, Tool: &trace})
			events(StreamEvent{Type: StreamTool, Tool: &trace})
		}()
	}
	if err := validateNestedToolTarget(ctx, name); err != nil {
		trace := ToolTrace{Tool: name, Args: args, Allowed: false, Reason: err.Error()}
		return ToolOutput{Invocations: []ToolInvocation{{Trace: trace}}}, err
	}
	if err := ctx.Err(); err != nil {
		trace := ToolTrace{Tool: name, Args: args, Allowed: false, Reason: string(ToolDenialAborted)}
		return ToolOutput{Invocations: []ToolInvocation{{Trace: trace}}}, err
	}
	if !i.budget.reserve() {
		err := errors.New("tool-call budget exhausted")
		trace := ToolTrace{Tool: name, Args: args, Allowed: false, Reason: err.Error()}
		return ToolOutput{Invocations: []ToolInvocation{{Trace: trace}}}, err
	}
	call := ToolCall{ID: callID, Name: name, Arguments: args}
	parallel := isParallelTool(i.tools, call)
	if parallel {
		i.nestedMu.RLock()
		defer i.nestedMu.RUnlock()
	} else {
		i.nestedMu.Lock()
		defer i.nestedMu.Unlock()
	}
	if err := ctx.Err(); err != nil {
		i.budget.release()
		trace := ToolTrace{CallID: callID, Tool: name, Args: args, Allowed: false, Reason: string(ToolDenialAborted)}
		return ToolOutput{Invocations: []ToolInvocation{{Trace: trace}}}, err
	}
	outcome := i.agent.runToolCall(ctx, i.exts, i.exempt, i.tools, call, i.limits, i.emitUpdate)
	if !outcome.executed {
		i.budget.release()
	}
	invocations := make([]ToolInvocation, 0, 1+len(outcome.invocations))
	invocations = append(invocations, ToolInvocation{Trace: outcome.trace, Executed: outcome.executed})
	invocations = append(invocations, outcome.invocations...)
	result = ToolOutput{
		Content:            outcome.message.Content,
		Parts:              append([]ContentPart(nil), outcome.message.ContentParts...),
		Invocations:        invocations,
		AdditionalContexts: append([]Message(nil), outcome.extra...),
		Terminate:          outcome.terminate,
	}
	if outcome.parked {
		return result, fmt.Errorf("tool %q cannot park for human input through a nested invocation; call it directly", name)
	}
	if !outcome.trace.Allowed {
		reason := strings.TrimSpace(outcome.trace.Reason)
		if reason == "" {
			reason = strings.TrimSpace(outcome.message.Content)
		}
		return result, fmt.Errorf("tool %q blocked: %s", name, reason)
	}
	if outcome.trace.Error != "" {
		return result, fmt.Errorf("tool %q failed: %s", name, outcome.trace.Error)
	}
	return result, nil
}

// isParallelTool reports whether one call targets a registered tool that opts
// into concurrent execution (ParallelTool). Unknown tools and tools that don't
// opt in run sequentially — the safe default.
func isParallelTool(tools *ToolSet, call ToolCall) bool {
	tool, ok := tools.Get(call.Name)
	if !ok {
		return false
	}
	p, ok := tool.(ParallelTool)
	return ok && p.Parallel()
}

// toolResult builds a tool-role message linked to its call.
func toolResult(call ToolCall, content string) Message {
	return Message{Role: RoleTool, ToolCallID: call.ID, Name: call.Name, Content: content}
}

// validateObject checks an arguments object against a JSON Schema object node:
// required presence first, then per-property type/enum for the fields present.
// It exists to produce concise common-case errors before the complete compiled
// JSON Schema validator handles nesting, arrays, combiners, patterns, bounds,
// and additionalProperties.
func validateObject(obj, schema map[string]any) error {
	if !isObjectSchema(schema) {
		return nil
	}
	if missing := missingRequired(obj, schema); len(missing) > 0 {
		return fmt.Errorf("missing required field(s): %s", strings.Join(missing, ", "))
	}
	props, _ := schema["properties"].(map[string]any)
	for name, raw := range obj {
		propSchema, ok := props[name].(map[string]any)
		if !ok {
			continue // unknown / additional property — permitted
		}
		if err := validateValue(name, raw, propSchema); err != nil {
			return err
		}
	}
	return nil
}

// validateValue checks one field value against its property schema (type + enum).
// Unknown or absent type constraints pass. A union type list — the common
// nullable form `"type": ["string", "null"]` — accepts a value matching ANY
// member; before this, a list-valued type skipped checking entirely, so a
// nullable field silently accepted every shape (pi #7243's nullable-schema
// validation gap).
func validateValue(name string, value any, propSchema map[string]any) error {
	if enum := stringOrAnySlice(propSchema["enum"]); len(enum) > 0 {
		if !enumContains(enum, value) {
			return fmt.Errorf("field %q must be one of %s", name, formatEnum(enum))
		}
	}
	switch typ := propSchema["type"].(type) {
	case string:
		if !matchesJSONType(value, typ) {
			return fmt.Errorf("field %q must be a %s", name, typ)
		}
	case []any:
		names := make([]string, 0, len(typ))
		for _, t := range typ {
			tn, ok := t.(string)
			if !ok {
				continue // malformed member — skip it, still honor the valid ones
			}
			if matchesJSONType(value, tn) {
				return nil
			}
			names = append(names, tn)
		}
		if len(names) > 0 {
			return fmt.Errorf("field %q must be a %s", name, strings.Join(names, " or "))
		}
	}
	return nil
}

// missingRequired returns the names listed in schema.required that are absent
// from obj.
func missingRequired(obj, schema map[string]any) []string {
	req := stringOrAnySlice(schema["required"])
	if req == nil {
		return nil
	}
	var missing []string
	for _, r := range req {
		name, ok := r.(string)
		if !ok {
			continue
		}
		if _, present := obj[name]; !present {
			missing = append(missing, name)
		}
	}
	return missing
}

// stringOrAnySlice normalizes a schema list that may be declared as []string
// (hand-written Go schemas) or []any (JSON-decoded schemas) into []any, so
// validation applies to both forms. Returns nil for anything else.
func stringOrAnySlice(v any) []any {
	switch s := v.(type) {
	case []any:
		return s
	case []string:
		out := make([]any, len(s))
		for i, x := range s {
			out[i] = x
		}
		return out
	default:
		return nil
	}
}

// isObjectSchema reports whether schema describes an object (so required/
// properties are meaningful). A schema with no explicit type but with
// properties/required is treated as an object.
func isObjectSchema(schema map[string]any) bool {
	if len(schema) == 0 {
		return false
	}
	if typ, ok := schema["type"].(string); ok {
		return typ == "object"
	}
	_, hasProps := schema["properties"]
	_, hasReq := schema["required"]
	return hasProps || hasReq
}

// matchesJSONType reports whether a value decoded from JSON matches a JSON
// Schema primitive type. JSON numbers decode to float64, so "integer" accepts a
// whole-valued float.
func matchesJSONType(value any, typ string) bool {
	switch typ {
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "number":
		_, ok := numericValue(value)
		return ok
	case "integer":
		if _, ok := value.(json.Number); ok {
			// JSON Schema supports arbitrary-precision numbers. Leave the exact
			// integral check to the compiled validator rather than rejecting a
			// valid value because it cannot fit in float64/int64 here.
			return true
		}
		f, ok := numericValue(value)
		return ok && !math.IsInf(f, 0) && math.Trunc(f) == f
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "null":
		return value == nil
	default:
		return true // unknown type constraint — don't reject
	}
}

func enumContains(enum []any, value any) bool {
	for _, e := range enum {
		if e == value {
			return true
		}
		// JSON decodes every number as float64; a hand-written schema may
		// declare integer enums as int/int64, which would never match.
		if ef, ok := numericValue(e); ok {
			if vf, ok := numericValue(value); ok && ef == vf {
				return true
			}
		}
	}
	return false
}

// numericValue reads any Go numeric type as float64 for enum comparison.
func numericValue(v any) (float64, bool) {
	switch n := v.(type) {
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case uint:
		return float64(n), true
	default:
		return 0, false
	}
}

func formatEnum(enum []any) string {
	parts := make([]string, 0, len(enum))
	for _, e := range enum {
		parts = append(parts, fmt.Sprintf("%v", e))
	}
	return strings.Join(parts, ", ")
}

// Idempotency keys give tools with external side effects (payments, emails,
// webhooks, row inserts) a dedupe handle that survives a crash-resume. The
// framework derives the key from (sessionID, toolCallID): both are already in
// the durable log, and RecoverSession replays a dangling retry-safe call with
// its original ToolCall.ID, so the SAME logical call always resolves to the
// SAME key — a re-run after a crash presents the identical key to the external
// system, which can then drop the duplicate. Runs without a durable session get
// no key: there is no replay without a log, and a key that isn't stable across
// process restarts would only invite tools to trust it.

// idempotencyKeyCtx is the context key carrying the current invocation's key.
type idempotencyKeyCtx struct{}

// toolInvocationScopeKey lets a native host distinguish provider call IDs that
// are reused in separate invocations. Nested calls inherit the same scope.
type toolInvocationScopeKey struct{}

// WithToolInvocationScope binds a tool call to an opaque durable invocation ID.
// A native session host must persist this identity before executing a tool;
// provider call IDs alone may be reused. Nested calls inherit the scope.
func WithToolInvocationScope(ctx context.Context, invocationID string) context.Context {
	return context.WithValue(ctx, toolInvocationScopeKey{}, invocationID)
}

func toolInvocationScope(ctx context.Context) string {
	id, _ := ctx.Value(toolInvocationScopeKey{}).(string)
	return id
}

// ToolInvocationScope returns the host identity for this invocation. Native
// checkpoint consumers journal effects with this scope and ToolCallID rather
// than treating identical arguments as the same intended action.
func ToolInvocationScope(ctx context.Context) (string, bool) {
	id := toolInvocationScope(ctx)
	return id, id != ""
}

// toolIdempotencyKey derives the stable key for one tool invocation. Hashing
// keeps the key fixed-length and opaque (session IDs may embed user-visible
// naming), sized for external APIs' idempotency-key fields.
func toolIdempotencyKey(sessionID, callID string) string {
	if sessionID == "" || callID == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(sessionID + "\x00" + callID))
	return "ik_" + hex.EncodeToString(sum[:16])
}

// withIdempotencyKey stamps the invocation key onto the context handed to the
// tool. A key of "" leaves the context unchanged.
func withIdempotencyKey(ctx context.Context, key string) context.Context {
	if key == "" {
		return ctx
	}
	return context.WithValue(ctx, idempotencyKeyCtx{}, key)
}

// IdempotencyKey returns the framework-generated idempotency key for the
// current tool invocation, for use inside Tool.Run. ok is false on a run
// without a durable session — a tool performing a non-idempotent external
// effect should treat that as "no crash-retry protection", not an error.
func IdempotencyKey(ctx context.Context) (key string, ok bool) {
	key, ok = ctx.Value(idempotencyKeyCtx{}).(string)
	return key, ok && key != ""
}

// toolCallIDCtx carries the provider-assigned ID of the current tool call. The
// spawn tool derives the child's deterministic session ID from it, so a
// replayed spawn call reattaches to the same child session (pi's deterministic
// child-session IDs from (parentSessionId, toolCallId)).
type toolCallIDCtx struct{}

// withToolCallID stamps the current tool call's ID onto the context handed to
// the tool. An empty id leaves the context unchanged.
func withToolCallID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, toolCallIDCtx{}, id)
}

// ToolCallID returns the provider-assigned ID of the current tool call, for
// use inside Tool.Run. ok is false when the tool was invoked outside the loop.
func ToolCallID(ctx context.Context) (id string, ok bool) {
	id, ok = ctx.Value(toolCallIDCtx{}).(string)
	return id, ok && id != ""
}
