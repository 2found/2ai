package agentcore

import (
	"encoding/json"
	"strings"
)

// This file is the AgentCore Lab read model: the single definition of "one step"
// of a run, and the pure fold that turns a run's recorded facts into ordered
// steps. The SAME fold powers a live-stepped (explain-mode) run and a replayed
// (historical) run, so a run reads identically both ways (the Lab's core
// invariant). It is built only from facts already captured by the loop trace —
// the request messages sent each turn, the assistant response, the tool calls,
// the advertised tools, and the real per-turn token/cost — so replay
// reconstructs only what was recorded and adds no new persistence.
//
// Secrets never appear: tool arguments are traced in {{cred:NAME}} placeholder
// form (resolution happens after the trace, at the boundary), so the fold sees
// placeholders, never literals. The fold MUST NOT resolve credentials.

// LabStepKind classifies one step of a run as the Lab presents it.
type LabStepKind string

const (
	// LabStepTurn is one agent turn: reason -> tool calls + results (the default
	// "one step" per the requirement's open question).
	LabStepTurn LabStepKind = "turn"
	// LabStepCompaction is a context-compaction step: the older span was summarized
	// and the recent tail kept. It is shown as its own step so a builder sees what
	// the summary kept and what was dropped.
	LabStepCompaction LabStepKind = "compaction"
)

// LabSkillRef is one skill advertised to the model in the system prompt (header
// only — id + name + description). Whether its body was actually loaded is
// tracked separately in LabStep.SkillsLoaded (the read_skill calls observed).
type LabSkillRef struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// LabToolCall is one tool invocation within a step: the model's request (name +
// placeholder-form args) paired with the result it received and the gate outcome.
type LabToolCall struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Args    string `json:"args"`   // {{cred:NAME}} placeholder form — never a literal
	Result  string `json:"result"` // the tool-result message content, if recorded
	Allowed bool   `json:"allowed"`
	Error   string `json:"error,omitempty"`
}

// LabStep is the full per-step inspector state the Lab renders: the context that
// entered the step, the assembled system prompt broken into persona / memory /
// advertised-skills, the tools available, the tool calls and their results, the
// assistant response, and both per-step and cumulative token/cost accounting
// (the real numbers from the run, never a re-estimate).
type LabStep struct {
	NativeTrace json.RawMessage `json:"native_trace,omitempty"`
	SessionKey  string          `json:"session_key,omitempty"`
	Depth       int             `json:"depth,omitempty"`
	Index       int             `json:"index"` // 0-based position in the step list
	Turn        int             `json:"turn"`  // 1-based agent turn this step belongs to
	Kind        LabStepKind     `json:"kind"`

	// Assembled context + prompt as it entered this step.
	System  string    `json:"system"`  // the full assembled system prompt
	Persona string    `json:"persona"` // the Identity (SOUL) section of the prompt
	Memory  []string  `json:"memory"`  // recalled-memory bullet lines
	Context []Message `json:"context"` // the messages sent to the model this turn

	// Skills: advertised (headers in the prompt) vs loaded (read_skill calls seen
	// up to and including this step).
	SkillsAdvertised []LabSkillRef `json:"skills_advertised"`
	SkillsLoaded     []string      `json:"skills_loaded"`

	// Tools available this turn (advertised schemas, by name) and the tool calls
	// the model made this turn paired with their results.
	Tools     []string      `json:"tools"`
	ToolCalls []LabToolCall `json:"tool_calls"`

	Response   string `json:"response"`
	StopReason string `json:"stop_reason,omitempty"`
	Error      string `json:"error,omitempty"`

	// Compaction (Kind == LabStepCompaction): the summary kept in place of the
	// dropped span. Context holds the retained tail.
	Summary string `json:"summary,omitempty"`

	// Per-step and cumulative accounting (real provider numbers).
	TokensIn     int     `json:"tokens_in"`
	TokensOut    int     `json:"tokens_out"`
	CostUSD      float64 `json:"cost_usd"`
	CumTokensIn  int     `json:"cum_tokens_in"`
	CumTokensOut int     `json:"cum_tokens_out"`
	CumCostUSD   float64 `json:"cum_cost_usd"`
}

// TurnRecord is one recorded LLM call within a run — the unit the fold consumes.
// A consumer maps its persisted trace rows (storage.AgentLLMCall) onto this
// neutral shape so the fold itself imports no storage.
type TurnRecord struct {
	NativeTrace     json.RawMessage
	SessionKey      string
	Depth           int
	Messages        []Message        // the request messages sent to the model this turn
	Response        string           // assistant text returned
	ReasoningBlocks []ReasoningBlock // opaque provider replay blocks returned with the assistant turn
	ToolCalls       []ToolCall       // the tool calls the model requested this turn
	// ToolGates is the gate outcome of each call in ToolCalls. Empty means the
	// recording predates this field (or the turn requested no tools) — FoldSteps
	// then keeps the historical default rather than inventing denials. See ToolGate.
	ToolGates  []ToolGate
	Tools      []string // advertised tool names this turn
	StopReason string
	Error      string
	TokensIn   int
	TokensOut  int
	CostUSD    float64
}

// ToolGate is the persisted gate outcome of one tool call the model requested.
// CallID matches ToolCall.ID. The LLM-call trace used to carry only the request
// (name + args), so FoldSteps had nothing to read and hardcoded Allowed: true —
// a replayed denial rendered as an allowed call, which is worse than missing
// data on a debugging surface. Empty ToolGates on a TurnRecord is the
// pre-column default; FoldSteps does not rewrite those as denied.
type ToolGate struct {
	CallID  string `json:"call_id"`
	Allowed bool   `json:"allowed"`
	Reason  string `json:"reason,omitempty"`
	Error   string `json:"error,omitempty"`
}

// ApplyLiveGates overlays in-memory dispatcher verdicts onto TurnRecords
// whose ToolGates are still empty. Live explain folds at StepGate, which
// fires before persistTrace writes tool_gates_json; FoldSteps then treats
// empty gates as the historical Allowed:true default and a live denial
// renders as allowed until the run ends. This is the consumer mapping
// that closes that gap without a DB round-trip and without importing
// storage into FoldSteps (TurnRecord exists for that reason).
//
// Records that already carry gates (replay, or a persistTrace that has
// landed) are left alone — live overlay must not rewrite persisted truth.
// A crash mid-run therefore degrades to today's empty-gates view rather
// than inventing denials. Pure: same inputs → same records.
//
// The join key is ToolCall.ID / ToolTrace.CallID. Traces without a CallID
// cannot be paired and are skipped.
func ApplyLiveGates(records []TurnRecord, traces []ToolTrace) []TurnRecord {
	if len(records) == 0 || len(traces) == 0 {
		return records
	}
	byID := make(map[string]ToolGate, len(traces))
	for _, t := range traces {
		if t.CallID == "" {
			continue
		}
		byID[t.CallID] = ToolGate{CallID: t.CallID, Allowed: t.Allowed, Reason: t.Reason, Error: t.Error}
	}
	if len(byID) == 0 {
		return records
	}
	out := make([]TurnRecord, len(records))
	copy(out, records)
	for i := range out {
		if len(out[i].ToolGates) > 0 || (len(out[i].NativeTrace) > 0 && string(out[i].NativeTrace) != "null") {
			continue
		}
		gates := make([]ToolGate, 0, len(out[i].ToolCalls))
		for _, c := range out[i].ToolCalls {
			g, ok := byID[c.ID]
			if !ok {
				continue
			}
			gates = append(gates, g)
		}
		if len(gates) > 0 {
			out[i].ToolGates = gates
		}
	}
	return out
}

// FoldSteps turns a run's ordered turn records into the Lab's step list. It
// detects a compaction by the summary message the loop folds into the history
// (a system message prefixed with summaryMarker that first appears on a given
// turn) and emits it as its own step before that turn. Everything else is one
// turn = one step. Pure: same input -> same steps, for live and replay alike.
func FoldSteps(records []TurnRecord) []LabStep {
	// The immediately following request in the same session contains this
	// turn's results. Sessions and repeated provider call IDs are independent;
	// a run-wide ID map can attach a child's result to its parent's call.
	results := make([]map[string]string, len(records))
	nativeGates := make([][]ToolGate, len(records))
	nextNative := map[string][]ToolGate{}
	next := map[string]map[string]string{}
	for i := len(records) - 1; i >= 0; i-- {
		r := records[i]
		results[i] = next[r.SessionKey]
		nativeGates[i] = nextNative[r.SessionKey]
		nextNative[r.SessionKey] = nativeTraceGates(r.NativeTrace)
		current := map[string]string{}
		for _, message := range r.Messages {
			if message.Role == RoleTool && message.ToolCallID != "" {
				current[message.ToolCallID] = message.Content
			}
		}
		next[r.SessionKey] = current
	}

	steps := make([]LabStep, 0, len(records)+1)
	loadedBySession := map[string][]string{}
	seenBySession := map[string]map[string]bool{}
	lastSummary := map[string]string{}
	var cumIn, cumOut int
	var cumCost float64

	for i, r := range records {
		turn := i + 1
		loaded := loadedBySession[r.SessionKey]
		loadedSeen := seenBySession[r.SessionKey]
		if loadedSeen == nil {
			loadedSeen = map[string]bool{}
			seenBySession[r.SessionKey] = loadedSeen
		}

		// Compaction detection: the most recent summary message in this turn's
		// context, if it differs from the last one we saw, means a compaction
		// happened just before this turn.
		summary := latestSummary(r.Messages)
		if summary != "" && summary != lastSummary[r.SessionKey] {
			steps = append(steps, LabStep{
				SessionKey: r.SessionKey, Depth: r.Depth,
				Index:        len(steps),
				Turn:         turn,
				Kind:         LabStepCompaction,
				Summary:      summary,
				Context:      r.Messages,
				CumTokensIn:  cumIn,
				CumTokensOut: cumOut,
				CumCostUSD:   cumCost,
			})
			lastSummary[r.SessionKey] = summary
		}

		// Accumulate read_skill loads visible at this step.
		for _, c := range r.ToolCalls {
			if c.Name == readSkillToolName {
				if id := skillIDFromArgs(c.Arguments); id != "" && !loadedSeen[id] {
					loadedSeen[id] = true
					loaded = append(loaded, id)
				}
			}
		}

		loadedBySession[r.SessionKey] = loaded
		system := systemPrompt(r.Messages)
		persona, memory, advertised := parseSystemPrompt(system)

		calls := make([]LabToolCall, 0, len(r.ToolCalls))
		for _, c := range r.ToolCalls {
			gates := r.ToolGates
			if len(r.NativeTrace) > 0 && string(r.NativeTrace) != "null" {
				gates = nativeGates[i]
			}
			allowed, errStr := gateOutcome(gates, c.ID)
			calls = append(calls, LabToolCall{
				ID:      c.ID,
				Name:    c.Name,
				Args:    c.Arguments,
				Result:  results[i][c.ID],
				Allowed: allowed,
				Error:   errStr,
			})
		}

		cumIn += r.TokensIn
		cumOut += r.TokensOut
		cumCost += r.CostUSD

		steps = append(steps, LabStep{
			NativeTrace: append(json.RawMessage(nil), r.NativeTrace...),
			SessionKey:  r.SessionKey, Depth: r.Depth,
			Index:            len(steps),
			Turn:             turn,
			Kind:             LabStepTurn,
			System:           system,
			Persona:          persona,
			Memory:           memory,
			Context:          r.Messages,
			SkillsAdvertised: advertised,
			SkillsLoaded:     append([]string{}, loaded...),
			Tools:            r.Tools,
			ToolCalls:        calls,
			Response:         r.Response,
			StopReason:       r.StopReason,
			Error:            r.Error,
			TokensIn:         r.TokensIn,
			TokensOut:        r.TokensOut,
			CostUSD:          r.CostUSD,
			CumTokensIn:      cumIn,
			CumTokensOut:     cumOut,
			CumCostUSD:       cumCost,
		})
	}
	return steps
}

// Native tool results carry their governed audit receipt. Match within the
// next request of the same session, rather than overlaying a run-wide call-ID
// map that can confuse siblings or repeated provider IDs.
func nativeTraceGates(raw json.RawMessage) []ToolGate {
	var trace struct {
		Context struct{ Messages []json.RawMessage }
	}
	if json.Unmarshal(raw, &trace) != nil {
		return nil
	}
	byID := map[string]ToolGate{}
	for _, raw := range trace.Context.Messages {
		var message struct {
			Role, ToolCallID string
			Details          PiToolOutcome
		}
		if json.Unmarshal(raw, &message) != nil || message.Role != "toolResult" || message.ToolCallID == "" {
			continue
		}
		audit := message.Details.Trace
		if audit.CallID != message.ToolCallID {
			continue
		}
		byID[audit.CallID] = ToolGate{CallID: audit.CallID, Allowed: audit.Allowed, Reason: audit.Reason, Error: audit.Error}
	}
	out := make([]ToolGate, 0, len(byID))
	for _, gate := range byID {
		out = append(out, gate)
	}
	return out
}

// gateOutcome looks up the recorded gate for one tool call. No gates on the
// record is the historical encoding (Allowed: true) — we must not rewrite
// those rows as denials. A matching gate is the live/replayed truth.
func gateOutcome(gates []ToolGate, callID string) (allowed bool, errStr string) {
	if len(gates) == 0 {
		return true, ""
	}
	for _, g := range gates {
		if g.CallID != callID {
			continue
		}
		errStr = g.Error
		if errStr == "" {
			errStr = g.Reason
		}
		return g.Allowed, errStr
	}
	return true, ""
}

// systemPrompt returns the leading system message's content, the assembled
// system prompt for that turn (empty when the turn carries none).
func systemPrompt(messages []Message) string {
	for _, m := range messages {
		if m.Role == RoleSystem && !strings.HasPrefix(m.Content, summaryMarker) {
			return m.Content
		}
	}
	return ""
}

// latestSummary returns the content (without the marker) of the last compaction
// summary message in a turn's context, or "" when none is present.
func latestSummary(messages []Message) string {
	out := ""
	for _, m := range messages {
		if m.Role != RoleSystem {
			continue
		}
		// A real summary (preferred) or, when summarization degraded, the elide
		// breadcrumb — either marks that a compaction happened before this turn.
		if strings.HasPrefix(m.Content, summaryMarker) {
			out = strings.TrimSpace(strings.TrimPrefix(m.Content, summaryMarker))
		} else if strings.HasPrefix(m.Content, elideMarker) {
			out = strings.TrimSpace(m.Content)
		}
	}
	return out
}

// parseSystemPrompt splits an assembled system prompt into its persona (Identity
// section), recalled-memory bullets, and advertised-skill headers. It mirrors the
// exact section markers buildSystemPrompt writes, so the Lab shows the same parts
// the loop assembled — no separate persistence needed.
func parseSystemPrompt(system string) (persona string, memory []string, skills []LabSkillRef) {
	if system == "" {
		return "", nil, nil
	}
	sections := splitSections(system)
	persona = strings.TrimSpace(sections["# Identity"])
	for _, line := range strings.Split(sections["# Recalled memory"], "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "- ") {
			memory = append(memory, strings.TrimSpace(strings.TrimPrefix(line, "- ")))
		}
	}
	for _, line := range strings.Split(sections["# Available skills"], "\n") {
		line = strings.TrimSpace(line)
		if ref, ok := parseSkillLine(line); ok {
			skills = append(skills, ref)
		}
	}
	return persona, memory, skills
}

// splitSections breaks a prompt into a map keyed by its "# Header" lines.
func splitSections(system string) map[string]string {
	out := map[string]string{}
	var key string
	var b strings.Builder
	flush := func() {
		if key != "" {
			out[key] = strings.TrimRight(b.String(), "\n")
		}
		b.Reset()
	}
	for _, line := range strings.Split(system, "\n") {
		if strings.HasPrefix(line, "# ") {
			flush()
			key = strings.TrimRight(line, " ")
			continue
		}
		b.WriteString(line)
		b.WriteString("\n")
	}
	flush()
	return out
}

// parseSkillLine parses one advertised-skill bullet of the form
// "- id: <id> — <name>: <desc>" written by buildSystemPrompt.
func parseSkillLine(line string) (LabSkillRef, bool) {
	if !strings.HasPrefix(line, "- id: ") {
		return LabSkillRef{}, false
	}
	rest := strings.TrimPrefix(line, "- id: ")
	idPart, after, ok := strings.Cut(rest, " — ")
	if !ok {
		return LabSkillRef{ID: strings.TrimSpace(rest)}, true
	}
	name, desc, _ := strings.Cut(after, ": ")
	return LabSkillRef{
		ID:          strings.TrimSpace(idPart),
		Name:        strings.TrimSpace(name),
		Description: strings.TrimSpace(desc),
	}, true
}

// skillIDFromArgs pulls the "id" field from a read_skill call's JSON arguments.
func skillIDFromArgs(args string) string {
	var a struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal([]byte(args), &a)
	return strings.TrimSpace(a.ID)
}

// A long run's table of contents.
//
// A run of a few dozen steps can be read by scrolling. A run of several
// thousand cannot, and paging it only turns "too much to read" into "too much
// to read, twelve at a time" — the reader still has no idea where in the task
// they are. What is missing is not a smaller page but a STRUCTURE, and the run
// already produced one without being asked to.
//
// Compaction is that structure. Every time the window fills, the loop asks the
// model to summarize everything that happened since the last time it asked, and
// keeps the answer. A run that compacted 353 times therefore contains 353
// model-written accounts of "what I had done by this point", in order, each
// bounding a span of turns. That is a chapter list, and it costs nothing to
// produce because it has already been paid for.
//
// So the Lab's navigation is: chapters (always whole — they are small, and they
// are how you find the part you want) plus one page of steps (large, and only
// useful once you know which part you want). The first chapter of a run that
// never compacted is the whole run, which is exactly right for the short runs
// where none of this matters.

// RunChapter is one span of a run bounded by compaction: the work between one
// context rewrite and the next, and the model's own account of it.
type RunChapter struct {
	// Index is the chapter's 0-based position.
	Index int `json:"index"`
	// FirstStep and LastStep are inclusive indices into the step list, so a
	// client turns a chapter into a page request without a second lookup.
	FirstStep int `json:"first_step"`
	LastStep  int `json:"last_step"`
	// FirstTurn and LastTurn are the run turns the chapter spans.
	FirstTurn int `json:"first_turn"`
	LastTurn  int `json:"last_turn"`
	// Title is a one-line label drawn from the closing summary — the model's own
	// words, not a generated paraphrase.
	Title string `json:"title"`
	// Summary is the compaction summary that CLOSED this chapter: what the agent
	// had accomplished by the end of the span. The final chapter has none (the
	// run ended before the next compaction), which is the honest answer — its
	// outcome is the run's final answer, not a summary.
	Summary string `json:"summary,omitempty"`
	// Steps is how many steps the chapter contains, and ToolCalls how many tool
	// calls were made in it — the two numbers that say whether a chapter is
	// worth opening.
	Steps     int `json:"steps"`
	ToolCalls int `json:"tool_calls"`
	// TokensIn/TokensOut/CostUSD are the chapter's own spend, differenced from
	// the cumulative totals the fold already computed.
	TokensIn  int     `json:"tokens_in"`
	TokensOut int     `json:"tokens_out"`
	CostUSD   float64 `json:"cost_usd"`
}

// RunChapters divides a folded step list into chapters at its compaction steps.
//
// A compaction step CLOSES the chapter it appears in rather than opening the
// next one, because its summary describes the span behind it. The result is
// always at least one chapter for a non-empty run.
func RunChapters(steps []LabStep) []RunChapter {
	if len(steps) == 0 {
		return []RunChapter{}
	}
	out := []RunChapter{}
	start := 0
	var baseIn, baseOut int
	var baseCost float64

	closeChapter := func(end int, summary string) {
		span := steps[start : end+1]
		last := span[len(span)-1]
		calls := 0
		for _, s := range span {
			calls += len(s.ToolCalls)
		}
		out = append(out, RunChapter{
			Index:     len(out),
			FirstStep: start,
			LastStep:  end,
			FirstTurn: span[0].Turn,
			LastTurn:  last.Turn,
			Title:     chapterTitle(summary, len(out)),
			Summary:   summary,
			Steps:     len(span),
			ToolCalls: calls,
			TokensIn:  last.CumTokensIn - baseIn,
			TokensOut: last.CumTokensOut - baseOut,
			CostUSD:   last.CumCostUSD - baseCost,
		})
		baseIn, baseOut, baseCost = last.CumTokensIn, last.CumTokensOut, last.CumCostUSD
		start = end + 1
	}

	for i, s := range steps {
		if s.Kind == LabStepCompaction {
			closeChapter(i, s.Summary)
		}
	}
	// Whatever follows the last compaction is the run's final, unsummarized
	// chapter. A run that never compacted has only this one.
	if start < len(steps) {
		closeChapter(len(steps)-1, "")
	}
	return out
}

// chapterTitle draws a one-line label out of a compaction summary.
//
// The summaries follow the checkpoint format the loop asks for ("## Goal",
// "## Progress", …), so the first line under a heading is a real sentence about
// the work rather than a heading itself. A summary that does not follow the
// format falls back to its first non-empty line, and a chapter with no summary
// (the run's last) is labelled by position.
func chapterTitle(summary string, index int) string {
	for _, line := range strings.Split(summary, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimLeft(line, "-*[]x ")
		if line = strings.TrimSpace(line); line != "" {
			return TruncateBytes(line, 120)
		}
	}
	if index == 0 {
		return "The run"
	}
	return "Final stretch"
}
