package agentcore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	nativehost "github.com/2found/2ai/agentcore/host"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
	"github.com/2found/2ai/telemetry"
)

const nativeCheckpointRevision = "eeac84ca92498ac18b6832754d01aef1d3c5f654"

// NativeRun supplies an opaque checkpoint and newly authored host input. State
// must come from RunResult.NativeState, never the display Messages projection.
// Consumers persist the checkpoint with their own successful-turn transaction.
type NativeRun struct {
	// Commands are explicit host controls, keyed by installed extension name.
	Commands map[string]json.RawMessage
	// ControlOnly applies commands and checkpoints without making a model request.
	ControlOnly bool
	State       json.RawMessage
	Input       []Message
	Task        string
	Sink        StreamSink
	// Lifecycle receives bounded setup steps that happen before the provider
	// loop starts, such as recalling relevant memory.
	Lifecycle  func(string)
	Compaction *nativehost.CompactionPolicy
	// Compact, when set, summarizes the conversation before the run continues:
	// the host's "/compact <focus>". With ControlOnly it only compacts.
	Compact   *ManualCompaction
	Telemetry telemetry.Context
	// SafeToolLabels binds host-approved public tool IDs without reading args,
	// results or private span attributes. Unbound tools have unknown labels.
	SafeToolLabels map[string]telemetry.SafeLabels
	// Checkpoint, when set, receives a resumable NativeState after every
	// settled turn, so a crash mid-run loses at most the turn in flight. It
	// runs on the loop: keep it quick. An error stops the run. Finalizers have
	// not run on these states; the final one is RunResult.NativeState.
	Checkpoint func(ctx context.Context, state json.RawMessage) error
	// Fallback, when set, is the host's fallback ladder shared across runs:
	// its rung, not the checkpoint's model, chooses where requests start. Nil
	// uses a ladder private to this run, resumed on the checkpoint's model.
	// Either way a run that fell back probes the rungs above it again (see
	// ai.FallbackLadder), so it returns to the primary once it recovers.
	Fallback *ai.FallbackLadder
}

// ManualCompaction is a host-requested compaction.
type ManualCompaction struct {
	// Instructions focus this summary, after the standing
	// CompactionSettings.Instructions.
	Instructions string
}

type nativeQuestion struct {
	CallID string          `json:"callId"`
	Args   json.RawMessage `json:"args"`
}

type nativeCheckpoint struct {
	Question      *nativeQuestion            `json:"question,omitempty"`
	Revision      string                     `json:"revision"`
	Model         json.RawMessage            `json:"model"`
	Messages      json.RawMessage            `json:"messages"`
	Summary       json.RawMessage            `json:"summary,omitempty"`
	Cleared       json.RawMessage            `json:"cleared,omitempty"`
	Goal          string                     `json:"goal,omitempty"`
	GoalRevisions []GoalRevision             `json:"goalRevisions,omitempty"`
	Plugins       map[string]json.RawMessage `json:"plugins,omitempty"`
}

// RunNative connects composed plugins/tools to the native engine. The engine
// owns all turn/tool scheduling; AI owns retry and fallback. This entry point
// uses consumer-owned checkpoints. The server's durable session host additionally
// journals in-flight effects, leases and parked delegations.
func (a *Agent) RunNative(ctx context.Context, input NativeRun) (RunResult, error) {
	if input.Telemetry.IsZero() {
		input.Telemetry = telemetry.FromContext(ctx)
	}
	input.Telemetry = telemetry.SafeContext(input.Telemetry, ctx)
	return telemetry.StartSpan(input.Telemetry, telemetry.SpanOptions{Name: "agentray.agent.run"}, func(span *telemetry.Span) (RunResult, error) {
		input.Telemetry = span.Context()
		result, err := a.runNative(telemetry.WithContext(ctx, span.Context()), input)
		if err == nil || result.StopReason == "aborted" {
			if result.StopReason == "aborted" {
				kind := "request_cancelled"
				if ctx.Err() == context.DeadlineExceeded {
					kind = "request_timeout"
				}
				span.SetSafeOutcome("cancelled", kind)
			} else if result.StopReason == "error" {
				span.SetSafeOutcome("failed", "unknown")
			} else if result.StopReason != "" && result.StopReason != "stop" && result.StopReason != "end_turn" {
				span.SetSafeOutcome("stopped", "none")
			}
		}
		span.SetAttributes(telemetry.NewAttributes(
			telemetry.Property{Name: "agent.stop_reason", Value: result.StopReason},
			telemetry.Property{Name: "agent.turns", Value: result.Turns},
			telemetry.Property{Name: "usage.input_tokens", Value: result.Usage.InputTokens},
			telemetry.Property{Name: "usage.output_tokens", Value: result.Usage.OutputTokens},
			telemetry.Property{Name: "usage.cost_usd", Value: result.Usage.CostUSD},
			telemetry.Property{Name: "agent.compactions", Value: len(result.Compactions)},
		))
		cost := result.Usage.CostUSD
		telemetry.ReconcileSafe(telemetry.WithContext(ctx, span.Context()), telemetry.SafeAccounting{Tokens: &telemetry.SafeTokens{Input: float64(result.Usage.InputTokens), Output: float64(result.Usage.OutputTokens), CacheRead: float64(result.Usage.CacheReadTokens), CacheWrite: float64(result.Usage.CacheWriteTokens)}, KnownCostUSD: &cost})
		return result, err
	})
}

func (a *Agent) runNative(ctx context.Context, input NativeRun) (result RunResult, failure error) {
	if a.nativeProvider == nil || len(a.nativeProvider.Candidates) == 0 {
		return result, errors.New("agentcore: native provider is required")
	}
	if a.session != nil || a.sessionID != "" {
		return result, errors.New("agentcore: durable execution requires the native session host")
	}
	if a.prepareNextTurn != nil {
		return result, errors.New("agentcore: native execution requires native turn preparation")
	}
	provider := *a.nativeProvider
	provider.Retry = a.retry
	// summarizer answers the standalone summary call: the compaction model
	// when one is configured, otherwise the run's own models from the one the
	// conversation settled on.
	summarizer := provider
	if a.compactionModel != nil {
		summarizer = *a.compactionModel
		summarizer.Retry = a.retry
	}
	for _, candidate := range slices.Concat(provider.Candidates, summarizer.Candidates) {
		if candidate.Stream == nil || !json.Valid(candidate.Model) {
			return result, errors.New("agentcore: native candidate requires a model and stream")
		}
	}
	checkpoint := nativeCheckpoint{Revision: nativeCheckpointRevision, Messages: json.RawMessage(`[]`)}
	selected := 0
	if len(input.State) > 0 {
		if err := json.Unmarshal(input.State, &checkpoint); err != nil {
			return result, err
		}
		if checkpoint.Revision != nativeCheckpointRevision {
			return result, errors.New("agentcore: unsupported native checkpoint revision")
		}
		if err := nativehost.ValidateMessages(checkpoint.Messages); err != nil {
			return result, err
		}
		// Resume on the candidate the conversation last settled on. When the
		// host has since replaced that model, the conversation continues on the
		// current primary: history is provider-neutral, like a fallback switch.
		for i, candidate := range provider.Candidates {
			if nativehost.SameJSON(checkpoint.Model, candidate.Model) {
				selected = i
				break
			}
		}
	}
	ladder := input.Fallback
	if ladder == nil {
		ladder = &ai.FallbackLadder{}
		ladder.Resume(selected)
	} else {
		selected = min(ladder.Rung(), len(provider.Candidates)-1)
	}
	var history []*ai.Message
	if err := json.Unmarshal(checkpoint.Messages, &history); err != nil {
		return result, err
	}
	// sentSystem is the system prompt the transcript carries; a new system
	// message is appended only when the prompt really changes.
	sentSystem := nativeSystemText(history)
	// settled is the transcript as the engine appends it, each message encoded
	// once: a request and a checkpoint join bytes instead of re-encoding the
	// whole history every turn.
	var settled []json.RawMessage
	encoded := make(map[*ai.Message]json.RawMessage, len(history))
	for _, message := range history {
		raw, err := json.Marshal(message)
		if err != nil {
			return result, err
		}
		encoded[message] = raw
	}
	encode := func(messages []*ai.Message) (json.RawMessage, error) {
		parts := make([]json.RawMessage, len(messages))
		for i, message := range messages {
			if parts[i] = encoded[message]; parts[i] == nil {
				raw, err := json.Marshal(message)
				if err != nil {
					return nil, err
				}
				parts[i] = raw
			}
		}
		return joinRawMessages(parts), nil
	}
	rawInput, err := nativehost.InputMessages(input.Input)
	if err != nil {
		return result, err
	}
	goal := checkpoint.Goal
	if goal == "" {
		goal = a.goal
	}
	answerQuestion := false
	if checkpoint.Question != nil && !input.ControlOnly {
		if len(input.Input) == 0 {
			return RunResult{Parked: true, StopReason: "parked", Question: checkpoint.Question.Args, NativeState: append(json.RawMessage(nil), input.State...), NativeRevision: nativeCheckpointRevision}, nil
		}
		if len(input.Input) != 1 || input.Input[0].Role != RoleUser {
			return result, errors.New("parked question requires one new user answer")
		}
		// The native receipt is immutable. The human answer arrives as a new
		// user message, matching the native session host's question protocol.
		answerQuestion = true
	}
	var sinkMu sync.Mutex
	progressOpen := true
	emit := func(event StreamEvent) {
		sinkMu.Lock()
		defer sinkMu.Unlock()
		if progressOpen && input.Sink != nil {
			input.Sink(event)
		}
	}
	ctx = WithRunProgress(ctx, func(note string) { emit(StreamEvent{Type: StreamProgress, Note: note}) })
	defer func() {
		sinkMu.Lock()
		progressOpen = false
		sinkMu.Unlock()
	}()

	h, err := a.openTools(ctx, false, goal)
	if err != nil {
		return result, err
	}
	defer h.Close()
	defer func() { result = h.CompleteRun(context.WithoutCancel(ctx), result) }()
	ctx = WithGoalRevisionRecorder(h.ctx, func(_ context.Context, revision GoalRevision) error {
		checkpoint.GoalRevisions = append(checkpoint.GoalRevisions, revision)
		return nil
	})
	for _, ext := range h.exts.all {
		if state, ok := ext.(NativeStateContributor); ok && len(checkpoint.Plugins[ext.Name()]) > 0 {
			if err := state.RestoreNativeState(checkpoint.Plugins[ext.Name()]); err != nil {
				return result, fmt.Errorf("restore %s: %w", ext.Name(), err)
			}
		}
	}

	result.CommandResults = make(map[string]json.RawMessage)
	commandNames := make([]string, 0, len(input.Commands))
	for name := range input.Commands {
		commandNames = append(commandNames, name)
	}
	slices.Sort(commandNames)
	for _, name := range commandNames {
		command := input.Commands[name]
		var handler RunCommandHandler
		for _, ext := range h.exts.all {
			if ext.Name() == name {
				handler, _ = ext.(RunCommandHandler)
				break
			}
		}
		if handler == nil {
			return result, fmt.Errorf("extension %q does not accept run commands", name)
		}
		raw, err := handler.HandleRunCommand(ctx, command)
		if err != nil {
			return result, fmt.Errorf("command %s: %w", name, err)
		}
		result.CommandResults[name] = raw
	}

	system, recalled, err := h.startRun(ctx, input.Task, input.Lifecycle, false)
	if err != nil {
		return result, err
	}
	if len(a.hooks.BeforeAgentStart) > 0 {
		start, err := a.hooks.runBeforeAgentStart(ctx, RunStart{System: system, Messages: slices.Clone(input.Input), Task: input.Task})
		if err != nil {
			return result, err
		}
		system = start.System
		if rawInput, err = nativehost.InputMessages(start.Messages); err != nil {
			return result, fmt.Errorf("before_agent_start: %w", err)
		}
	}
	external, err := json.Marshal(rawInput)
	if err != nil {
		return result, err
	}
	if err := h.ObserveMessages(ctx, PhaseRestore, 0, checkpoint.Messages); err != nil {
		return result, err
	}
	if err := h.ObserveMessages(ctx, PhaseExternalInput, 0, external); err != nil {
		return result, err
	}
	tools, err := h.nativeTools(ctx, input.Telemetry, input.SafeToolLabels, emit)
	if err != nil {
		return result, err
	}
	account := func(outcome ai.AttemptOutcome) {
		if message := outcome.Message(); message != nil {
			raw, err := json.Marshal(message)
			if err != nil {
				return
			}
			projected, err := ai.ProjectMessage(raw)
			if err == nil && projected.Usage != nil {
				result.Usage = result.Usage.Add(*projected.Usage)
			}
		}
	}
	budget := effectiveBudget(a.limits.MaxContextTokens, a.contextWindow)
	policy := nativehost.CompactionPolicy{Budget: budget, KeepRecent: a.compaction.keepRecent(budget), Clearing: a.compaction.ClearToolResults}
	if input.Compaction != nil {
		policy = *input.Compaction
	}
	policy.Task = input.Task
	// A skill's instructions stay in force for the whole run.
	policy.Clearing.ExcludeTools = append(slices.Clone(policy.Clearing.ExcludeTools), readSkillToolName)
	if policy.Clearing.CacheTTL == 0 {
		switch a.cacheRetention {
		case "long":
			policy.Clearing.CacheTTL = time.Hour
		case "none":
			policy.Clearing.CacheTTL = time.Nanosecond
		}
	}
	if policy.Reattach == nil {
		policy.Reattach = h.reattach
	}
	if policy.Archive == nil {
		for _, ext := range h.exts.all {
			if archive, ok := ext.(ToolResultArchiver); ok {
				policy.Archive = func(ctx context.Context, name, id, text string) (string, error) {
					return archive.ArchiveToolResult(ctx, ToolCall{ID: id, Name: name}, text)
				}
				break
			}
		}
	}
	if policy.Instructions == "" {
		policy.Instructions = a.compaction.Instructions
	}
	// shared is the request the loop is about to send, bound by AdmitRequest
	// before each compaction check. Admission is serialized, so one slot is enough.
	var shared struct {
		index    int
		controls map[string]any
		ok       bool
	}
	if policy.Summarize == nil {
		// summaryCall runs one summary request and returns the model's reply text.
		summaryCall := func(ctx context.Context, provider *ai.FallbackProvider, mode string, start, limit int, open func(index int) (ai.TranscriptContext, map[string]any)) (string, Usage, error) {
			emit(StreamEvent{Type: StreamProgress, Note: "Compacting context"})
			defer func() {
				emit(StreamEvent{Type: StreamProgress, Note: "Context summary chunk settled; compaction may continue"})
			}()
			var usage Usage
			out := ai.NewAssistantMessageEventStream()
			final, err := telemetry.StartSpan(input.Telemetry, telemetry.SpanOptions{Name: "agentray.ai.compaction", Attributes: telemetry.NewAttributes(telemetry.Property{Name: "compaction.mode", Value: mode})}, func(span *telemetry.Span) (ai.AttemptOutcome, error) {
				ctx = telemetry.WithContext(ctx, span.Context())
				trace := ai.NewAttemptTrace(span)
				return provider.Run(ctx, out, ai.FallbackRequest{Start: start, Candidates: limit,
					Open: func(ctx context.Context, index, _ int) (*ai.AssistantMessageEventStream, error) {
						candidate := provider.Candidates[index]
						transcript, controls := open(index)
						trace.Start(candidate.Model, transcript)
						return candidate.Stream(ctx, slices.Clone(candidate.Model), transcript, controls)
					},
					Observe: func(ctx context.Context, _ int, attempt ai.FallbackAttempt) error {
						trace.Finish(attempt)
						if message := attempt.Outcome.Message(); message != nil {
							raw, _ := json.Marshal(message)
							projected, err := ai.ProjectMessage(raw)
							if err != nil {
								return err
							}
							if projected.Usage != nil {
								usage = usage.Add(*projected.Usage)
							}
						}
						return nil
					},
				})
			})
			out.End()
			if err != nil {
				return "", usage, err
			}
			message := final.Message()
			if message == nil || message.StopReason == "error" || message.StopReason == "aborted" {
				return "", usage, errors.New("native compaction failed")
			}
			raw, _ := json.Marshal(message)
			projected, err := ai.ProjectMessage(raw)
			if err == nil && (len(projected.ToolCalls) > 0 || message.StopReason == "toolUse" || message.StopReason == "length") {
				return "", usage, errors.New("native compaction did not produce a complete text summary")
			}
			return projected.Content, usage, err
		}
		// allowance sizes the handoff against actual input capacity. A low
		// trigger must not erase critical facts merely to fit below an
		// immutable tool head.
		allowance := func(prefix json.RawMessage) (int, error) {
			effectiveBudget := policy.Budget
			inputCeiling := policy.MaxInputTokens
			if effectiveBudget <= 0 {
				effectiveBudget = 64 * 1024
			}
			for _, candidate := range provider.Candidates {
				var model struct{ ContextWindow int }
				_ = json.Unmarshal(candidate.Model, &model)
				if candidateBudget := policy.ForWindow(model.ContextWindow).Budget; candidateBudget > 0 {
					effectiveBudget = min(effectiveBudget, candidateBudget)
				}
				if ceiling := policy.ForWindow(model.ContextWindow).MaxInputTokens; ceiling > 0 && (inputCeiling <= 0 || ceiling < inputCeiling) {
					inputCeiling = ceiling
				}
			}
			summaryBudget := effectiveBudget
			if inputCeiling > 0 {
				summaryBudget = inputCeiling
			}
			return nativehost.SummaryAllowance(prefix, policy.Task, summaryBudget)
		}
		inputBudget := 64 * 1024 // conservative input ceiling for an unknown window
		for _, candidate := range summarizer.Candidates {
			var model struct{ ContextWindow int }
			_ = json.Unmarshal(candidate.Model, &model)
			if model.ContextWindow > 0 {
				inputBudget = min(inputBudget, max(256, model.ContextWindow-min(4096, max(256, model.ContextWindow/4))-1024))
			}
		}
		// The standalone summarizer reads a text rendering of the prefix, not its
		// native JSON: signatures, usage records and inline image bytes cost
		// tokens and tell a summarizer nothing. A prefix too long for one call
		// is folded on message boundaries.
		policy.Summarize = func(ctx context.Context, prefix json.RawMessage, focus string) (string, Usage, error) {
			summaryTokens, err := allowance(prefix)
			if err != nil {
				return "", Usage{}, err
			}
			var messages []json.RawMessage
			if err := json.Unmarshal(prefix, &messages); err != nil {
				return "", Usage{}, err
			}
			instructions := nativehost.SummaryInstructions(summaryTokens, focus, false)
			// The scratchpad is discarded, so it gets its own output room.
			controls := map[string]any{"maxTokens": summaryTokens + min(summaryTokens, 4096), "reasoning": "off", "toolChoice": "none", "cacheRetention": "none"}
			reserve := (len(nativehost.SummarySystemPrompt)+len(instructions))/4 + summaryTokens + 256
			return nativehost.FoldRendered(ctx, nativehost.RenderTranscript(messages), max(256, inputBudget-reserve), func(ctx context.Context, previous, segment string) (string, Usage, error) {
				content := "<conversation>\n" + segment + "\n</conversation>\n\n" + instructions
				if previous != "" {
					content = "<summary_so_far>\n" + previous + "\n</summary_so_far>\n\nThe conversation continues below. Fold it into the summary so far and write one updated handoff.\n\n" + content
				}
				start := selected
				if a.compactionModel != nil {
					start = 0
				}
				reply, usage, err := summaryCall(ctx, &summarizer, "standalone", start, len(summarizer.Candidates), func(int) (ai.TranscriptContext, map[string]any) {
					return ai.NormalizeContext(ai.Context{SystemPrompt: nativehost.SummarySystemPrompt, Messages: []ai.Message{{Role: "user", Content: ai.TextContent(content)}}}), maps.Clone(controls)
				})
				if err != nil {
					return "", usage, err
				}
				return nativehost.ClampSummary(nativehost.ExtractSummary(reply), summaryTokens), usage, nil
			})
		}
		// With SharePromptCache the summary is first asked for as one more turn
		// of the live conversation: the provider reads the whole transcript from
		// its prompt cache, and one call replaces a chain of folds. A forced
		// tool choice or an output schema would shape the reply, so those runs
		// keep the standalone summarizer.
		if a.compaction.SharePromptCache && policy.SummarizeView == nil && a.toolChoice.Mode == ToolChoiceDefault && a.outputSchema == nil {
			policy.SummarizeView = func(ctx context.Context, request nativehost.SummaryRequest) (string, Usage, error) {
				if !shared.ok {
					return "", Usage{}, errors.New("no live request to share")
				}
				index, controls := shared.index, shared.controls
				var model struct{ ContextWindow int }
				_ = json.Unmarshal(provider.Candidates[index].Model, &model)
				if ceiling := policy.ForWindow(model.ContextWindow).MaxInputTokens; ceiling > 0 && nativehost.ContextTokens(request.View)+1024 > ceiling {
					return "", Usage{}, errors.New("the live request does not leave room for a summary instruction")
				}
				var raws []json.RawMessage
				if err := json.Unmarshal(request.View, &raws); err != nil || request.Cut <= 0 || request.Cut > len(raws) {
					return "", Usage{}, errors.New("invalid summary view")
				}
				prefix, _ := json.Marshal(raws[:request.Cut])
				summaryTokens, err := allowance(prefix)
				if err != nil {
					return "", Usage{}, err
				}
				var messages []ai.Message
				if err := json.Unmarshal(request.View, &messages); err != nil {
					return "", Usage{}, err
				}
				visible := messages[:0]
				for _, message := range messages {
					switch message.Role {
					case "system", "user", "assistant", "toolResult":
						visible = append(visible, message)
					}
				}
				visible = append(visible, ai.Message{Role: "user", Content: ai.TextContent(nativehost.SummaryInstructions(summaryTokens, request.Instructions, true)), Timestamp: time.Now().UnixMilli()})
				transcript := ai.NormalizeContext(ai.Context{Messages: visible})
				// Everything that keys the provider's cache stays as the live
				// request sent it: tools, tool choice, reasoning and the
				// cleared view. Only the output allowance grows, so the
				// scratchpad and the handoff both fit.
				controls = maps.Clone(controls)
				if want := summaryTokens + min(summaryTokens, 4096); controlInt(controls["maxTokens"]) < want {
					controls["maxTokens"] = want
				}
				reply, usage, err := summaryCall(ctx, &provider, "shared", index, index+1, func(int) (ai.TranscriptContext, map[string]any) {
					return transcript, maps.Clone(controls)
				})
				if err != nil {
					return "", usage, err
				}
				// A reply without the block is the model carrying on with the
				// task, not a handoff.
				if !strings.Contains(reply, "<summary>") {
					return "", usage, errors.New("shared compaction reply carried no summary block")
				}
				return nativehost.ClampSummary(nativehost.ExtractSummary(reply), summaryTokens), usage, nil
			}
		}
	}
	if len(a.hooks.BeforeCompact) > 0 && policy.Allow == nil {
		policy.Allow = func(ctx context.Context, prefix json.RawMessage) bool {
			var raws []json.RawMessage
			_ = json.Unmarshal(prefix, &raws)
			settings := a.compaction
			settings.KeepRecentTokens = policy.KeepRecent
			request := CompactRequest{Turn: result.Turns, Budget: policy.Budget, Settings: settings}
			for _, raw := range raws {
				if message, err := ai.ProjectMessage(raw); err == nil {
					request.Messages = append(request.Messages, message)
				}
			}
			decision, err := a.hooks.runBeforeCompact(ctx, request)
			if err != nil {
				return false
			}
			if decision.Messages != nil {
				_ = a.hooks.emitErr("before_compact", errors.New("a native run keeps its transcript: this hook can skip a compaction but cannot replace the messages; use CompactionSettings.Instructions to shape the summary"))
			}
			return !decision.Skip
		}
	}
	observeCompaction := policy.Observe
	policy.Observe = func(event nativehost.CompactionEvent) {
		result.Compactions = append(result.Compactions, event)
		if observeCompaction != nil {
			observeCompaction(event)
		}
	}
	compactor := nativehost.NewCompactor(nativehost.CompactorOptions{Revision: nativeCheckpointRevision, Policy: policy, Usage: func(u Usage) { result.Usage = result.Usage.Add(u) }})
	if len(checkpoint.Summary) > 0 {
		if err := compactor.Restore(checkpoint.Summary); err != nil {
			// The full opaque transcript remains available; discard a bad
			// summary rather than replacing evidence or blocking continuation.
			emit(StreamEvent{Type: StreamProgress, Note: "Discarding invalid context summary; retaining full history."})
		}
	}
	if len(checkpoint.Cleared) > 0 {
		if err := compactor.RestoreCleared(checkpoint.Cleared); err != nil {
			emit(StreamEvent{Type: StreamProgress, Note: "Discarding invalid tool-result clearing; retaining full results."})
		}
	}
	// resumable is the checkpoint of the last settled turn. A run that fails after
	// it returns this state, so the consumer can resume from it.
	var resumable struct {
		checkpoint nativeCheckpoint
		count      int
		ok         bool
	}
	safeState := func() json.RawMessage {
		if !resumable.ok {
			return nil
		}
		state := resumable.checkpoint
		state.Messages = joinRawMessages(settled[:resumable.count])
		raw, _ := json.Marshal(state)
		return raw
	}
	capture := func(ctx context.Context) error {
		plugins, err := h.nativePluginState()
		if err != nil {
			return err
		}
		resumable.checkpoint = nativeCheckpoint{Question: checkpoint.Question, Goal: h.committedGoal, GoalRevisions: slices.Clone(checkpoint.GoalRevisions), Plugins: plugins, Revision: nativeCheckpointRevision, Model: slices.Clone(provider.Candidates[selected].Model), Summary: compactor.Checkpoint(), Cleared: compactor.Cleared()}
		resumable.count, resumable.ok = len(settled), true
		if input.Checkpoint == nil {
			return nil
		}
		return input.Checkpoint(context.WithoutCancel(ctx), safeState())
	}
	defer func() {
		if failure != nil && result.NativeState == nil {
			if state := safeState(); state != nil {
				result.NativeState, result.NativeRevision = state, nativeCheckpointRevision
			}
		}
	}()
	if input.Compact != nil {
		raw, err := encode(history)
		if err != nil {
			return result, err
		}
		manual := policy.ForWindow(nativeContextWindow(provider.Candidates[selected].Model))
		manual.Instructions = strings.TrimSpace(policy.Instructions + "\n\n" + input.Compact.Instructions)
		if err := compactor.Compact(ctx, h.TransformContext(ctx, raw), manual); err != nil {
			return result, fmt.Errorf("compact: %w", err)
		}
	}
	var pending []Message
	var stopReason string
	bookkeepingTurns := 0
	prepare := func(turn int, model string) (*engine.TurnUpdate, error) {
		prepared, err := h.PrepareTurn(ctx, StepInfo{Turn: turn, BookkeepingTurns: bookkeepingTurns, Model: model, Usage: result.Usage})
		if err != nil {
			return nil, err
		}
		if prepared.StopReason != "" {
			stopReason = prepared.StopReason
		}
		pending = append(pending, prepared.Messages...)
		for _, item := range a.drainInbox(InboxSteer) {
			pending = append(pending, item.Message)
		}
		additions, err := nativehost.InputMessages(pending)
		if err != nil {
			return nil, err
		}
		pending = nil
		messages := engine.NewList[*ai.Message]()
		for _, raw := range additions {
			var message ai.Message
			if err := json.Unmarshal(raw, &message); err != nil {
				return nil, err
			}
			messages.Append(&message)
		}
		system, changed, err := h.RefreshGoal()
		if err != nil {
			return nil, err
		}
		if changed && len(a.hooks.BeforeAgentStart) > 0 {
			// A goal revision re-assembles the prompt; the start hooks shape
			// it again so their edits survive. There are no seeds to shape.
			start, err := a.hooks.runBeforeAgentStart(ctx, RunStart{System: system, Task: input.Task})
			if err != nil {
				return nil, err
			}
			system = start.System
		}
		if changed && system != sentSystem {
			messages.Append(nativeSystemMessage(system))
			sentSystem = system
		}
		if prepared.Note != "" {
			emit(StreamEvent{Type: StreamProgress, Turn: turn, Note: prepared.Note})
		}
		return &engine.TurnUpdate{Messages: messages}, nil
	}
	options := engine.AgentOptions{InitialState: engine.InitialState{Model: provider.Candidates[selected].Model, ThinkingLevel: a.reasoningEffort, Messages: history, Tools: tools}}
	options.Options = map[string]any{}
	if a.maxTokens > 0 {
		options.Options["maxTokens"] = a.maxTokens
	}
	if a.cacheKey != "" {
		options.Options["sessionId"] = a.cacheKey
		options.Options["cacheRetention"] = a.cacheRetention
	}
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	if err := ai.ValidateNativeToolChoice(a.toolChoice, names); err != nil {
		return result, err
	}
	if a.toolChoice.Mode != ToolChoiceDefault || a.parallelToolCalls != nil || a.outputSchema != nil {
		options.Options["onPayload"] = func(_ context.Context, payload, model json.RawMessage) (json.RawMessage, error) {
			var selected struct{ API string }
			if err := json.Unmarshal(model, &selected); err != nil {
				return nil, err
			}
			return ai.ApplyNativeControls(selected.API, payload, ai.NativeGenerationControls{ToolChoice: a.toolChoice, ParallelToolCalls: a.parallelToolCalls, OutputSchema: a.outputSchema})
		}
	}
	// The wrap-up turn after a ceiling keeps its tool declarations, so the
	// provider still reads the cached prefix, and is asked for with tool
	// choice "none" instead; a call made anyway is refused as finalizing.
	wrapUp := func(controls map[string]any) map[string]any {
		controls = maps.Clone(controls)
		controls["toolChoice"] = "none"
		controls["onPayload"] = func(_ context.Context, payload, model json.RawMessage) (json.RawMessage, error) {
			var selected struct{ API string }
			_ = json.Unmarshal(model, &selected)
			out, err := ai.ApplyNativeControls(selected.API, payload, ai.NativeGenerationControls{ToolChoice: ToolChoice{Mode: ToolChoiceNone}, ParallelToolCalls: a.parallelToolCalls, OutputSchema: a.outputSchema})
			if err != nil {
				// The "toolChoice" option already carries the choice to a
				// client this payload rewrite does not know.
				return payload, nil
			}
			return out, nil
		}
		return controls
	}
	options.AdmitRequest = func(ctx context.Context, source engine.Request, controls map[string]any) (*engine.RequestAdmission, error) {
		if h.finalizing.Load() {
			controls = wrapUp(controls)
		}
		return nativehost.Admit(ctx, func(ctx context.Context, out *ai.AssistantMessageEventStream, accept func(engine.Request)) (engine.Request, ai.AttemptOutcome, error) {
			var prepared engine.Request
			lastView := map[int]json.RawMessage{}
			recoveryView := map[int]json.RawMessage{}
			final, err := telemetry.StartSpan(input.Telemetry, telemetry.SpanOptions{Name: "agentray.ai.request"}, func(span *telemetry.Span) (ai.AttemptOutcome, error) {
				ctx = telemetry.WithContext(ctx, span.Context())
				trace := ai.NewAttemptTrace(span)
				outcome, failure := provider.Run(ctx, out, ladder.Bind(ai.FallbackRequest{Candidates: len(provider.Candidates),
					Open: func(ctx context.Context, index, _ int) (*ai.AssistantMessageEventStream, error) {
						candidate := provider.Candidates[index]
						prepared = engine.Request{Context: source.Context, Model: candidate.Model, ThinkingLevel: source.ThinkingLevel}
						raw, err := encode(source.Context.Messages.Values())
						if err != nil {
							return nil, &ai.PreparationError{Cause: err}
						}
						raw = h.TransformContext(ctx, raw)
						var model struct{ ContextWindow int }
						_ = json.Unmarshal(candidate.Model, &model)
						prior := compactor.Checkpoint()
						shared.index, shared.controls, shared.ok = index, controls, true
						if recovered := recoveryView[index]; recovered != nil {
							raw = recovered
						} else {
							raw, err = compactor.Prepare(ctx, raw, policy.ForWindow(model.ContextWindow))
						}
						if err != nil {
							var budgetError *nativehost.ContextBudgetError
							if errors.As(err, &budgetError) {
								emit(StreamEvent{Type: StreamProgress, Turn: result.Turns, Note: budgetError.Error()})
							}
							return nil, &ai.PreparationError{Cause: err}
						}
						lastView[index] = slices.Clone(raw)
						if summary := compactor.Checkpoint(); len(summary) > 0 && !bytes.Equal(prior, summary) {
							if err := h.ObserveMessages(ctx, PhaseRebase, result.Turns, raw); err != nil {
								return nil, &ai.PreparationError{Cause: err}
							}
						}
						var messages []ai.Message
						if err := json.Unmarshal(raw, &messages); err != nil {
							return nil, &ai.PreparationError{Cause: err}
						}
						visible := messages[:0]
						for _, message := range messages {
							switch message.Role {
							case "system", "user", "assistant", "toolResult":
								visible = append(visible, message)
							}
						}
						transcript := ai.NormalizeContext(ai.Context{Messages: visible})
						view, err := json.Marshal(transcript.Messages())
						if err != nil {
							return nil, &ai.PreparationError{Cause: err}
						}
						if err := h.ObserveMessages(ctx, PhaseRequest, result.Turns, view); err != nil {
							return nil, &ai.PreparationError{Cause: err}
						}
						if len(a.hooks.BeforeProviderRequest) > 0 {
							if err := a.observeNativeRequest(ctx, candidate.Model, view, source.Context.Tools); err != nil {
								return nil, &ai.PreparationError{Cause: err}
							}
						}
						trace.Start(candidate.Model, transcript)
						return candidate.Stream(ctx, slices.Clone(candidate.Model), transcript, maps.Clone(controls))
					},
					Commit: func(_ context.Context, index int) error { selected = index; accept(prepared); return nil },
					Observe: func(ctx context.Context, _ int, attempt ai.FallbackAttempt) error {
						account(attempt.Outcome)
						trace.Finish(attempt)
						return nil
					},
					Recover: func(ctx context.Context, index int, attempt ai.FallbackAttempt) (bool, error) {
						if !ai.IsContextOverflow(attempt.Failure) || attempt.Outcome.Committed || ctx.Err() != nil {
							return false, nil
						}
						var model struct{ ContextWindow int }
						_ = json.Unmarshal(provider.Candidates[index].Model, &model)
						forced := policy.ForWindow(model.ContextWindow)
						forced.Force = true
						// The provider just rejected this view as too large.
						shared.ok = false
						original, _ := encode(source.Context.Messages.Values())
						original = h.TransformContext(ctx, original)
						view, err := compactor.Prepare(ctx, original, forced)
						if err != nil {
							return false, nil
						}
						if len(view) >= len(lastView[index]) {
							return false, nil
						}
						recoveryView[index] = view
						emit(StreamEvent{Type: StreamProgress, Note: "Context overflow rejected before output; retrying once with a smaller request"})
						return true, nil
					},
				}))
				if message := outcome.Message(); message != nil && (message.StopReason == "error" || message.StopReason == "aborted") {
					span.SetStatus(telemetry.SpanStatus{Status: "error"})
					if message.StopReason == "aborted" {
						kind := "request_cancelled"
						if ctx.Err() == context.DeadlineExceeded {
							kind = "request_timeout"
						}
						span.SetSafeOutcome("cancelled", kind)
					}
				}
				return outcome, failure
			})
			return prepared, final, err
		})
	}
	finishTurn := func(ctx context.Context, turn *engine.Turn) (string, error) {
		raw, err := json.Marshal(turn.Message)
		if err != nil {
			return "", err
		}
		message, err := ai.ProjectMessage(raw)
		if err != nil {
			return "", err
		}
		var outcomes []ToolOutcome
		for _, value := range turn.ToolResults.Values() {
			data, err := json.Marshal(value.Details)
			if err != nil {
				return "", err
			}
			var outcome ToolOutcome
			if json.Unmarshal(data, &outcome) == nil && outcome.Trace.CallID == value.ToolCallID {
				outcomes = append(outcomes, outcome)
			}
		}
		if len(message.ToolCalls) > 0 {
			administrative := len(outcomes) == len(message.ToolCalls)
			for _, outcome := range outcomes {
				if !outcome.Executed {
					administrative = false
				}
			}
			for _, call := range message.ToolCalls {
				if !isBookkeeping(h.tools, call.Name) {
					administrative = false
					break
				}
			}
			if administrative {
				bookkeepingTurns++
			}
		}
		decision, err := h.FinishTurn(ctx, CompletedTurn{Info: StopInfo{Final: message.Content, Turns: result.Turns, Tools: slices.Clone(result.Tools)}, Usage: result.Usage, Model: turn.Message.Model, StopReason: turn.Message.StopReason, Calls: message.ToolCalls, Outcomes: outcomes})
		if err != nil {
			return "", err
		}
		pending = append(pending, decision.Inject...)
		if decision.StopReason != "" {
			stopReason = decision.StopReason
		}
		if decision.Parked {
			stopReason = "parked"
			result.Parked = true
			for _, outcome := range outcomes {
				if outcome.Parked {
					result.Question = json.RawMessage(outcome.Trace.Args)
					checkpoint.Question = &nativeQuestion{CallID: outcome.Trace.CallID, Args: result.Question}
					break
				}
			}
			emit(StreamEvent{Type: StreamQuestion, Turn: result.Turns, Question: result.Question})
		}
		if decision.Note != "" {
			emit(StreamEvent{Type: StreamProgress, Turn: result.Turns, Note: decision.Note})
		}
		if decision.End {
			return "end", nil
		}
		if len(message.ToolCalls) == 0 && !decision.Continue {
			for _, item := range a.drainInbox(InboxFollow) {
				pending = append(pending, item.Message)
			}
			if len(pending) > 0 {
				return "continue", nil
			}
		}
		if decision.Continue {
			return "continue", nil
		}
		return "", nil
	}
	options.FinishTurn = func(ctx context.Context, turn *engine.Turn) (string, error) {
		decision, err := finishTurn(ctx, turn)
		if err != nil {
			return "", err
		}
		// Every message of the turn has settled: a safe point to resume from.
		if err := capture(ctx); err != nil {
			return "", fmt.Errorf("checkpoint: %w", err)
		}
		return decision, nil
	}
	options.PrepareNextTurnWithContext = func(ctx context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
		return prepare(result.Turns+1, turn.Message.Model)
	}
	agent, err := engine.NewAgent(options)
	if err != nil {
		return result, err
	}
	// settled starts as the agent's own initial transcript: the engine opens
	// an empty history with its tool declarations without an event.
	for _, message := range agent.State().Messages.Values() {
		raw := encoded[message]
		if raw == nil {
			if raw, err = json.Marshal(message); err != nil {
				return result, err
			}
		}
		settled, encoded[message] = append(settled, raw), raw
	}
	// Live completion events follow completion order; the returned audit follows
	// invocation order, independently of parallel-tool scheduling.
	type toolGroup struct {
		order  int
		traces []ToolTrace
	}
	var groups []toolGroup
	order := map[string]int{}
	ordinal := 0
	defer func() {
		slices.SortStableFunc(groups, func(a, b toolGroup) int { return a.order - b.order })
		result.Tools = nil
		for _, group := range groups {
			result.Tools = append(result.Tools, group.traces...)
		}
	}()
	agent.Subscribe(&engine.Listener{Handle: func(ctx context.Context, event engine.Event) error {
		stream := StreamEvent{Turn: result.Turns}
		switch event.Type {
		case "agent_start":
			stream.Type = StreamAgentStart
		case "agent_end":
			stream.Type = StreamAgentEnd
		case "turn_start":
			result.Turns++
			stream.Type = StreamTurnStart
			stream.Turn = result.Turns
		case "turn_end":
			stream.Type = StreamTurnEnd
		case "message_start":
			if event.Message.Role != "assistant" {
				return nil
			}
			stream.Type = StreamMessageStart
			stream.Turn = result.Turns
		case "message_update":
			if event.AssistantMessageEvent.Type != "text_delta" {
				return nil
			}
			stream.Type = StreamToken
			stream.Token = event.AssistantMessageEvent.Delta
		case "message_end":
			data, err := json.Marshal(event.Message)
			if err != nil {
				return err
			}
			settled, encoded[event.Message] = append(settled, data), data
			if err := h.ObserveMessages(ctx, PhaseAppend, result.Turns, joinRawMessages([]json.RawMessage{data})); err != nil {
				return err
			}
			if event.Message.Role != "assistant" {
				return nil
			}
			projected, err := ai.ProjectMessage(data)
			if err != nil {
				return err
			}
			if err := a.hooks.runMessageEnd(ctx, projected); err != nil {
				return err
			}
			result.Final = projected.Content
			result.StopReason = event.Message.StopReason
			stream.Type = StreamMessageEnd
		case "tool_execution_start":
			order[event.ToolCallID] = ordinal
			ordinal++
			stream.Type = StreamToolExecStart
			stream.Tool = &ToolTrace{CallID: event.ToolCallID, Tool: event.ToolName, Args: string(event.Args)}
		case "tool_execution_update":
			// spawn_subagent emits short lifecycle notes as streaming partials.
			// Those are useful to a live viewer; other tools' partial output may
			// contain arbitrary result data and stays private to AgentCore.
			if event.ToolName != "spawn_subagent" || event.PartialResult == nil || event.PartialResult.Content == nil {
				return nil
			}
			raw, err := json.Marshal(event.PartialResult.Content)
			if err != nil {
				return nil
			}
			note, _, _, err := ai.ProjectContent(raw)
			if err != nil || note == "" {
				return nil
			}
			stream.Type = StreamToolExecUpdate
			stream.Note = note
		case "tool_execution_end":
			stream.Type = StreamToolExecEnd
			var audit ToolOutcome
			if event.Result != nil {
				raw, _ := json.Marshal(event.Result.Details)
				_ = json.Unmarshal(raw, &audit)
			}
			trace := audit.Trace
			if trace.CallID == "" {
				trace = ToolTrace{CallID: event.ToolCallID, Tool: event.ToolName, Args: string(event.Args), Allowed: !event.IsError}
				if event.Result != nil {
					raw, _ := json.Marshal(event.Result.Content)
					text, _, _, _ := ai.ProjectContent(raw)
					trace.ResultMeta = text
					if event.IsError {
						trace.Error = text
					}
				}
			}
			group := toolGroup{order: order[event.ToolCallID], traces: []ToolTrace{trace}}
			for _, nested := range audit.Invocations {
				group.traces = append(group.traces, nested.Trace)
			}
			groups = append(groups, group)
			stream.Tool = &trace
			emit(stream)
			emit(StreamEvent{Type: StreamTool, Turn: result.Turns, Tool: &trace})
			return nil
		default:
			return nil
		}
		emit(stream)
		return nil
	}})

	reason, err := h.ControlRun(ctx, StepInfo{Turn: 1, Model: a.model})
	if err != nil {
		return result, err
	}
	if reason != "" {
		stopReason = reason
	}
	if input.ControlOnly && stopReason == "" {
		stopReason = "controlled"
	}
	if stopReason == "" {
		first, err := prepare(1, a.model)
		if err != nil {
			return result, err
		}
		var prompts []*ai.Message
		if system != sentSystem {
			prompts = append(prompts, nativeSystemMessage(system))
			sentSystem = system
		}
		if memory := nativeRecalledMemory(recalled, history, checkpoint.Summary); memory != nil {
			prompts = append(prompts, memory)
		}
		for _, raw := range rawInput {
			var message ai.Message
			if err := json.Unmarshal(raw, &message); err != nil {
				return result, err
			}
			prompts = append(prompts, &message)
		}
		prompts = append(prompts, first.Messages.Values()...)
		if answerQuestion {
			checkpoint.Question = nil
		}
		failure = agent.Prompt(ctx, engine.MessageValues(prompts))
	}

	state := agent.State()
	if stopReason != "" {
		result.StopReason = stopReason
	}
	if state.ErrorMessage != nil && *state.ErrorMessage != "" && failure == nil {
		failure = errors.New(*state.ErrorMessage)
	}
	messages, err := json.Marshal(state.Messages)
	if err == nil {
		// A run stopped inside a tool batch leaves no resumable transcript;
		// the last settled turn's state is returned instead.
		err = nativehost.ValidateMessages(messages)
	}
	if err != nil {
		return result, errors.Join(failure, err)
	}

	for _, message := range state.Messages.Values() {
		raw, _ := json.Marshal(message)
		projected, err := ai.ProjectMessage(raw)
		if err != nil {
			return result, errors.Join(failure, err)
		}
		result.Messages = append(result.Messages, projected)
	}
	finalResult := result
	finalResult.Usage = finalResult.Usage.Add(a.peekChildUsage())
	for i := len(h.exts.all) - 1; i >= 0; i-- {
		ext := h.exts.all[i]
		if f, ok := ext.(RunFinalizer); ok {
			finalResult.Usage = result.Usage.Add(a.peekChildUsage())
			var err error
			if panicErr := safe(func() { err = f.FinalizeRun(ctx, finalResult, failure) }); panicErr != nil {
				err = panicErr
			}
			if err != nil {
				return result, errors.Join(failure, fmt.Errorf("finalize %s: %w", ext.Name(), err))
			}
		}
	}
	pluginState, err := h.nativePluginState()
	if err != nil {
		return result, errors.Join(failure, err)
	}
	checkpoint = nativeCheckpoint{Question: checkpoint.Question, Goal: h.committedGoal, GoalRevisions: checkpoint.GoalRevisions, Plugins: pluginState, Revision: nativeCheckpointRevision, Model: slices.Clone(provider.Candidates[selected].Model), Messages: messages, Summary: compactor.Checkpoint(), Cleared: compactor.Cleared()}
	result.NativeRevision = nativeCheckpointRevision
	result.NativeState, err = json.Marshal(checkpoint)
	if err != nil {
		return result, errors.Join(failure, err)
	}

	return result, failure
}

// nativeRecalledMemory is the run's recalled memory as a context message ahead
// of its input, or nil when there is none. It enters the transcript, so later
// requests read it from the prompt cache instead of rewriting the prefix, and
// a bullet the request still carries verbatim is not repeated; one covered by a
// compaction summary is, since the summary may have dropped it.
func nativeRecalledMemory(recalled []MemoryEntry, history []*ai.Message, summary json.RawMessage) *ai.Message {
	visible := history
	var saved nativehost.Summary
	if len(summary) > 0 && json.Unmarshal(summary, &saved) == nil && saved.PrefixCount > 0 && saved.PrefixCount <= len(visible) {
		visible = visible[saved.PrefixCount:]
	}
	memory := renderRecalledMemory(recalled, func(bullet string) bool {
		for _, message := range visible {
			if message != nil && message.Role == "user" && strings.Contains(ai.ContentText(message.Content), bullet) {
				return true
			}
		}
		return false
	})
	if memory == "" {
		return nil
	}
	return &ai.Message{Role: "user", Content: ai.TextContent(memory), Timestamp: time.Now().UnixMilli()}
}

// observeNativeRequest hands the BeforeProviderRequest hooks a projection of
// the request about to be sent: the model, the visible messages (the system
// prompt first), the tool schemas and the cache identity. Native requests keep
// their provider JSON, so the projection is read-only: a hook that returns a
// different request is reported as a hook failure (use Hooks.Context or
// generation controls to change what is sent).
func (a *Agent) observeNativeRequest(ctx context.Context, model json.RawMessage, view json.RawMessage, tools *engine.ToolList) error {
	var selected struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(model, &selected)
	req := ChatRequest{Model: selected.ID, MaxTokens: a.maxTokens, SessionID: a.cacheKey, CacheKey: a.cacheKey, CacheRetention: a.cacheRetention}
	var raws []json.RawMessage
	if err := json.Unmarshal(view, &raws); err != nil {
		return err
	}
	for _, raw := range raws {
		message, err := ai.ProjectMessage(raw)
		if err != nil {
			return err
		}
		// An empty system message only announces tool changes; Tools
		// carries those.
		if message.Role == RoleSystem && message.Content == "" {
			continue
		}
		req.Messages = append(req.Messages, message)
	}
	if tools != nil {
		for _, tool := range tools.Values() {
			var parameters map[string]any
			_ = json.Unmarshal(tool.Parameters, &parameters)
			req.Tools = append(req.Tools, ToolSchema{Name: tool.Name, Description: tool.Description, Parameters: parameters})
		}
	}
	before, err := json.Marshal(req)
	if err != nil {
		return err
	}
	out, err := a.hooks.runBeforeProviderRequest(ctx, req)
	if err != nil {
		return err
	}
	if after, err := json.Marshal(out); err == nil && !bytes.Equal(before, after) {
		return a.hooks.emitErr("before_provider_request", errors.New("native requests are read-only to this hook; rewrite messages with a Hooks.Context hook"))
	}
	return nil
}

// nativeSystemText is the system prompt a transcript already carries: the
// latest value of the host's section, which a later system message replaces.
// A resumed run that would send the same text appends nothing, so the
// transcript does not gain one identical system message per run.
func nativeSystemText(history []*ai.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i] == nil || history[i].Role != "system" {
			continue
		}
		for j := len(history[i].Sections) - 1; j >= 0; j-- {
			if section := history[i].Sections[j]; section.Name == nativeSystemSection {
				if section.Value == nil {
					return ""
				}
				return *section.Value
			}
		}
	}
	return ""
}

const nativeSystemSection = "agentray"

func nativeSystemMessage(system string) *ai.Message {
	return &ai.Message{Role: "system", Content: ai.TextContent(""), Sections: ai.SystemSections{{Name: nativeSystemSection, Value: &system}}, Timestamp: time.Now().UnixMilli()}
}

func (h *ToolHost) nativeTools(ctx context.Context, parent telemetry.Context, safeLabels map[string]telemetry.SafeLabels, events func(StreamEvent)) ([]*engine.Tool, error) {
	definitions, err := h.Definitions(ctx)
	if err != nil {
		return nil, err
	}
	var raw []json.RawMessage
	if err := json.Unmarshal(definitions, &raw); err != nil {
		return nil, err
	}
	var tools []*engine.Tool
	for _, declaration := range raw {
		tool := new(engine.Tool)
		if err := json.Unmarshal(declaration, &tool.Tool); err != nil {
			return nil, err
		}
		var extra struct{ Label, ExecutionMode string }
		if err := json.Unmarshal(declaration, &extra); err != nil {
			return nil, err
		}
		tool.Label, tool.ExecutionMode = extra.Label, extra.ExecutionMode
		for _, name := range []string{"label", "executionMode", "agentrayPrepareArguments"} {
			delete(tool.Extra, name)
		}
		original, _ := h.tools.Get(tool.Name)
		if preparer, ok := original.(ArgPreparer); ok {
			tool.PrepareArguments = func(raw json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(preparer.PrepareArguments(string(raw))), nil
			}
		}
		tool.Execute = func(ctx context.Context, id string, args any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
			return telemetry.StartSpan(parent, telemetry.SpanOptions{Name: "agentray.tool.execute", SafeLabels: safeLabels[tool.Name], Attributes: telemetry.NewAttributes(telemetry.Property{Name: "tool.name", Value: tool.Name})}, func(span *telemetry.Span) (*engine.ToolResult, error) {
				ctx = telemetry.WithContext(ctx, span.Context())
				encoded, err := jsonjs.MarshalValue(args)
				if err != nil {
					return nil, err
				}
				params, err := json.Marshal(map[string]any{"toolCallId": id, "toolName": tool.Name, "args": json.RawMessage(encoded)})
				if err != nil {
					return nil, err
				}
				// No automatic replay occurs on this consumer-owned checkpoint
				// path. Every engine execution is a new intention, even when the
				// provider reuses a call ID in a later turn.
				ctx = WithToolInvocationScope(ctx, newEntryID())
				ctx = context.WithValue(ctx, nestedToolEventsKey{}, events)
				raw, audit, err := h.Execute(ctx, params, func(raw json.RawMessage) error {
					var result engine.ToolResult
					if err := json.Unmarshal(raw, &result); err != nil {
						return err
					}
					update(&result)
					return nil
				})
				if err != nil {
					return nil, err
				}
				if !audit.Trace.Allowed {
					span.SetSafeOutcome("failed", "tool_denied")
				}
				var result engine.ToolResult
				if err := json.Unmarshal(raw, &result); err != nil {
					return nil, err
				}
				if result.IsError != nil && *result.IsError {
					span.SetStatus(telemetry.SpanStatus{Status: "error"})
				}
				return &result, nil
			})
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

// joinRawMessages encodes a JSON array of already encoded messages.
func joinRawMessages(messages []json.RawMessage) json.RawMessage {
	size := 2
	for _, raw := range messages {
		size += len(raw) + 1
	}
	out := make([]byte, 0, size)
	out = append(out, '[')
	for i, raw := range messages {
		if i > 0 {
			out = append(out, ',')
		}
		out = append(out, raw...)
	}
	return append(out, ']')
}

// nativeContextWindow is a native model's declared window, or 0.
func nativeContextWindow(model json.RawMessage) int {
	var declared struct{ ContextWindow int }
	_ = json.Unmarshal(model, &declared)
	return declared.ContextWindow
}

// nativePluginState collects every extension's checkpoint state.
func (h *ToolHost) nativePluginState() (map[string]json.RawMessage, error) {
	plugins := make(map[string]json.RawMessage)
	for _, ext := range h.exts.all {
		if state, ok := ext.(NativeStateContributor); ok {
			raw, err := state.NativeState()
			if err != nil {
				return nil, fmt.Errorf("checkpoint %s: %w", ext.Name(), err)
			}
			plugins[ext.Name()] = raw
		}
	}
	return plugins, nil
}

// controlInt reads a numeric request control, or 0.
func controlInt(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}
