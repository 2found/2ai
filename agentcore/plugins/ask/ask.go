// Package ask contributes the `ask` tool: the agent poses a structured
// question to the human — a prompt plus labeled options, optionally
// multi-select — and the run parks until the answer arrives.
//
// Parking is a loop mechanism, not a block: Run returns agentcore.ErrParked,
// the loop records an EntryQuestion for the call and ends the run WITHOUT a
// tool result, leaving the call dangling in the durable log. The answer
// arrives out-of-band as an EntryAnswer keyed on the same call id; a resumed
// run closes the dangling call with that answer as its tool result — the
// model sees the human's reply exactly as if the tool had returned it. A
// resume that finds the question still unanswered re-issues the call (the
// tool is retry-safe), which parks the run again on the same question.
//
// The tool is advertised only where a human can answer — the host wires it on
// chat-triggered runs and leaves it off scheduled/webhook/delegate/manual
// runs. It is self-gated: it returns nothing but what the user typed, so it
// cannot reach anything the run does not already have.
package ask

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

// ToolName is the model-visible identifier.
const ToolName = "ask"

// Bounds on what a question may carry. The clamps live in PrepareArguments so
// the durable EntryQuestion — and the card the UI renders — holds the bounded
// form, never the model's raw overshoot.
const (
	maxQuestionLen = 2000
	maxOptionLen   = 200
	maxOptions     = 8
)

// Option is one labeled choice the human can pick.
type Option struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// args is the tool's input shape.
type args struct {
	Question string   `json:"question"`
	Options  []Option `json:"options,omitempty"`
	Multi    bool     `json:"multi,omitempty"`
}

// clamp bounds one args value in place: text lengths, option count, and the
// one-question-per-call rule (a questions array is folded into the first).
func clamp(in *args) {
	in.Question = strings.TrimSpace(in.Question)
	in.Question = boundedText(in.Question, maxQuestionLen)
	if len(in.Options) > maxOptions {
		in.Options = in.Options[:maxOptions]
	}
	for i := range in.Options {
		in.Options[i].Label = strings.TrimSpace(in.Options[i].Label)
		in.Options[i].Label = boundedText(in.Options[i].Label, maxOptionLen)
		in.Options[i].Description = boundedText(in.Options[i].Description, maxOptionLen)
	}
}

func boundedText(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	for limit > 0 && !utf8.RuneStart(s[limit]) {
		limit--
	}
	return s[:limit]
}

// Tool is the ask capability. It carries no state: everything durable lives in
// the session log the loop writes, so the same value serves every run.
type Tool struct{}

// Name identifies the tool.
func (Tool) Name() string { return ToolName }

// PiArgumentPreparation selects this plugin's bundled synchronous Pi hook.
func (Tool) PiArgumentPreparation() string { return "ask-v1" }

// Schema advertises the tool to the model.
func (Tool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolName,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Ask the user a structured question and wait for their answer. " +
			"Use when you need a decision only the user can make — a choice between approaches, " +
			"a missing detail, a confirmation. The run pauses until they answer; their reply " +
			"comes back as this call's result. Ask ONE question per call. Prefer labeled options " +
			"the user can pick with one click; set multi when several may apply.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"question": map[string]any{
					"type":        "string",
					"description": "The question to ask, phrased for the user.",
				},
				"options": map[string]any{
					"type":        "array",
					"description": "Labeled choices the user can pick from. Omit for a free-text answer.",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"label":       map[string]any{"type": "string", "description": "Short choice label."},
							"description": map[string]any{"type": "string", "description": "Optional detail shown under the label."},
						},
						"required": []string{"label"},
					},
				},
				"multi": map[string]any{
					"type":        "boolean",
					"description": "Allow picking several options. Default false.",
				},
			},
			"required": []string{"question"},
		},
	}
}

// PrepareArguments normalizes the raw call before validation: clamps bound the
// question and options so the durable EntryQuestion records the form the human
// actually sees.
func (Tool) PrepareArguments(raw string) string {
	var in args
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return raw // malformed JSON fails validation on the original text
	}
	clamp(&in)
	out, err := json.Marshal(in)
	if err != nil {
		return raw
	}
	return string(out)
}

// Run never produces a result: a valid call parks the run. The answer reaches
// the model through the durable log (EntryAnswer), not through this return.
func (Tool) Run(_ context.Context, raw string) (string, error) {
	var in args
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Question) == "" {
		return "", errors.New("question is required")
	}
	return "", agentcore.ErrParked
}

// SelfGated exempts the tool from the permission policy: it returns only what
// the user typed, so it reaches nothing the run does not already have.
func (Tool) SelfGated() bool { return true }

// RetrySafe makes a dangling ask re-issue on resume: re-parking on the same
// question is the correct recovery when the answer never arrived.
func (Tool) RetrySafe() bool { return true }

// Plugin offers questions on an interactive top-level run. Delegated children
// report missing information to their parent, which owns the human channel.
type Plugin struct{}

func (Plugin) Name() string                           { return ToolName }
func (p Plugin) Register(r *agentcore.Registry) error { r.AddExtension(p); return nil }
func (p Plugin) BeginRun(_ context.Context, info agentcore.RunInfo) (agentcore.Extension, error) {
	if info.Depth > 0 {
		return nil, nil
	}
	return p, nil
}
func (Plugin) Tools() []agentcore.Tool { return []agentcore.Tool{Tool{}} }
func (Plugin) SelfGated() bool         { return true }

// QuestionText renders the bounded question for hosts with a plain-text human
// channel. Structured clients may render the original payload directly.
func QuestionText(raw json.RawMessage) string {
	var question args
	if json.Unmarshal(raw, &question) != nil {
		return ""
	}
	clamp(&question)
	out := question.Question
	for _, option := range question.Options {
		out += "\n- " + option.Label
		if option.Description != "" {
			out += ": " + option.Description
		}
	}
	return out
}
