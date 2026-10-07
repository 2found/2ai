// Package engine is the native Go port of Pi's agent state machine. It works
// with the lossless ai transcript rather than the legacy agentcore messages.
package engine

import (
	"context"
	"encoding/json"
	"time"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

// Context retains live message/tool collections shared with hooks. Replacing a
// field detaches that collection; edits through a retained list remain visible.
type Context struct {
	Messages        *MessageList `json:"messages"`
	Tools           *ToolList    `json:"-"`
	assistantStream *ai.AssistantMessageEventStream
}

func (c *Context) readAssistant(read func()) {
	if c.assistantStream != nil {
		c.assistantStream.Synchronize(read)
		return
	}
	read()
}

// Tool is shared by pointer. Preparation selects the object before invoking
// argument and lifecycle hooks; later field edits affect that selected tool,
// while replacing its entry in Context.Tools only affects future selection.
// Callers must synchronize concurrent mutation of tool definitions/callbacks.
type Tool struct {
	ai.Tool
	Label            string
	OutputSchema     json.RawMessage
	Replay           string
	ExecutionMode    string
	PrepareArguments func(json.RawMessage) (json.RawMessage, error)
	// Execute shares validated *Object/*Array values with hooks; primitives
	// are passed by value. Nested edits are shared, local reassignment is not.
	// Updates share result objects with hooks and events.
	// Tools must synchronize access when retaining or sharing these pointers.
	Execute func(context.Context, string, any, func(*ToolResult)) (*ToolResult, error)
}

func (t Tool) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(t.Tool)
	if err != nil {
		return nil, err
	}
	var fields jsonjs.RawObject
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	fields["label"] = jsonjs.QuoteString(t.Label)
	if t.OutputSchema != nil {
		fields["outputSchema"] = t.OutputSchema
	}
	if t.Replay != "" {
		fields["replay"], _ = json.Marshal(t.Replay)
	}
	if t.ExecutionMode != "" {
		fields["executionMode"], _ = json.Marshal(t.ExecutionMode)
	}
	return json.Marshal(fields)
}

// ToolResult shares its content list, Details and StructuredContent with hooks/events.
// Values use *Object, *Array or primitives. nil/Undefined omit an optional
// field; Null retains explicit JSON null. Concurrent access needs synchronization.
type ToolResult struct {
	preserved         map[string]json.RawMessage
	Content           *ai.BlockList `json:"content"`
	Details           any           `json:"-"`
	StructuredContent any           `json:"-"`
	Usage             *ai.Usage     `json:"usage,omitempty"`
	IsError           *bool         `json:"isError,omitempty"`
	Terminate         *bool         `json:"terminate,omitempty"`
}

func (r ToolResult) MarshalJSON() ([]byte, error) {
	type plain ToolResult
	encoded, err := json.Marshal(plain(r))
	if err != nil {
		return nil, err
	}
	var fields jsonjs.RawObject
	if err := json.Unmarshal(encoded, &fields); err != nil {
		return nil, err
	}
	if r.Content == nil {
		delete(fields, "content")
	}
	for _, name := range []string{"details", "structuredContent"} {
		// Read each value when visited: a details hook may replace or mutate
		// the shared structured-content graph before its export.
		value := r.Details
		if name == "structuredContent" {
			value = r.StructuredContent
		}
		raw, err := jsonjs.MarshalOptional(value, name)
		if err != nil {
			return nil, err
		}
		if raw != nil {
			fields[name] = raw
		}
	}
	for name, value := range r.preserved {
		if _, exists := fields[name]; !exists {
			fields[name] = value
		}
	}
	return json.Marshal(fields)
}

// JS tools can return additional result metadata and explicit nulls. Keep only
// fields the typed projection would omit. Dynamic value fields decode into
// the shared live graph and do not need another retained raw copy. Native field updates take
// precedence, and copies of a result share only immutable source metadata.
func (r *ToolResult) UnmarshalJSON(raw []byte) error {
	type plain ToolResult
	*r = ToolResult{}
	if err := json.Unmarshal(raw, (*plain)(r)); err != nil {
		return err
	}
	var values jsonjs.RawObject
	if err := json.Unmarshal(raw, &values); err != nil {
		return err
	}
	var err error
	r.Details, err = jsonjs.DecodeOptional(values["details"])
	if err != nil {
		return err
	}
	r.StructuredContent, err = jsonjs.DecodeOptional(values["structuredContent"])
	if err != nil {
		return err
	}
	normalized, err := json.Marshal(r)
	if err != nil {
		return err
	}
	var source, projected jsonjs.RawObject
	if err := json.Unmarshal(raw, &source); err != nil {
		return err
	}
	if err := json.Unmarshal(normalized, &projected); err != nil {
		return err
	}
	for name := range source {
		if _, exists := projected[name]; exists {
			delete(source, name)
		}
	}
	if len(source) > 0 {
		r.preserved = source
	}
	return nil
}

type BeforeToolCall struct {
	AssistantMessage *ai.Message
	// ToolCall is the original block. Field edits survive into execution and
	// lifecycle events; replacing this hook's pointer does not replace it.
	ToolCall *ai.ContentBlock
	// Args contains the validated *Object, *Array or primitive value. Nested
	// edits are shared with execution and After without revalidation. Assigning
	// a different value only replaces this hook context's field, as in Pi.
	// ToolCall.Arguments remains the separate raw input.
	Args    any
	Context *Context
}

type BeforeToolResult struct {
	Block     bool
	Reason    string
	Terminate bool
}

type AfterToolCall struct {
	BeforeToolCall
	// Result refers to the executed result object. Mutations survive even when
	// After returns nil; a returned override is applied to that mutated value.
	// Replacing this pointer only changes the hook's context, as in Pi.
	Result  *ToolResult
	IsError bool
}

type AfterToolResult = ToolResult

type ToolHooks struct {
	Before func(context.Context, *BeforeToolCall) (*BeforeToolResult, error)
	After  func(context.Context, AfterToolCall) (*AfterToolResult, error)
}

type ToolOutcome struct {
	ToolCall *ai.ContentBlock `json:"toolCall"`
	Result   *ToolResult      `json:"result"`
	IsError  bool             `json:"isError"`
}

// Turn is shared by pointer between FinishTurn and PrepareNextTurn. Its initial
// fields reference the completed message, per-turn tool results, active context
// and accumulating run result. Replacing fields changes this retained view, not
// the loop's own references; context replacement takes effect through TurnUpdate.
// Edits belong to awaited callbacks; concurrent access requires synchronization.
type Turn struct {
	Message     *ai.Message
	ToolResults *MessageList
	Context     *Context
	NewMessages *MessageList
}

type TurnUpdate struct {
	Context *Context
	// Messages stays live until the loop spreads prepared and pending messages.
	Messages      *MessageList
	Model         json.RawMessage
	ThinkingLevel *string
}

type Request struct {
	Context       *Context
	Model         json.RawMessage
	ThinkingLevel string
}

// StreamFn must encode provider failures as error/aborted stream messages.
// A returned Go error is a contract failure and interrupts the low-level loop.
// Models and option fields remain raw to retain provider-specific metadata.
type StreamFn = ai.StreamFn

// RequestAdmission is a Go host extension for orchestration which can select a
// different prepared request before releasing provider events. Context contains
// native messages/tools, not the transformed provider-only transcript.
type RequestAdmission struct {
	Request Request
	Stream  *ai.AssistantMessageEventStream
}

type AdmitRequestFn func(context.Context, Request, map[string]any) (*RequestAdmission, error)

type Config struct {
	Model         json.RawMessage
	Reasoning     string
	Options       map[string]any
	ToolExecution string
	ToolHooks
	ConvertToLLM     func(*MessageList) (*MessageList, error)
	TransformContext func(context.Context, *MessageList) (*MessageList, error)
	GetAPIKey        func(string) (string, error)
	FinishTurn       func(context.Context, *Turn) (string, error)
	PrepareRequest   func(context.Context, Request) (*TurnUpdate, error)
	// AdmitRequest, when supplied by a Go host, owns prepare/transform/convert,
	// credential acquisition and streaming. The nil path follows Pi unchanged.
	// It must return the selected request before any of its events are consumed.
	AdmitRequest        AdmitRequestFn
	PrepareNextTurn     func(*Turn) (*TurnUpdate, error)
	GetSteeringMessages func() (*MessageList, error)
	GetFollowUpMessages func() (*MessageList, error)
	// Now supplies Date.now for deterministic replay and differential testing.
	Now func() int64
	// Preserve an explicit empty level from a wrapper or update, while the
	// native Config zero value still represents omitted reasoning.
	reasoningDefined bool
}

func (c Config) now() int64 {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now().UnixMilli()
}

type Event struct {
	payloadYield          ai.PayloadYield
	payloadAccess         func(func())
	Type                  string
	Message               *ai.Message
	Messages              *MessageList
	ToolResults           *MessageList
	AssistantMessageEvent *ai.AssistantMessageEvent
	ToolCallID            string
	ToolName              string
	Args                  json.RawMessage
	Result                *ToolResult
	PartialResult         *ToolResult
	IsError               bool
}

// Await yields a provider payload lock while a synchronous event sink waits,
// then reacquires it before returning. Do not inspect live message fields in
// work. Non-provider events call work directly. A provider event's Await is
// scoped to its sink invocation and must not be retained for later use.
func (e Event) Await(work func() error) error {
	if e.payloadYield != nil {
		return e.payloadYield(work)
	}
	return work()
}

// Synchronize protects access to a retained provider event's live payload.
// Sinks already hold this lock; use this method only outside the sink callback.
// Like Pi, retaining an event does not detach its nested messages or blocks.
func (e Event) Synchronize(read func()) {
	if e.payloadAccess != nil {
		e.payloadAccess(read)
		return
	}
	read()
}

// MarshalJSON emits exactly the fields belonging to each Pi event variant.
func (e Event) MarshalJSON() ([]byte, error) {
	value := map[string]any{"type": e.Type}
	switch e.Type {
	case "agent_end":
		value["messages"] = messageListOrEmpty(e.Messages)
	case "turn_end":
		value["message"], value["toolResults"] = e.Message, messageListOrEmpty(e.ToolResults)
	case "message_start", "message_end":
		value["message"] = e.Message
	case "message_update":
		value["message"], value["assistantMessageEvent"] = e.Message, e.AssistantMessageEvent
	case "tool_execution_start", "tool_execution_update", "tool_execution_end":
		value["toolCallId"], value["toolName"] = json.RawMessage(jsonjs.QuoteString(e.ToolCallID)), json.RawMessage(jsonjs.QuoteString(e.ToolName))
		if e.Type == "tool_execution_end" {
			value["result"], value["isError"] = e.Result, e.IsError
		} else {
			value["args"] = e.Args
			if e.Type == "tool_execution_update" {
				value["partialResult"] = e.PartialResult
			}
		}
	}
	return json.Marshal(value)
}

// EventSink is awaited. Parallel tools may enter it concurrently; a sink that
// keeps mutable state must synchronize its own writes.
type EventSink func(Event) error
