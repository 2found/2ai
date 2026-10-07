package agentcore

import "encoding/json"

// What a run hands back, and what it emits while running.
//
// These types are the loop's output vocabulary: the terminal RunResult, and
// the StreamEvent increments a live viewer consumes on the way there. They are
// contracts — a consumer renders them and a plugin observes them — so they
// stay separate from the machinery that produces them.

// RunResult is the outcome of a run: the final assistant text, the full message
// history (working memory), the tool trace, and summed usage.
type RunResult struct {
	Final      string      `json:"final"`
	Messages   []Message   `json:"messages"`
	Tools      []ToolTrace `json:"tool_calls"`
	Usage      Usage       `json:"usage"`
	Turns      int         `json:"turns"`
	StopReason string      `json:"stop_reason"`
	// CommandResults contains structured receipts from explicit host controls.
	CommandResults map[string]json.RawMessage `json:"command_results,omitempty"`
	// NativeState and NativeTelemetry retain the original Pi artifacts. When
	// present, Messages is only a display projection and must not seed a run.
	NativeState     json.RawMessage `json:"native_state,omitempty"`
	NativeTelemetry json.RawMessage `json:"native_telemetry,omitempty"`
	NativeRevision  string          `json:"native_revision,omitempty"`
	// UnpersistedEntries counts durable session entries the run buffered but
	// could not commit because the session store kept failing through the final
	// flush. Zero on a healthy (or storeless) run. Non-zero means the durable
	// log holds a valid prefix of the run — recovery stays safe (the
	// conservative resume never re-runs non-idempotent work) but the tail of
	// this run cannot be reconstructed (pi's Faulted state, degraded to a
	// flagged result instead of a halt).
	UnpersistedEntries int `json:"unpersisted_entries,omitempty"`
	// Parked is true when the run ended inside a tool call that is waiting on a
	// human (the ask tool): the call stays dangling in the durable log behind an
	// EntryQuestion, and the run resumes when an EntryAnswer lands. Distinct
	// from every other stop — the run is neither done nor failed, it is paused.
	Parked bool `json:"parked,omitempty"`
	// Question carries the parked call's validated arguments when Parked is
	// true, so a consumer can render the prompt/options directly from RunResult.
	Question json.RawMessage `json:"question,omitempty"`
}

// StreamEventType classifies an incremental event emitted during a streamed run.
type StreamEventType string

const (
	StreamToken    StreamEventType = "token"    // a text fragment of the assistant's answer
	StreamTool     StreamEventType = "tool"     // a completed tool-call trace
	StreamProgress StreamEventType = "progress" // a plain-language progress note (no tool identifier)
	StreamCard     StreamEventType = "card"     // a structured result card (stat | series)

	// Granular lifecycle events (pi's event vocabulary). They are additive: a
	// consumer that only reads token/tool/progress/card keeps working, while an
	// observability layer can reconstruct turn / message / tool-execution
	// boundaries. Emitted only on a streamed run (nil sink => none).
	StreamAgentStart     StreamEventType = "agent_start"           // run begins
	StreamTurnStart      StreamEventType = "turn_start"            // a reasoning turn begins
	StreamMessageStart   StreamEventType = "message_start"         // assistant message begins (before tokens)
	StreamMessageEnd     StreamEventType = "message_end"           // assistant message complete
	StreamToolExecStart  StreamEventType = "tool_execution_start"  // a tool call begins
	StreamToolExecUpdate StreamEventType = "tool_execution_update" // a streaming tool's partial output (P8)
	StreamToolExecEnd    StreamEventType = "tool_execution_end"    // a tool call finished (carries the trace)
	StreamTurnEnd        StreamEventType = "turn_end"              // the turn (reason + act) is complete
	StreamSavePoint      StreamEventType = "save_point"            // a turn's buffered durable writes were flushed atomically
	StreamAgentEnd       StreamEventType = "agent_end"             // run ends (any exit path)
	// StreamQuestion carries a parked call's validated arguments: a tool asked
	// the human a structured question and the run is now waiting on the answer.
	StreamQuestion StreamEventType = "question"
)

// ResultCard is a compact, structured answer artifact a consumer may attach to a
// streamed turn so the UI can render a stat block or a small chart instead of
// prose alone. It is deliberately product-agnostic: a title, a kind, and either
// a few stat rows or a short series of points. agentcore never builds one — a
// consumer (e.g. the orchestrator) emits it via the sink; the type lives here so
// the stream vocabulary is shared.
type ResultCard struct {
	Title  string      `json:"title"`
	Kind   string      `json:"kind"`           // "stat" | "series"
	Unit   string      `json:"unit,omitempty"` // optional label for the values
	Stats  []CardStat  `json:"stats,omitempty"`
	Points []CardPoint `json:"points,omitempty"`
}

// CardStat is one labeled metric row in a "stat" card.
type CardStat struct {
	Label string `json:"label"`
	Value string `json:"value"`
}

// CardPoint is one point in a "series" card (e.g. a time bucket count).
type CardPoint struct {
	Label string  `json:"label"`
	Value float64 `json:"value"`
}

// StreamEvent is one increment surfaced to a live viewer (the SSE chat endpoint)
// while a run is in flight. Callbacks fire from the run goroutine, in order.
// Token/Tool/ToolExecUpdate are emitted by the core loop; Progress/Card are
// emitted by a consumer wrapping the loop (the core never sets them).
type StreamEvent struct {
	Type     StreamEventType
	Token    string          // set when Type == StreamToken
	Tool     *ToolTrace      // set when Type == StreamTool
	Note     string          // set for StreamProgress and textual tool updates
	Card     *ResultCard     // set when Type == StreamCard
	Question json.RawMessage // set when Type == StreamQuestion (the parked call's args)
	Turn     int
}

// StreamSink receives display events during a run. A nil sink suppresses
// these notifications; native provider execution still consumes its stream.
type StreamSink func(StreamEvent)
