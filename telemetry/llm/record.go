// Package llm defines the export format for model-call telemetry.
// It contains no provider wrappers or agent plugins.
package llm

import (
	"context"
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// traceCtxKey carries an opaque correlation id down the context into the
// provider. agentcore stays generic — it never learns what the id *means* (the
// consumer maps it run → agent); it only stamps it onto every emitted record so
// a trace stream can be attributed after the fact.
type traceCtxKey struct{}

// WithTraceID tags ctx with a correlation id the host stamps onto
// every TraceRecord produced under it. The consumer (e.g. the Runner) sets it to
// its run id just before driving the loop; an empty id is a no-op.
func WithTraceID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, traceCtxKey{}, id)
}

// TraceIDFrom reads the correlation id set by WithTraceID ("" when absent, e.g.
// a classifier call made outside any run).
func TraceIDFrom(ctx context.Context) string {
	if v, ok := ctx.Value(traceCtxKey{}).(string); ok {
		return v
	}
	return ""
}

// TraceRecord is one observed LLM call: what was sent, what came back, what it
// cost, and how long it took. It is the durable, debuggable unit behind an agent
// run — the "message sent to the LLM + est. fee" trace. Emitted once per Chat or
// streamed turn.
type TraceRecord struct {
	// NativeTrace holds the authoritative native request, response, and spans.
	// Messages and Response remain a display projection for legacy consumers.
	NativeTrace json.RawMessage `json:"native_trace,omitempty"`
	TraceID     string          `json:"trace_id,omitempty"` // correlation id (the run id), set via WithTraceID
	// SessionKey identifies WHICH agent made the call — the run's own session,
	// or a spawned child's derived session. A parent and its children share one
	// provider and one ctx chain, so without this their calls are one
	// undifferentiated stream. Empty outside a durable run.
	SessionKey string `json:"session_key,omitempty"`
	// Depth is the delegation depth of the caller: 0 for the top-level run.
	Depth     int                `json:"depth,omitempty"`
	Timestamp time.Time          `json:"timestamp"`
	Provider  string             `json:"provider"`
	Model     string             `json:"model"`
	Messages  []protocol.Message `json:"messages"`           // the request sent to the model
	Tools     []string           `json:"tools,omitempty"`    // tool names advertised this turn
	Response  string             `json:"response,omitempty"` // assistant text returned
	// ReasoningBlocks carries opaque provider replay state (for example,
	// Anthropic signed thinking). It is recorded separately from Response so it
	// can be replayed without ever rendering it as user-visible assistant text.
	ReasoningBlocks []protocol.ReasoningBlock `json:"reasoning_blocks,omitempty"`
	ToolCalls       []protocol.ToolCall       `json:"tool_calls,omitempty"` // tool calls the model requested
	StopReason      string                    `json:"stop_reason,omitempty"`
	Usage           protocol.Usage            `json:"usage"` // tokens + computed CostUSD
	LatencyMS       int64                     `json:"latency_ms"`
	Streamed        bool                      `json:"streamed"`
	Err             string                    `json:"error,omitempty"`
}

// Sink receives a TraceRecord per LLM call. Implementations fan out to a
// file, stdout, or an analytics store. It never computes or changes usage.
type Sink interface {
	Record(TraceRecord)
}

// SinkFunc adapts a function to a Sink.
type SinkFunc func(TraceRecord)

func (f SinkFunc) Record(r TraceRecord) { f(r) }

// FileSink appends one JSON object per line (JSONL) to an open file. Safe
// for concurrent runs. Close it when the process shuts down.
type FileSink struct {
	mu  sync.Mutex
	w   *os.File
	enc *json.Encoder
}

// NewFileSink opens (creating/appending) a JSONL trace file at path.
func NewFileSink(path string) (*FileSink, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &FileSink{w: f, enc: json.NewEncoder(f)}, nil
}

// Record writes one JSONL line. Encoding/IO errors are dropped — tracing must
// never break a run.
func (s *FileSink) Record(r TraceRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = s.enc.Encode(r)
}

// Close releases the underlying file.
func (s *FileSink) Close() error { return s.w.Close() }

// MultiSink fans one record out to several sinks (e.g. file + stdout).
type MultiSink []Sink

func (m MultiSink) Record(r TraceRecord) {
	for _, s := range m {
		if s != nil {
			s.Record(r)
		}
	}
}
