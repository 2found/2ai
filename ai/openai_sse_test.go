package ai

import (
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// Some OpenAI-compatible gateways return text/event-stream for a non-streaming
// Chat() request (observed with 9router routing summarizer calls to a cheap
// streamed model). Chat() must fold those SSE frames into one protocol.ChatResponse
// rather than failing the JSON decode and degrading the caller to elide.
func TestDecodeSSEResponse_FoldsContentAndUsage(t *testing.T) {
	body := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n" +
		"data: [DONE]\n\n"
	p := &OpenAIProvider{}
	resp, err := decodeSSEResponse(p, nil, []byte(body))
	if err != nil {
		t.Fatalf("decodeSSEResponse: %v", err)
	}
	if resp.Message.Content != "Hello world" {
		t.Fatalf("content = %q, want %q", resp.Message.Content, "Hello world")
	}
	if resp.StopReason != "stop" {
		t.Fatalf("stop reason = %q, want stop", resp.StopReason)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 2 {
		t.Fatalf("usage = %+v, want in=11 out=2", resp.Usage)
	}
}

// Tool-call deltas arrive fragmented across SSE chunks keyed by index; they must
// be reassembled into whole ToolCalls.
func TestDecodeSSEResponse_ReassemblesToolCalls(t *testing.T) {
	body := "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.txt\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	p := &OpenAIProvider{}
	resp, err := decodeSSEResponse(p, nil, []byte(body))
	if err != nil {
		t.Fatalf("decodeSSEResponse: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_1" || tc.Name != "read_file" || tc.Arguments != `{"path":"a.txt"}` {
		t.Fatalf("tool call = %+v", tc)
	}
}

// A gateway that writes its 200 and the opening role frame, then drops the
// connection, produced a stream with no content, no tool calls and no
// finish_reason. That folded into a well-formed empty ChatResponse, so the agent
// loop saw a model that had simply chosen to say nothing and stopped at turn 1.
// The body here is the exact shape observed from the live gateway.
func TestDecodeSSEResponse_TruncatedStreamIsRetryableError(t *testing.T) {
	body := "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
	p := &OpenAIProvider{}
	_, err := decodeSSEResponse(p, nil, []byte(body))
	if err == nil {
		t.Fatal("truncated SSE stream decoded as a successful empty response")
	}
	// It must be retryable: the stream broke in transit, so the next attempt may
	// well succeed. A non-retryable error here silently drops the turn.
	if !protocol.IsRetryable(err) {
		t.Fatalf("truncated stream error is not retryable: %v", err)
	}
}

// The truncation guard keys on "nothing usable arrived", not on finish_reason
// alone: a gateway that omits finish_reason but still returns content has given
// us a real answer and must not be turned into an error.
func TestDecodeSSEResponse_ContentWithoutFinishReasonStillDecodes(t *testing.T) {
	body := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"
	p := &OpenAIProvider{}
	resp, err := decodeSSEResponse(p, nil, []byte(body))
	if err != nil {
		t.Fatalf("decodeSSEResponse: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Fatalf("content = %q, want %q", resp.Message.Content, "Hello")
	}
}

func TestIsSSEResponse(t *testing.T) {
	cases := []struct {
		ct   string
		data string
		want bool
	}{
		{"text/event-stream; charset=utf-8", "data: {}", true},
		{"application/json", "data: {\"x\":1}", true},
		{"application/json", "{\"choices\":[]}", false},
		{"", "  data: {}", true},
	}
	for _, c := range cases {
		if got := isSSEResponse(c.ct, []byte(c.data)); got != c.want {
			t.Errorf("isSSEResponse(%q, %q) = %v, want %v", c.ct, c.data, got, c.want)
		}
	}
}
