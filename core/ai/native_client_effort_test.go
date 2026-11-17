package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The host names the model and its effort, so xhigh and max reach the wire;
// before, the option builder lowered both to high for every native model.
func TestNativeCandidateSendsTheHostsEffort(t *testing.T) {
	var effort string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Effort string `json:"reasoning_effort"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		effort = body.Effort
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: "+`{"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	client, err := NewNativeClient(ClientSpec{Name: "openai", APIKey: "key", BaseURL: server.URL, Compat: DefaultCompat()})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := client.Candidate("gpt-6-sol", 1_000_000)
	if err != nil {
		t.Fatal(err)
	}
	transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hi")}}})
	for _, level := range []string{"low", "medium", "high", "xhigh", "max"} {
		effort = ""
		stream, err := candidate.Stream(context.Background(), candidate.Model, transcript, map[string]any{"reasoning": level, "maxTokens": 1024})
		if err != nil {
			t.Fatal(err)
		}
		if result, err := stream.Result(context.Background()); err != nil || result.StopReason == "error" {
			t.Fatalf("%s: %+v %v", level, result, err)
		}
		if effort != level {
			t.Fatalf("asked %s, sent %q", level, effort)
		}
	}
}
