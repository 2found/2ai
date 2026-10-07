package engine_test

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiStreamNumbers(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-stream-numbers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Mode, Stage, Kind string }
			Expected json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 24 {
		t.Fatal("unexpected stream number coverage")
	}
	classify := func(n float64) string {
		if math.IsNaN(n) {
			return "NaN"
		}
		if math.IsInf(n, 1) {
			return "Infinity"
		}
		if math.IsInf(n, -1) {
			return "-Infinity"
		}
		if n == 0 && math.Signbit(n) {
			return "-0"
		}
		return strconv.FormatFloat(n, 'g', -1, 64)
	}
	usage := func(n float64) *ai.Usage {
		return &ai.Usage{Input: n, Output: n, CacheRead: n, CacheWrite: n, CacheWrite1h: &n, Reasoning: &n, TotalTokens: n, Cost: ai.UsageCost{Input: n, Output: n, CacheRead: n, CacheWrite: n, Total: n}}
	}
	describe := func(message *ai.Message) map[string]any {
		values := map[string]any{}
		if u := message.Usage; u != nil {
			values = map[string]any{"input": classify(u.Input), "output": classify(u.Output), "cacheRead": classify(u.CacheRead), "cacheWrite": classify(u.CacheWrite), "totalTokens": classify(u.TotalTokens), "cost": map[string]string{"input": classify(u.Cost.Input), "output": classify(u.Cost.Output), "cacheRead": classify(u.Cost.CacheRead), "cacheWrite": classify(u.Cost.CacheWrite), "total": classify(u.Cost.Total)}}
			if u.CacheWrite1h != nil {
				values["cacheWrite1h"] = classify(*u.CacheWrite1h)
			}
			if u.Reasoning != nil {
				values["reasoning"] = classify(*u.Reasoning)
			}
		}
		wire, err := json.Marshal(message.Usage)
		if err != nil {
			wire = []byte("ERROR: " + err.Error())
		}
		return map[string]any{"stopReason": message.StopReason, "usage": values, "wire": string(wire)}
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Mode+"/"+tc.Input.Stage+"/"+tc.Input.Kind, func(t *testing.T) {
			n, err := strconv.ParseFloat(tc.Input.Kind, 64)
			if err != nil {
				t.Fatal(err)
			}
			message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), API: "test", Provider: "test", Model: "test", Usage: usage(n), StopReason: "stop", Timestamp: 1}
			final := message
			if tc.Input.Stage == "ignored" {
				copy := *message
				copy.Usage = usage(0)
				final = &copy
			}
			stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				if tc.Input.Stage == "updates" {
					s.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
				}
				if tc.Input.Stage != "terminal" {
					s.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "done", Partial: message})
				}
				if tc.Input.Stage == "ignored" {
					s.Push(ai.AssistantMessageEvent{Type: "vendor", Partial: message})
				}
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: final})
				return s, nil
			}
			events, completed := []map[string]any{}, []map[string]any{}
			emit := func(event engine.Event) error {
				value := map[string]any{"type": event.Type}
				if event.Message != nil && event.Message.Role == "assistant" {
					value["message"] = describe(event.Message)
				}
				events = append(events, value)
				if event.Type == "agent_end" {
					for _, m := range event.Messages.Values() {
						if m.Role == "assistant" {
							completed = append(completed, describe(m))
						}
					}
				}
				return nil
			}
			model := json.RawMessage(`{"id":"test","api":"test","provider":"test"}`)
			prompt := &ai.Message{Role: "user", Content: ai.TextContent("go")}
			if tc.Input.Mode == "loop" {
				_, err = engine.Run(context.Background(), engine.NewList([]*ai.Message{prompt}...), engine.Context{}, engine.Config{Model: model, ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }}, emit, stream)
			} else {
				agent, createErr := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: model}, AgentConfig: engine.AgentConfig{StreamFn: stream}})
				if createErr != nil {
					t.Fatal(createErr)
				}
				agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error { return emit(event) }})
				err = agent.Prompt(context.Background(), prompt)
			}
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(map[string]any{"events": events, "completed": completed})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err = json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
