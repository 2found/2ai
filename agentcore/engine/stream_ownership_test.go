package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiStreamOwnership(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-stream-ownership.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Phase, Field string
			Expected     json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 12 {
		t.Fatal("unexpected ownership oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Phase+"/"+tc.Field, func(t *testing.T) {
			assistant := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "original"}), API: "test", Provider: "test", Model: "m", Usage: &ai.Usage{Input: 1}, StopReason: "stop", Timestamp: 1}
			retained := []*ai.Message{}
			finalSame := false
			mutate := func(phase string, message *ai.Message) {
				if phase != tc.Phase {
					return
				}
				switch tc.Field {
				case "timestamp":
					message.Timestamp = 77
				case "text":
					message.Content.Blocks.Get(0).Text = "changed"
				case "usage":
					message.Usage.Input = 77
				}
			}
			summary := func(message *ai.Message) map[string]any {
				return map[string]any{"timestamp": message.Timestamp, "text": message.Content.Blocks.Get(0).Text, "input": message.Usage.Input}
			}
			config := engine.Config{Model: json.RawMessage(`{"api":"test","provider":"test","id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }, FinishTurn: func(_ context.Context, turn *engine.Turn) (string, error) {
				mutate("finish", turn.Message)
				return "end", nil
			}}
			messages, err := engine.Run(context.Background(), engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{}, config, func(event engine.Event) error {
				if event.Message != nil && event.Message.Role == "assistant" {
					retained = append(retained, event.Message)
					mutate(event.Type, event.Message)
					if event.Type == "message_end" {
						finalSame = event.Message == assistant
					}
				}
				return nil
			}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
				s := ai.NewAssistantMessageEventStream()
				s.Push(ai.AssistantMessageEvent{Type: "start", Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "original", Partial: assistant})
				s.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: assistant})
				s.End()
				return s, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			var result *ai.Message
			for _, message := range messages.Values() {
				if message.Role == "assistant" {
					result = message
				}
			}
			observed := []map[string]any{}
			for _, message := range retained {
				observed = append(observed, summary(message))
			}
			actual, err := json.Marshal(map[string]any{"provider": summary(assistant), "result": summary(result), "retained": observed, "finalSame": finalSame, "resultSame": result == assistant})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestStreamListenerCanAwaitProvider(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	stream := ai.NewAssistantMessageEventStream()
	message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "original"}), StopReason: "pending", Usage: &ai.Usage{}}
	stream.Push(ai.AssistantMessageEvent{Type: "start", Partial: message})
	start, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		<-start
		stream.Synchronize(func() {
			message.Content.Blocks.Get(0).Text += "/provider"
			message.StopReason = "stop"
			stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
		})
		stream.End()
	}()
	var retained engine.Event
	_, err := engine.Run(ctx, engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{}, engine.Config{Model: json.RawMessage(`{"id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }}, func(event engine.Event) error {
		if event.Type == "message_start" && event.Message.Role == "assistant" {
			retained = event
			event.Message.Content.Blocks.Get(0).Text = "listener"
			close(start)
			if err := event.Await(func() error { _, err := stream.Result(ctx); return err }); err != nil {
				return err
			}
			if event.Message.Content.Blocks.Get(0).Text != "listener/provider" {
				t.Error("provider/listener lost shared block")
			}
		}
		return nil
	}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		return stream, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	<-finished
	retained.Synchronize(func() {
		if retained.Message.Content.Blocks.Get(0).Text != "listener/provider" {
			t.Error("retained event lost provider edits")
		}
	})
	if err := retained.Await(func() error { t.Error("expired event awaited work"); return nil }); err == nil {
		t.Fatal("expired event yield accepted")
	}
}

func TestLoopReadsTerminalMessageUnderProviderLock(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	stream := ai.NewAssistantMessageEventStream()
	message := &ai.Message{Role: "assistant", Content: ai.BlockContent(), StopReason: "stop", Usage: &ai.Usage{}}
	stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
	release, written := make(chan struct{}), make(chan struct{})
	go func() {
		<-release
		stream.Synchronize(func() { message.StopReason = "stop" })
		close(written)
		stream.End()
	}()
	_, err := engine.Run(ctx, engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{}, engine.Config{
		Model: json.RawMessage(`{"id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil },
		FinishTurn: func(context.Context, *engine.Turn) (string, error) {
			select {
			case <-written:
				return "end", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	}, func(event engine.Event) error {
		if event.Type == "message_end" && event.Message.Role == "assistant" {
			close(release)
		}
		return nil
	}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		return stream, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestLoopTerminalEventCanAwaitLateProviderWork(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	stream := ai.NewAssistantMessageEventStream()
	message := &ai.Message{Role: "assistant", Content: ai.BlockContent(), StopReason: "stop", Usage: &ai.Usage{Input: 1}}
	stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
	release := make(chan struct{})
	go func() { <-release; stream.Synchronize(func() { message.Usage.Input = 2 }); stream.End() }()
	_, err := engine.Run(ctx, engine.NewList(&ai.Message{Role: "user", Content: ai.TextContent("run")}), engine.Context{}, engine.Config{Model: json.RawMessage(`{"id":"m"}`), ConvertToLLM: func(m *engine.MessageList) (*engine.MessageList, error) { return m, nil }}, func(event engine.Event) error {
		if event.Type == "agent_end" {
			close(release)
			if err := event.Await(func() error { return stream.WaitForEnd(ctx) }); err != nil {
				return err
			}
			if event.Messages.Get(event.Messages.Len()-1) != message || message.Usage.Input != 2 {
				t.Error("terminal event lost late provider work")
			}
		}
		return nil
	}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		return stream, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
