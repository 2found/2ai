package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiPublicLoopStreamOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/loop-stream.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          json.RawMessage
		Assistant      ai.Message
		Now            int64
		Cases          []struct {
			Input struct {
				Name                                                                          string
				Resume, Empty, AssistantTail, Custom, Missing, UseDefault, OmitTerminal, Tool bool
				StopReason, ProviderFailure, TransformFailure                                 string
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 13 {
		t.Fatal("unexpected loop stream oracle coverage/revision")
	}
	defer engine.SetDefaultStreamFn(nil)
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			events := []engine.Event{}
			requests := []map[string]any{}
			config := engine.Config{Model: fixture.Model, Now: func() int64 { return fixture.Now }, ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) {
				copy := engine.MessagePointers(engine.MessageValues(messages.Values()))
				for i := range copy {
					if copy[i].Role == "custom" {
						copy[i].Role = "user"
					}
				}
				return engine.NewList(copy...), nil
			}}
			if tc.Input.TransformFailure != "" {
				config.TransformContext = func(context.Context, *engine.MessageList) (*engine.MessageList, error) {
					return nil, errors.New(tc.Input.TransformFailure)
				}
			}
			tools := []*engine.Tool{}
			if tc.Input.Tool {
				tools = append(tools, &engine.Tool{Tool: ai.Tool{Name: "echo", Description: "Echo", Parameters: json.RawMessage(`{"type":"object"}`)}, Label: "Echo", Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
					update(&engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "working"})})
					terminate := true
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "done"}), Terminate: &terminate}, nil
				}})
			}
			provider := engine.StreamFn(func(_ context.Context, model json.RawMessage, transcript ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
				requests = append(requests, map[string]any{"model": model, "context": transcript, "options": options})
				if tc.Input.ProviderFailure != "" {
					return nil, errors.New(tc.Input.ProviderFailure)
				}
				message := fixture.Assistant
				if tc.Input.StopReason != "" {
					message.StopReason = tc.Input.StopReason
					failure := "failed"
					message.ErrorMessage = &failure
				}
				if tc.Input.Tool {
					message.StopReason = "toolUse"
					message.Content = ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)})
				}
				stream := ai.NewAssistantMessageEventStream()
				if !tc.Input.OmitTerminal {
					stream.Push(ai.AssistantMessageEvent{Type: "start", Partial: &message})
					if message.StopReason == "error" || message.StopReason == "aborted" {
						stream.Push(ai.AssistantMessageEvent{Type: "error", Reason: message.StopReason, Error: &message})
					} else {
						stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: message.StopReason, Message: &message})
					}
				}
				stream.End(&message)
				return stream, nil
			})
			engine.SetDefaultStreamFn(nil)
			if tc.Input.UseDefault {
				engine.SetDefaultStreamFn(provider)
			}
			if tc.Input.UseDefault || tc.Input.Missing {
				provider = nil
			}
			user := ai.Message{Role: "user", Content: ai.TextContent("go"), Timestamp: fixture.Now}
			messages := []*ai.Message{&user}
			if tc.Input.Empty {
				messages = nil
			} else if tc.Input.AssistantTail {
				messages = []*ai.Message{&fixture.Assistant}
			} else if tc.Input.Custom {
				messages[0].Role = "custom"
			}
			var stream *engine.AgentEventStream
			var syncError, producerError *string
			if tc.Input.Resume {
				stream, err = engine.AgentLoopContinue(ctx, engine.Context{Messages: engine.NewList(messages...), Tools: engine.NewList(tools...)}, config, provider)
			} else {
				stream = engine.AgentLoop(ctx, engine.NewList([]*ai.Message{&user}...), engine.Context{Tools: engine.NewList(tools...)}, config, provider)
				err = nil
			}
			if err != nil {
				failure := err.Error()
				syncError = &failure
			}
			var result *engine.MessageList
			resultPending, queuePending := false, false
			if stream != nil {
				// Pi queues the first event synchronously. A cancelled reader must still
				// receive it immediately; waiting for producer completion would hide this.
				canceled, stop := context.WithCancel(ctx)
				stop()
				first, ok, err := stream.Next(canceled)
				if err != nil || !ok || first.Type != "agent_start" {
					t.Fatalf("missing synchronous admission: %+v %v", first, err)
				}
				events = append(events, first)
				if err := stream.Wait(ctx); err != nil {
					failure := err.Error()
					producerError = &failure
				} else {
					result, err = stream.Result(ctx)
					if err != nil {
						t.Fatal(err)
					}
				}
				reader, stop := context.WithTimeout(ctx, 5*time.Millisecond)
				defer stop()
				for {
					event, ok, err := stream.Next(reader)
					if errors.Is(err, context.DeadlineExceeded) {
						queuePending = true
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					if !ok {
						break
					}
					events = append(events, event)
				}
				if producerError != nil {
					reader, stop := context.WithTimeout(ctx, 5*time.Millisecond)
					_, err := stream.Result(reader)
					stop()
					resultPending = errors.Is(err, context.DeadlineExceeded)
					stream.End(engine.NewList[*ai.Message]())
				}
			}
			actual, err := json.Marshal(map[string]any{"events": events, "requests": requests, "result": result, "syncError": syncError, "producerError": producerError, "resultPending": resultPending, "queuePending": queuePending})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			_ = json.Unmarshal(actual, &got)
			_ = json.Unmarshal(tc.Expected, &want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("public stream mismatch\nGo: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

func TestAgentLoopStreamDoesNotWaitForReaders(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	providerStarted := make(chan struct{})
	release := make(chan struct{})
	releaseProvider := sync.OnceFunc(func() { close(release) })
	defer releaseProvider()
	config := engine.Config{Model: json.RawMessage(`{"id":"test"}`), ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}
	stream := engine.AgentLoop(ctx, engine.NewList([]*ai.Message{{Role: "user", Content: ai.TextContent("go")}}...), engine.Context{}, config, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		close(providerStarted)
		<-release
		response := ai.NewAssistantMessageEventStream()
		message := &ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), StopReason: "stop"}
		response.Push(ai.AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
		response.End()
		return response, nil
	})
	select {
	case <-providerStarted:
	case <-ctx.Done():
		t.Fatal("producer waited for an event reader")
	}
	waiting, stop := context.WithCancel(ctx)
	stop()
	if err := stream.Wait(waiting); !errors.Is(err, context.Canceled) {
		t.Fatalf("wait cancellation: %v", err)
	}
	releaseProvider()
	result, err := stream.Result(ctx)
	if err != nil || result.Len() != 2 {
		t.Fatalf("result without draining: %v %v", result, err)
	}
	if err := stream.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	count := 0
	for {
		event, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		count++
		if event.Type == "agent_end" && event.Messages.Len() != result.Len() {
			t.Fatal("terminal result differs")
		}
	}
	if count < 6 {
		t.Fatal("result consumed queued lifecycle")
	}
}

func TestAgentLoopStreamReportsPanickingProducerWithoutInventingEnd(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	original := errors.New("hook panic")
	stream := engine.AgentLoop(ctx, nil, engine.Context{}, engine.Config{GetSteeringMessages: func() (*engine.MessageList, error) { panic(original) }}, func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		t.Error("unexpected provider")
		return nil, nil
	})
	if err := stream.Wait(ctx); err != original {
		t.Fatalf("panic identity lost: %v", err)
	}
	reader, stop := context.WithTimeout(ctx, 5*time.Millisecond)
	defer stop()
	for {
		event, ok, err := stream.Next(reader)
		if errors.Is(err, context.DeadlineExceeded) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Fatal("panic closed the event queue")
		}
		if event.Type == "agent_end" {
			t.Fatal("panic fabricated a completed run")
		}
	}
}
