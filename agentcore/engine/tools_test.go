package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestProgrammaticCallUsesExplicitToolsAndScopedUpdates(t *testing.T) {
	var retained func(*engine.ToolResult)
	updates := 0
	before, after := false, false
	tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}`)}}
	tool.Execute = func(_ context.Context, id string, args any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
		if id != "call" || argumentJSON(t, args) != `{"count":42}` {
			t.Errorf("execution arguments: %s %s", id, args)
		}
		retained = update
		update(&engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: "partial"})})
		return &engine.ToolResult{}, errors.New("tool failed")
	}
	current := &engine.Context{} // Explicit nested-call tools need not be in context.Tools.
	call := ai.ContentBlock{Type: "toolCall", Name: "echo", ID: "call", Arguments: json.RawMessage(`{"count":"42"}`)}
	result, err := engine.RunToolCall(context.Background(), &call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{Role: "assistant"}, current, engine.ToolHooks{
		Before: func(_ context.Context, hook *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
			before = true
			if hook.Context != current || string(hook.ToolCall.Arguments) != `{"count":"42"}` || argumentJSON(t, hook.Args) != `{"count":42}` {
				t.Error("hook lost raw/prepared arguments or context")
			}
			return nil, nil
		},
		After: func(_ context.Context, hook engine.AfterToolCall) (*engine.AfterToolResult, error) {
			after = true
			if !hook.IsError || hook.Result.Content.Get(0).Text != "tool failed" {
				t.Error("after hook missed executed failure")
			}
			return nil, nil
		},
	}, func(*engine.ToolResult) error { updates++; return nil })
	if err != nil || !result.IsError || !before || !after || updates != 1 {
		t.Fatalf("outcome=%+v err=%v hooks=%v/%v updates=%d", result, err, before, after, updates)
	}
	retained(&engine.ToolResult{})
	if updates != 1 {
		t.Fatal("late update was accepted")
	}
}

func TestToolWaitsForAdmittedUpdates(t *testing.T) {
	entered, release, executed, done := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}}
	tool.Execute = func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
		go update(&engine.ToolResult{})
		<-entered
		close(executed)
		return &engine.ToolResult{}, nil
	}
	go func() {
		defer close(done)
		_, err := engine.RunToolCall(context.Background(), &ai.ContentBlock{Name: "echo", Arguments: json.RawMessage(`{}`)}, engine.NewList([]*engine.Tool{tool}...), nil, &engine.Context{}, engine.ToolHooks{}, func(*engine.ToolResult) error { close(entered); <-release; return nil })
		if err != nil {
			t.Error(err)
		}
	}()
	<-executed
	select {
	case <-done:
		t.Fatal("tool finalized while an admitted update was unsettled")
	default:
	}
	close(release)
	<-done
}

func TestParallelSinkFailureRejectsWithoutResultMessages(t *testing.T) {
	blocked, release, siblingDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	tool := &engine.Tool{Tool: ai.Tool{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}}
	tool.Execute = func(_ context.Context, id string, _ any, _ func(*engine.ToolResult)) (*engine.ToolResult, error) {
		if id == "one" {
			close(blocked)
			<-release
		} else {
			<-blocked
		}
		return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)}, nil
	}
	failure := errors.New("end sink failed")
	var mu sync.Mutex
	resultMessages := 0
	sink := func(event engine.Event) error {
		if event.Type == "message_start" && event.Message.Role == "toolResult" {
			mu.Lock()
			resultMessages++
			mu.Unlock()
		}
		if event.Type == "tool_execution_end" {
			if event.ToolCallID == "two" {
				return failure
			}
			close(siblingDone)
		}
		return nil
	}
	stream := func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		message := &ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(
			ai.ContentBlock{Type: "toolCall", ID: "one", Name: "echo", Arguments: json.RawMessage(`{}`)},
			ai.ContentBlock{Type: "toolCall", ID: "two", Name: "echo", Arguments: json.RawMessage(`{}`)},
		)}
		result := ai.NewAssistantMessageEventStream()
		result.Push(ai.AssistantMessageEvent{Type: "done", Message: message})
		return result, nil
	}
	done := make(chan error, 1)
	go func() {
		_, err := engine.Run(context.Background(), nil, engine.Context{Tools: engine.NewList([]*engine.Tool{tool}...)}, engine.Config{ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}, sink, stream)
		done <- err
	}()
	select {
	case err := <-done:
		if err != failure {
			t.Errorf("failure identity: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Error("loop waited for a sibling after Promise.all rejection")
	}
	close(release)
	<-siblingDone
	mu.Lock()
	defer mu.Unlock()
	if resultMessages != 0 {
		t.Fatalf("published %d result messages from a failed batch", resultMessages)
	}
}

func TestDefaultStreamBinding(t *testing.T) {
	engine.SetDefaultStreamFn(nil)
	t.Cleanup(func() { engine.SetDefaultStreamFn(nil) })
	if _, err := engine.GetDefaultStreamFn(); err == nil {
		t.Fatal("missing default accepted")
	}
	called := false
	engine.SetDefaultStreamFn(func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
		called = true
		stream := ai.NewAssistantMessageEventStream()
		stream.End(&ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent()})
		return stream, nil
	})
	_, err := engine.Run(context.Background(), nil, engine.Context{}, engine.Config{ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }}, func(engine.Event) error { return nil }, nil)
	if err != nil || !called {
		t.Fatalf("default stream: called=%v err=%v", called, err)
	}
}
