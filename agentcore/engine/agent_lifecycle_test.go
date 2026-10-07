package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func completedAgentStream(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
	stream := ai.NewAssistantMessageEventStream()
	stream.End(&ai.Message{Role: "assistant", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"}), StopReason: "stop"})
	return stream, nil
}

func newAgentForTest(t *testing.T, config engine.AgentConfig) *engine.Agent {
	t.Helper()
	if config.StreamFn == nil {
		config.StreamFn = completedAgentStream
	}
	agent, err := engine.NewAgent(engine.AgentOptions{AgentConfig: config})
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

func TestAgentWaitsForEndSubscribersAndKeepsSignal(t *testing.T) {
	agent := newAgentForTest(t, engine.AgentConfig{})
	entered, release := make(chan struct{}), make(chan struct{})
	var signal context.Context
	observations := []string{}
	agent.Subscribe(&engine.Listener{Handle: func(ctx context.Context, event engine.Event) error {
		if event.Type != "agent_end" {
			return nil
		}
		signal = ctx
		if !agent.State().IsStreaming || agent.Signal() != ctx {
			t.Error("agent became idle before end subscribers settled")
		}
		observations = append(observations, "first entered")
		close(entered)
		<-release
		observations = append(observations, "first settled")
		return nil
	}})
	agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Type == "agent_end" {
			observations = append(observations, "second")
		}
		return nil
	}})
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	done := make(chan error, 1)
	go func() { done <- agent.Prompt(parent, "go") }()
	<-entered
	select {
	case <-done:
		t.Error("Prompt returned while end subscriber was active")
	default:
	}
	if err := agent.Prompt(context.Background(), "busy"); err == nil {
		t.Error("concurrent prompt admitted")
	}
	if err := agent.Continue(context.Background()); err == nil {
		t.Error("concurrent continuation admitted")
	}
	if err := agent.Reset(); err == nil {
		t.Error("concurrent reset admitted")
	}
	waitContext, cancelWait := context.WithCancel(context.Background())
	cancelWait()
	if err := agent.WaitForIdle(waitContext); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled wait: %v", err)
	}
	if signal.Err() != nil {
		t.Error("cancelling a waiter aborted the run")
	}
	waited := make(chan error, 1)
	go func() { waited <- agent.WaitForIdle(context.Background()) }()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-waited; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(observations, []string{"first entered", "first settled", "second"}) {
		t.Fatalf("subscriber ordering: %v", observations)
	}
	if agent.State().IsStreaming || agent.Signal() != nil || agent.State().StreamingMessage != nil {
		t.Fatal("agent did not settle")
	}
	cancelParent()
	if signal.Err() != nil {
		t.Fatal("successful run's saved signal was spuriously aborted")
	}
	if err := agent.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestAgentListenersUseLiveIdentitySet(t *testing.T) {
	agent := newAgentForTest(t, engine.AgentConfig{})
	order := []string{}
	third := &engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Type == "agent_start" {
			order = append(order, "third")
		}
		return nil
	}}
	second := &engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Type == "agent_start" {
			order = append(order, "second")
		}
		return nil
	}}
	var removeSecond func()
	first := &engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
		if event.Type == "agent_start" {
			order = append(order, "first")
			removeSecond()
			agent.Subscribe(third)
		}
		return nil
	}}
	removeFirst := agent.Subscribe(first)
	agent.Subscribe(first) // Same listener identity is not a second subscription.
	removeSecond = agent.Subscribe(second)
	if err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"first", "third"}) {
		t.Fatalf("live listener set: %v", order)
	}
	removeFirst()
	order = nil
	if err := agent.Prompt(context.Background(), "again"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"third"}) {
		t.Fatalf("unsubscribe: %v", order)
	}
}

func TestAgentAbortAndConcurrentQueueAccess(t *testing.T) {
	providerEntered := make(chan struct{})
	var observed context.Context
	agent := newAgentForTest(t, engine.AgentConfig{StreamFn: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		observed = ctx
		close(providerEntered)
		<-ctx.Done()
		message := "aborted"
		stream := ai.NewAssistantMessageEventStream()
		stream.End(&ai.Message{Role: "assistant", Content: ai.BlockContent(), StopReason: "aborted", ErrorMessage: &message})
		return stream, nil
	}})
	done := make(chan error, 1)
	go func() { done <- agent.Prompt(context.Background(), "go") }()
	<-providerEntered
	var group sync.WaitGroup
	for i := 0; i < 20; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			agent.Steer(&ai.Message{Role: "user", Content: ai.TextContent("steer")})
			agent.FollowUp(&ai.Message{Role: "user", Content: ai.TextContent("follow")})
			_ = agent.State()
			_ = agent.PeekQueuedMessages()
			_ = agent.HasQueuedMessages()
			_ = agent.Signal()
		}()
	}
	group.Wait()
	agent.Abort()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if observed.Err() == nil || agent.State().ErrorMessage == nil || *agent.State().ErrorMessage != "aborted" {
		t.Fatal("abort state not recorded")
	}
	agent.SetSteeringMode("all")
	if len(agent.PeekQueuedMessages()) != 20 {
		t.Fatal("abort consumed queued steering")
	}
	agent.ClearSteeringQueue()
	agent.SetFollowUpMode("all")
	if len(agent.PeekQueuedMessages()) != 20 {
		t.Fatal("abort consumed queued follow-up")
	}
	if err := agent.Reset(); err != nil {
		t.Fatal(err)
	}
	if agent.HasQueuedMessages() || agent.State().ErrorMessage != nil {
		t.Fatal("reset did not clear queues/error")
	}
}

func TestAgentCopiesAssignedArraysAndSharesStateLists(t *testing.T) {
	messages := []*ai.Message{{Role: "user", Content: ai.TextContent("original")}}
	tools := []*engine.Tool{{Tool: ai.Tool{Name: "original", Parameters: json.RawMessage(`{}`)}}}
	agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Messages: messages, Tools: tools}, AgentConfig: engine.AgentConfig{StreamFn: completedAgentStream}})
	if err != nil {
		t.Fatal(err)
	}
	messages[0] = &ai.Message{Role: "custom"}
	tools[0] = &engine.Tool{}
	if agent.State().Messages.Get(1).Role != "user" || agent.State().Tools.Get(0).Name != "original" {
		t.Fatal("constructor retained caller array")
	}
	newMessages := []*ai.Message{{Role: "user", Content: ai.TextContent("replacement")}}
	newTools := []*engine.Tool{{Tool: ai.Tool{Name: "replacement"}}}
	agent.SetMessages(newMessages)
	agent.SetTools(newTools)
	newMessages[0] = &ai.Message{}
	newTools[0] = &engine.Tool{}
	state := agent.State()
	if state.Messages.Get(0).Role != "user" || state.Tools.Get(0).Name != "replacement" {
		t.Fatal("state setter retained caller array")
	}
	state.Messages.Set(0, &ai.Message{Role: "custom"})
	state.Tools.Set(0, &engine.Tool{Tool: ai.Tool{Name: "edited"}})
	if agent.State().Messages.Get(0).Role != "custom" || agent.State().Tools.Get(0).Name != "edited" {
		t.Fatal("state getter detached live collection")
	}
}

func TestAgentUsesLatestInstalledPreparationCallback(t *testing.T) {
	var agent *engine.Agent
	turns, oldCalls, newCalls := 0, 0, 0
	config := engine.AgentConfig{StreamFn: completedAgentStream, PrepareNextTurn: func(context.Context) (*engine.TurnUpdate, error) { oldCalls++; return nil, nil }}
	config.FinishTurn = func(context.Context, *engine.Turn) (string, error) {
		turns++
		if turns == 1 {
			updated := config
			updated.PrepareNextTurn = func(context.Context) (*engine.TurnUpdate, error) { newCalls++; return nil, nil }
			if err := agent.Configure(updated); err != nil {
				return "", err
			}
			return "continue", nil
		}
		return "end", nil
	}
	agent = newAgentForTest(t, config)
	if err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if oldCalls != 0 || newCalls != 1 || turns != 2 {
		t.Fatalf("callback replacement: old=%d new=%d turns=%d", oldCalls, newCalls, turns)
	}
}

func TestAgentParentCancellationAndDefaultBinding(t *testing.T) {
	engine.SetDefaultStreamFn(completedAgentStream)
	t.Cleanup(func() { engine.SetDefaultStreamFn(nil) })
	agent, err := engine.NewAgent(engine.AgentOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Prompt(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	engine.SetDefaultStreamFn(nil)
	if _, err := engine.NewAgent(engine.AgentOptions{}); err == nil {
		t.Fatal("missing stream accepted")
	}
	entered := make(chan struct{})
	agent = newAgentForTest(t, engine.AgentConfig{StreamFn: func(ctx context.Context, _ json.RawMessage, _ ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- agent.Prompt(ctx, "cancel") }()
	<-entered
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parent cancellation did not reach stream")
	}
	state := agent.State()
	if state.Messages.Get(state.Messages.Len()-1).StopReason != "aborted" || state.IsStreaming {
		t.Fatal("parent cancellation did not settle aborted lifecycle")
	}
}
