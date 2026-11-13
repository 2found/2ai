package ai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"
	"time"
)

func TestPiEventStreamOracle(t *testing.T) {
	type event struct {
		Type  string `json:"type"`
		Value int    `json:"value"`
	}
	var fixtures struct {
		Streams []struct {
			Name       string
			Operations []struct {
				Type   string
				Event  event
				Result *int
			}
			Expected json.RawMessage
		}
	}
	data, err := os.ReadFile("testdata/pi-transcript.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures.Streams {
		t.Run(fixture.Name, func(t *testing.T) {
			stream := NewEventStream(func(e event) bool { return e.Type == "done" || e.Type == "error" }, func(e event) int { return e.Value })
			for _, operation := range fixture.Operations {
				if operation.Type == "push" {
					stream.Push(operation.Event)
				} else if operation.Result == nil {
					stream.End()
				} else {
					stream.End(*operation.Result)
				}
			}
			events := []event{}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			for {
				e, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				events = append(events, e)
			}
			cancel()
			value, err := stream.Result(ctx)
			var result *int
			if err == nil {
				result = &value
			}
			assertPiJSON(t, fixture.Expected, map[string]any{"events": events, "resultResolved": err == nil, "result": result})
		})
	}
}

func TestPiEventStreamResultDoesNotDrain(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	message := &Message{Role: "assistant", Content: BlockContent(), StopReason: "pending"}
	stream.Push(AssistantMessageEvent{Type: "start", Partial: message})
	message.StopReason = "stop"
	stream.Push(AssistantMessageEvent{Type: "done", Reason: "stop", Message: message})
	stream.Push(AssistantMessageEvent{Type: "text_delta", Delta: "ignored"})
	stream.End(&Message{StopReason: "error"})
	result, err := stream.Result(context.Background())
	if err != nil || result != message {
		t.Fatalf("final result identity lost: %v %v", result, err)
	}
	first, ok, err := stream.Next(context.Background())
	if err != nil || !ok || first.Type != "start" || first.Partial != message || first.Partial.StopReason != "stop" {
		t.Fatalf("live partial lost: %+v %v %v", first, ok, err)
	}
	last, ok, err := stream.Next(context.Background())
	if err != nil || !ok || last.Type != "done" {
		t.Fatalf("terminal event missing: %+v %v %v", last, ok, err)
	}
	if _, ok, err := stream.Next(context.Background()); ok || err != nil {
		t.Fatalf("stream did not end: %v %v", ok, err)
	}
}

func TestPiEventStreamErrorBeforeStart(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	message := &Message{Role: "assistant", StopReason: "error"}
	stream.Push(AssistantMessageEvent{Type: "error", Reason: "error", Error: message})
	result, err := stream.Result(context.Background())
	if err != nil || result != message {
		t.Fatal("setup error did not settle result")
	}
	value, ok, err := stream.Next(context.Background())
	if err != nil || !ok || value.Error != message {
		t.Fatal("setup error event lost")
	}
}

func TestPiEventStreamEndWithoutResult(t *testing.T) {
	stream := NewEventStream(func(v int) bool { return v < 0 }, func(v int) int { return v })
	stream.Push(7)
	stream.End()
	stream.Push(8)
	value, ok, err := stream.Next(context.Background())
	if err != nil || !ok || value != 7 {
		t.Fatal("end discarded queued event")
	}
	if _, ok, err := stream.Next(context.Background()); ok || err != nil {
		t.Fatal("push after end was accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := stream.Result(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("end without result resolved the promise: %v", err)
	}
	stream.End(9)
	stream.End(10)
	if result, err := stream.Result(ctx); err != nil || result != 9 {
		t.Fatalf("first settled result did not win: %v %v", result, err)
	}
}

func TestPiEventStreamConcurrentFIFO(t *testing.T) {
	stream := NewEventStream(func(v int) bool { return v == 10000 }, func(v int) int { return v })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i <= 10000; i++ {
			stream.Push(i)
		}
		stream.End()
	}()
	for i := 0; i <= 10000; i++ {
		value, ok, err := stream.Next(ctx)
		if err != nil || !ok || value != i {
			t.Fatalf("event %d: %d %v %v", i, value, ok, err)
		}
	}
	wg.Wait()
	if _, ok, err := stream.Next(ctx); ok || err != nil {
		t.Fatal("missing stream termination")
	}
}

func TestPiEventStreamCanceledReaderKeepsNextEvent(t *testing.T) {
	stream := NewEventStream(func(v int) bool { return false }, func(v int) int { return v })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("reader cancellation: %v", err)
	}
	stream.Push(42)
	value, ok, err := stream.Next(context.Background())
	if err != nil || !ok || value != 42 {
		t.Fatal("cancelled reader consumed the next event")
	}
	stream.End()
}

type waitingStreamContext struct {
	context.Context
	waiting chan struct{}
	once    sync.Once
}

func (c *waitingStreamContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}

func TestPiEventStreamCancelPendingReader(t *testing.T) {
	stream := NewEventStream(func(v int) bool { return false }, func(v int) int { return v })
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &waitingStreamContext{Context: base, waiting: make(chan struct{})}
	finished := make(chan error, 1)
	go func() { _, _, err := stream.Next(ctx); finished <- err }()
	<-ctx.waiting // Next has installed the waiter before evaluating Done.
	cancel()
	if err := <-finished; !errors.Is(err, context.Canceled) {
		t.Fatalf("pending reader: %v", err)
	}
	stream.Push(42)
	value, ok, err := stream.Next(context.Background())
	if err != nil || !ok || value != 42 {
		t.Fatal("pending cancelled reader consumed a future event")
	}
	stream.End()
}

func TestPiEventStreamEndWakesAllReaders(t *testing.T) {
	stream := NewEventStream(func(v int) bool { return false }, func(v int) int { return v })
	type outcome struct {
		reader, value int
		ok            bool
		err           error
	}
	finished := make(chan outcome, 2)
	base, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	for i := 0; i < 2; i++ {
		ctx := &waitingStreamContext{Context: base, waiting: make(chan struct{})}
		go func(reader int) { value, ok, err := stream.Next(ctx); finished <- outcome{reader, value, ok, err} }(i)
		<-ctx.waiting
	}
	stream.Push(42)
	stream.End(9)
	for i := 0; i < 2; i++ {
		got := <-finished
		if got.err != nil || (got.reader == 0 && (!got.ok || got.value != 42)) || (got.reader == 1 && got.ok) {
			t.Fatalf("waiting readers were not served in order: %+v", got)
		}
	}
}

func TestAssistantStreamSnapshotAndProducerSettlement(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	message := &Message{Role: "assistant", Content: BlockContent(ContentBlock{Type: "text", Text: "before"}), Usage: &Usage{}, StopReason: "pending"}
	event := AssistantMessageEvent{Type: "start", Partial: message}
	stream.Synchronize(func() { stream.Push(event) })
	snapshot, err := stream.SnapshotEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	stream.Synchronize(func() {
		message.Content.Blocks.Get(0).Text = "after"
		message.StopReason = "stop"
		stream.Push(AssistantMessageEvent{Type: "done", Message: message})
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := stream.WaitForEnd(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal("terminal event was mistaken for producer settlement")
	}
	result, err := stream.SnapshotResult(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stream.Synchronize(func() { message.Content.Blocks.Get(0).Text = "post-terminal" })
	stream.End()
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	if snapshot.Partial.Content.Blocks.Get(0).Text != "before" || result.Content.Blocks.Get(0).Text != "after" || event.Partial.Content.Blocks.Get(0).Text != "post-terminal" {
		t.Fatal("snapshot/live payload semantics changed")
	}
}

func TestAssistantStreamSnapshotPreservesObjectGraph(t *testing.T) {
	stream := NewAssistantMessageEventStream()
	shared := &ContentBlock{Type: "toolCall", ID: "call", Name: "echo", Arguments: json.RawMessage(`{}`)}
	distinct := *shared
	usage := &Usage{Input: 3}
	partial := &Message{Role: "assistant", Content: BlockReferences(shared, shared, &distinct), Usage: usage}
	other := &Message{Role: "assistant", Content: BlockReferences(shared), Usage: usage}
	source := AssistantMessageEvent{Type: "toolcall_end", ToolCall: shared, Partial: partial, Message: other, Error: partial}
	snapshot, err := stream.SnapshotEvent(source)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Partial == partial || snapshot.Message == other || snapshot.ToolCall == shared || snapshot.Partial.Usage == usage {
		t.Fatal("snapshot retained a mutable producer object")
	}
	if snapshot.Partial != snapshot.Error || snapshot.Message == snapshot.Partial || snapshot.Message.Usage != snapshot.Partial.Usage {
		t.Fatal("message or usage aliases were lost or unrelated objects merged")
	}
	blocks := snapshot.Partial.Content.Blocks.Values()
	if blocks[0] != blocks[1] || blocks[0] != snapshot.ToolCall || blocks[0] != snapshot.Message.Content.Blocks.Get(0) || blocks[0] == blocks[2] {
		t.Fatal("block aliases were lost or equal but distinct blocks merged")
	}
	before, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	after, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatalf("snapshot changed wire data: %s", after)
	}
	snapshot.ToolCall.Name = "snapshot"
	snapshot.Partial.Usage.Input = 9
	if blocks[1].Name != "snapshot" || snapshot.Message.Content.Blocks.Get(0).Name != "snapshot" || snapshot.Message.Usage.Input != 9 || shared.Name != "echo" || usage.Input != 3 {
		t.Fatal("snapshot edits did not follow the detached object graph")
	}
	stream.Push(AssistantMessageEvent{Type: "done", Message: partial})
	stream.End()
	result, err := stream.SnapshotResult(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result == partial || result.Content.Blocks.Get(0) == shared || result.Content.Blocks.Get(0) != result.Content.Blocks.Get(1) || result.Content.Blocks.Get(0) == result.Content.Blocks.Get(2) {
		t.Fatal("result snapshot lost repeated block identity")
	}
	if result.Content.Blocks.Get(0).Name != "echo" || result.Usage.Input != 3 {
		t.Fatal("snapshots share mutable objects with each other")
	}
}
