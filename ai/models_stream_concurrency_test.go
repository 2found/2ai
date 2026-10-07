package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

func TestPiLazyStreamLateResultPreservesLiveMessage(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	final := fixtureStreamMessage()
	step := 0
	outer := LazyStream(context.Background(), NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"}), func(ctx context.Context) (*ProviderEventSource, error) {
		shared := ctx.Value(assistantStreamSynchronizationKey{}).(*sync.Mutex)
		return &ProviderEventSource{Next: func(context.Context) (AssistantMessageEvent, bool, error) {
			n := step
			step++
			if n == 0 {
				return AssistantMessageEvent{Type: "start", Partial: final}, true, nil
			}
			if n == 1 {
				return AssistantMessageEvent{Type: "done", Reason: "stop", Message: final}, true, nil
			}
			return AssistantMessageEvent{}, false, nil
		}, Result: func(context.Context) (*Message, bool, error) {
			close(entered)
			<-release
			shared.Lock()
			final.Content = BlockContent(ContentBlock{Type: "text", Text: "late"})
			shared.Unlock()
			return nil, false, errors.New("late failure")
		}}, nil
	}, func() int64 { return 1000 })
	publicationAwait(t, entered)
	before, err := outer.Result(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	trace := map[string]any{"beforeLength": before.Content.Blocks.Len()}
	probe, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	if err := outer.WaitForEnd(probe); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("forwarding ended before source.result: %v", err)
	}
	once.Do(func() { close(release) })
	ctx, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	if err := outer.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	types := NewArray()
	for {
		event, ok, err := outer.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		types.Append(event.Type)
		if event.Type == "start" && event.Partial != final {
			t.Fatal("partial identity lost")
		}
	}
	after, err := outer.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	trace["sameResult"] = before == after && after == final
	trace["events"] = types
	trace["content"] = streamFixtureWire(after.Content)
	catalogCompare(t, trace, readModelsStreamFixture(t).LateResult)
}

func TestPiModelsStreamCancellationDoesNotCancelForwarding(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	final := fixtureStreamMessage()
	step := 0
	models := NewModels()
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
		return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "key"})}), nil
	}}}, Stream: func(context.Context, any, TranscriptContext, *Object) (*ProviderEventSource, error) {
		return &ProviderEventSource{Next: func(wait context.Context) (AssistantMessageEvent, bool, error) {
			if wait.Err() != nil {
				return AssistantMessageEvent{}, false, wait.Err()
			}
			n := step
			step++
			if n == 0 {
				return AssistantMessageEvent{Type: "start", Partial: final}, true, nil
			}
			if n == 1 {
				<-release
				return AssistantMessageEvent{Type: "done", Reason: "stop", Message: final}, true, nil
			}
			return AssistantMessageEvent{}, false, nil
		}, Result: func(context.Context) (*Message, bool, error) { return final, true, nil }}, nil
	}})
	stream := models.Stream(ctx, NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"}), Context{})
	first, ok, err := stream.Next(context.Background())
	if err != nil || !ok {
		t.Fatalf("first event: %v", err)
	}
	cancel(errors.New("cancelled"))
	once.Do(func() { close(release) })
	wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	if err := stream.WaitForEnd(wait); err != nil {
		t.Fatal(err)
	}
	types := NewArray(first.Type)
	for {
		event, ok, err := stream.Next(wait)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		types.Append(event.Type)
	}
	result, err := stream.Result(wait)
	if err != nil {
		t.Fatal(err)
	}
	catalogCompare(t, map[string]any{"events": types, "result": streamFixtureWire(result)}, readModelsStreamFixture(t).CancelledIteration)
}

func TestModelsStreamNativeHTTPIntegration(t *testing.T) {
	for _, simple := range []bool{false, true} {
		name := "stream"
		if simple {
			name = "simple"
		}
		t.Run(name, func(t *testing.T) {
			requests := make(chan *Object, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				body, err := jsonjs.DecodeValue(raw)
				if err != nil {
					t.Error(err)
				}
				requests <- NewObject(Property{Name: "path", Value: r.URL.Path}, Property{Name: "auth", Value: r.Header.Get("Authorization")}, Property{Name: "shared", Value: r.Header.Get("X-Shared")}, Property{Name: "transform", Value: r.Header.Get("X-Transform")}, Property{Name: "body", Value: body})
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"id\":\"response\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"response\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer server.Close()
			model := catalogDecode(t, []byte(`{"id":"test","api":"openai-completions","provider":"openai","baseUrl":"https://unused.invalid/v1","reasoning":false,"input":["text"],"maxTokens":8192,"contextWindow":32000,"cost":{"input":1,"output":3,"cacheRead":0.1,"cacheWrite":2},"headers":{"x-shared":"model"}}`))
			relay := NewAssistantMessageEventStream()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			ctx = WithAssistantStreamSynchronization(ctx, relay)
			models := NewModels()
			sharedLock := make(chan bool, 1)
			callback := func(c context.Context, m any, tr TranscriptContext, o *Object) (*ProviderEventSource, error) {
				rawModel, err := jsonjs.MarshalValue(m)
				if err != nil {
					return nil, err
				}
				rawOptions, err := jsonjs.MarshalValue(o)
				if err != nil {
					return nil, err
				}
				options := OpenAICompletionsStreamOptions{Options: rawOptions, Client: server.Client(), Now: func() int64 { return 1000 }}
				var inner *AssistantMessageEventStream
				if simple {
					inner, err = StreamOpenAICompletionsSimple(c, rawModel, tr, options)
				} else {
					inner = StreamOpenAICompletions(c, rawModel, tr, options)
				}
				if err != nil {
					return nil, err
				}
				sharedLock <- inner.payloadMu == relay.payloadMu
				return SourceFromAssistantStream(inner), nil
			}
			provider, factoryErr := NewModelProvider(&ProviderFactoryOptions{ID: "openai", Models: NewArray(model), BaseURL: "https://metadata.invalid", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
				return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "stored-key"}, Property{Name: "baseUrl", Value: server.URL + "/v1"}, Property{Name: "headers", Value: NewObject(Property{Name: "X-Shared", Value: "auth"})})}), nil
			}}}, API: &ProviderStreams{Stream: callback, StreamSimple: callback}})
			if factoryErr != nil {
				t.Fatal(factoryErr)
			}
			models.SetProvider(provider)
			catalog := models.GetAllModels("openai")
			if catalog.Len() != 1 || catalog.Get(0) != model {
				t.Fatal("factory catalog identity lost")
			}
			input := Context{SystemPrompt: "system", Messages: []Message{{Role: "user", Content: TextContent("hi"), Timestamp: 7}}}
			options := &ModelsRequestOptions{Values: NewObject(Property{Name: "maxTokens", Value: 32}, Property{Name: "headers", Value: NewObject(Property{Name: "X-SHARED", Value: "request"})}), TransformHeaders: func(h any) (any, error) { h.(*Object).Set("X-Transform", "yes"); return h, nil }}
			var stream *AssistantMessageEventStream
			if simple {
				stream = models.StreamSimple(ctx, model, input, options)
			} else {
				stream = models.Stream(ctx, model, input, options)
			}
			if stream.payloadMu != relay.payloadMu {
				t.Fatal("outer stream lost host payload lock")
			}
			kinds := map[string]bool{}
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				snapshot, err := stream.SnapshotEvent(event)
				if err != nil {
					t.Fatal(err)
				}
				kinds[snapshot.Type] = true
			}
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if result.StopReason != "stop" || result.Content.Blocks.Len() != 1 || result.Content.Blocks.Get(0).Text != "hello!" || !kinds["text_delta"] || !kinds["done"] {
				t.Fatalf("native forwarding failed: %s", mustStreamTestJSON(result))
			}
			if !publicationAwait(t, sharedLock) {
				t.Fatal("inner provider lost shared payload lock")
			}
			request := publicationAwait(t, requests)
			if request.Get("path") != "/v1/chat/completions" || request.Get("auth") != "Bearer stored-key" || request.Get("shared") != "request" || request.Get("transform") != "yes" {
				t.Fatalf("request auth/header precedence: %s", catalogStringify(t, request))
			}
			body := request.Get("body").(*Object)
			messages := body.Get("messages").(*Array)
			if messages.Len() != 2 || catalogProperty(messages.Get(0), "role") != "system" || catalogProperty(messages.Get(1), "content") != "hi" {
				t.Fatalf("transcript not normalized: %s", catalogStringify(t, body))
			}
		})
	}
}
func mustStreamTestJSON(value any) string {
	raw, err := jsonjs.MarshalValue(streamFixtureWire(value))
	if err != nil {
		panic(err)
	}
	return string(raw)
}
