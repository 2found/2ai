package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/2found/2ai/jsonjs"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

type modelsStreamFixture struct {
	UpstreamCommit                 string
	LateResult, CancelledIteration json.RawMessage
	Lazy                           []struct {
		Mode                       string
		Model, Events, Result, Log json.RawMessage
	}
	ModelsCases []struct {
		Method, Mode        string
		Events, Result, Log json.RawMessage
	}
	APICases []struct {
		Method, Mode        string
		HasFetch, HasCancel bool
		Outputs, Log        json.RawMessage
		Loads               int
	}
}

func readModelsStreamFixture(t *testing.T) modelsStreamFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-models-stream.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture modelsStreamFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Lazy) != 36 || len(fixture.ModelsCases) != 48 || len(fixture.APICases) != 24 {
		t.Fatal("unexpected stream coverage")
	}
	return fixture
}
func streamFixtureWire(value any) any {
	raw, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	decoded, err := jsonjs.DecodeValue(raw)
	if err != nil {
		panic(err)
	}
	return decoded
}

func fixtureStreamMessage() *Message {
	return &Message{Role: "assistant", Content: BlockContent(), API: "test", Provider: "p", Model: "done", Usage: &Usage{}, StopReason: "stop", Timestamp: 1000}
}
func fixtureEventSource(mode string, log *Array) *ProviderEventSource {
	step := 0
	final := fixtureStreamMessage()
	terminal := mode == "done" || mode == "late-result-error" || mode == "late-iterator-error"
	source := &ProviderEventSource{Next: func(context.Context) (AssistantMessageEvent, bool, error) {
		n := step
		step++
		log.Append(NewObject(Property{Name: "next", Value: n}))
		if mode == "iterator-error" || (mode == "after-start-error" && n == 1) || (mode == "late-iterator-error" && n == 2) {
			return AssistantMessageEvent{}, false, errors.New("iterator failed")
		}
		if n == 0 {
			return AssistantMessageEvent{Type: "start", Partial: final}, true, nil
		}
		if n == 1 && terminal {
			return AssistantMessageEvent{Type: "done", Reason: "stop", Message: final}, true, nil
		}
		return AssistantMessageEvent{}, false, nil
	}}
	if mode != "no-result" {
		source.Result = func(context.Context) (*Message, bool, error) {
			log.Append("result")
			if mode == "result-error" || mode == "late-result-error" {
				return nil, false, errors.New("result failed")
			}
			if mode == "undefined-result" {
				return nil, false, nil
			}
			if mode == "null-result" {
				return nil, true, nil
			}
			return final, true, nil
		}
	}
	return source
}
func observeFixtureStream(t *testing.T, stream *AssistantMessageEventStream) (*Array, any) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	events := NewArray()
	for {
		event, present, err := stream.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !present {
			break
		}
		events.Append(streamFixtureWire(event))
	}
	probe, cancelProbe := context.WithCancel(context.Background())
	cancelProbe()
	result, err := stream.Result(probe)
	if err != nil {
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		return events, map[string]any{"pending": true}
	}
	return events, authResolveCapture(streamFixtureWire(result), nil)
}
func TestPiLazyStream(t *testing.T) {
	for index, tc := range readModelsStreamFixture(t).Lazy {
		t.Run(fmt.Sprintf("%d/%s", index, tc.Mode), func(t *testing.T) {
			log := NewArray()
			source := fixtureEventSource(tc.Mode, log)
			stream := LazyStream(context.Background(), catalogDecode(t, tc.Model), func(context.Context) (*ProviderEventSource, error) {
				log.Append("setup")
				if tc.Mode == "setup-error" {
					return nil, errors.New("setup failed")
				}
				if tc.Mode == "missing-source" {
					return nil, nil
				}
				return source, nil
			}, func() int64 { return 1000 })
			events, result := observeFixtureStream(t, stream)
			catalogCompare(t, events, tc.Events)
			catalogCompare(t, result, tc.Result)
			catalogCompare(t, log, tc.Log)
		})
	}
}
func TestPiModelsStreamDispatch(t *testing.T) {
	for _, tc := range readModelsStreamFixture(t).ModelsCases {
		t.Run(tc.Method+"/"+tc.Mode, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			if tc.Mode == "pre-abort" {
				cancel(errors.New("cancelled"))
			}
			authGate := make(chan struct{})
			var releaseAuth sync.Once
			defer releaseAuth.Do(func() { close(authGate) })

			log := NewArray()
			kind := "chat"
			if tc.Mode == "wrong-type" {
				kind = "image"
			}
			model := NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"}, Property{Name: "type", Value: kind})
			input := Context{SystemPrompt: "system", Messages: []Message{{Role: "user", Content: TextContent("hi"), Timestamp: 7}}}
			handle := NewObject(Property{Name: "id", Value: "handle"})
			values := NewObject(Property{Name: "apiKey", Value: "explicit"}, Property{Name: "headers", Value: NewObject(Property{Name: "X-Request", Value: "yes"})}, Property{Name: "signal", Value: NewObject()})
			settings := &ModelsRequestOptions{Values: values, Now: func() int64 { return 1000 }, TransformHeaders: func(h any) (any, error) {
				log.Append(NewObject(Property{Name: "transform", Value: h}))
				if tc.Mode == "transform-error" {
					return nil, errors.New("transform failed")
				}
				result := authSpread(h)
				result.Set("X-Transformed", "yes")
				return result, nil
			}}
			mode := tc.Mode
			if mode == "snapshot" {
				mode = "done"
			}
			source := fixtureEventSource(mode, log)
			dispatch := func(callCtx context.Context, m, arg any, o *Object) (*ProviderEventSource, error) {
				log.Append(NewObject(Property{Name: "dispatch", Value: tc.Method}, Property{Name: "model", Value: m}, Property{Name: "input", Value: streamFixtureWire(arg)}, Property{Name: "options", Value: o}, Property{Name: "aborted", Value: callCtx.Err() != nil}))
				if tc.Mode == "provider-error" {
					return nil, errors.New("provider failed")
				}
				return source, nil
			}
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(args APIKeyAuthInput) (any, error) {
				log.Append(NewObject(Property{Name: "auth", Value: authResolveCapture(args.Credential, nil)}))
				if tc.Mode == "snapshot" {
					<-authGate
				}
				if tc.Mode == "setup-error" {
					return nil, errors.New("auth failed")
				}
				if tc.Mode == "no-auth" {
					return Undefined, nil
				}
				return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "auth-key"}, Property{Name: "headers", Value: NewObject(Property{Name: "X-Auth", Value: "yes"})})}), nil
			}}}, Stream: func(c context.Context, m any, tr TranscriptContext, o *Object) (*ProviderEventSource, error) {
				return dispatch(c, m, tr, o)
			}, StreamSimple: func(c context.Context, m any, tr TranscriptContext, o *Object) (*ProviderEventSource, error) {
				return dispatch(c, m, tr, o)
			}, FetchDeferred: dispatch}
			if tc.Mode == "missing-method" {
				provider.Stream = nil
				provider.StreamSimple = nil
				provider.FetchDeferred = nil
			}
			models := NewModels()
			if tc.Mode != "unknown-provider" {
				models.SetProvider(provider)
			}
			if strings.HasPrefix(tc.Method, "complete") || tc.Method == "fetchDeferred" {
				var value *Message
				var err error
				if tc.Method == "complete" {
					value, err = models.Complete(ctx, model, input, settings)
				} else if tc.Method == "completeSimple" {
					value, err = models.CompleteSimple(ctx, model, input, settings)
				} else {
					value, err = models.FetchDeferred(ctx, model, handle, settings)
				}
				catalogCompare(t, authResolveCapture(streamFixtureWire(value), err), tc.Result)
			} else {
				var stream *AssistantMessageEventStream
				if tc.Method == "stream" {
					stream = models.Stream(ctx, model, input, settings)
				} else if tc.Method == "streamSimple" {
					stream = models.StreamSimple(ctx, model, input, settings)
				} else {
					stream = models.StreamDeferred(ctx, model, handle, settings)
				}
				if tc.Mode == "snapshot" {
					models.DeleteProvider("p")
					values.Set("apiKey", "late")
					releaseAuth.Do(func() { close(authGate) })
				}

				events, result := observeFixtureStream(t, stream)
				catalogCompare(t, events, tc.Events)
				catalogCompare(t, result, tc.Result)
			}
			catalogCompare(t, log, tc.Log)
		})
	}
}
func observeFixtureSource(t *testing.T, source *ProviderEventSource) any {
	t.Helper()
	events := NewArray()
	for {
		event, present, err := source.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !present {
			break
		}
		events.Append(streamFixtureWire(event))
	}
	value, present, err := source.Result(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var result any = map[string]any{"pending": true}
	if present {
		result = authResolveCapture(streamFixtureWire(value), nil)
	}
	return map[string]any{"events": events, "result": result}
}
func TestPiLazyAPI(t *testing.T) {
	for _, tc := range readModelsStreamFixture(t).APICases {
		t.Run(tc.Method+"/"+tc.Mode, func(t *testing.T) {
			log := NewArray()
			loads := 0
			model := NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"})
			transcript := NormalizeContext(Context{})
			handle := NewObject(Property{Name: "id", Value: "handle"})
			options := NewObject(Property{Name: "custom", Value: "same"})
			source := fixtureEventSource("result-only", log)
			callback := func(_ context.Context, m, arg any, o *Object) (*ProviderEventSource, error) {
				sameInput := false
				if strings.Contains(tc.Method, "Deferred") {
					sameInput = arg == any(handle)
				} else {
					_, sameInput = arg.(TranscriptContext)
				}
				log.Append(NewObject(Property{Name: "method", Value: tc.Method}, Property{Name: "sameModel", Value: m == any(model)}, Property{Name: "sameInput", Value: sameInput}, Property{Name: "sameOptions", Value: o == options}))
				return source, nil
			}
			implementation := &ProviderStreams{Stream: func(c context.Context, m any, tr TranscriptContext, o *Object) (*ProviderEventSource, error) {
				return callback(c, m, tr, o)
			}, StreamSimple: func(c context.Context, m any, tr TranscriptContext, o *Object) (*ProviderEventSource, error) {
				return callback(c, m, tr, o)
			}}
			if tc.Mode != "unsupported" {
				implementation.FetchDeferred = callback
				implementation.CancelDeferred = func(c context.Context, m, h any, o *Object) error { _, err := callback(c, m, h, o); return err }
			}
			api := LazyAPI(func(context.Context) (*ProviderStreams, error) {
				loads++
				log.Append("load")
				if tc.Mode == "load-error" || (tc.Mode == "retry-load" && loads == 1) {
					return nil, errors.New("load failed")
				}
				return implementation, nil
			}, LazyAPICapabilities{FetchDeferred: tc.Mode != "disabled", CancelDeferred: tc.Mode != "disabled"}, func() int64 { return 1000 })
			if (api.FetchDeferred != nil) != tc.HasFetch || (api.CancelDeferred != nil) != tc.HasCancel {
				t.Fatal("capability mismatch")
			}
			attempts := 1
			if tc.Mode == "retry-load" || tc.Mode == "repeat" {
				attempts = 2
			}
			outputs := NewArray()
			for attempt := 0; attempt < attempts; attempt++ {
				if (tc.Method == "fetchDeferred" && api.FetchDeferred == nil) || (tc.Method == "cancelDeferred" && api.CancelDeferred == nil) {
					outputs.Append(NewObject(Property{Name: "absent", Value: true}))
					continue
				}
				if tc.Method == "cancelDeferred" {
					err := api.CancelDeferred(context.Background(), model, handle, options)
					if err != nil {
						outputs.Append(NewObject(Property{Name: "error", Value: err.Error()}))
					} else {
						outputs.Append(NewObject(Property{Name: "absent", Value: true}))
					}
					continue
				}
				var stream *ProviderEventSource
				var err error
				if tc.Method == "stream" {
					stream, err = api.Stream(context.Background(), model, transcript, options)
				} else if tc.Method == "streamSimple" {
					stream, err = api.StreamSimple(context.Background(), model, transcript, options)
				} else {
					stream, err = api.FetchDeferred(context.Background(), model, handle, options)
				}
				if err != nil {
					t.Fatal(err)
				}
				outputs.Append(observeFixtureSource(t, stream))
			}
			catalogCompare(t, outputs, tc.Outputs)
			catalogCompare(t, log, tc.Log)
			if loads != tc.Loads {
				t.Fatalf("loads %d, want %d", loads, tc.Loads)
			}
		})
	}
}
