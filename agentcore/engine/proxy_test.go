package engine_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

type proxyTransport func(*http.Request) (*http.Response, error)

func (f proxyTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type proxyPanicReader struct {
	failure any
	closed  *bool
}

func (r proxyPanicReader) Read([]byte) (int, error) { panic(r.failure) }
func (r proxyPanicReader) Close() error             { *r.closed = true; return nil }

type proxyFailureSpec struct {
	Kind, Message string
	Value         any
}

type proxyRequestFailure struct {
	before  func()
	failure any
	mode    string
}

func (f proxyRequestFailure) MarshalJSON() ([]byte, error) {
	if f.before != nil {
		f.before()
	}
	if f.mode == "nested-return" {
		return json.Marshal(map[string]any{"nested": proxyRequestFailure{failure: f.failure, mode: "return"}})
	}
	if f.mode == "return" {
		return nil, f.failure.(error)
	}
	panic(f.failure)
}

type proxyJSONMutation func()

func (advance proxyJSONMutation) MarshalJSON() ([]byte, error) {
	advance()
	return []byte(`{"serialized":true}`), nil
}

type proxyJSONRecord struct {
	key    string
	record func(string)
}

func (v proxyJSONRecord) MarshalJSON() ([]byte, error) { v.record(v.key); return json.Marshal(v.key) }

type proxyRawJSON string

func (raw proxyRawJSON) MarshalJSON() ([]byte, error) { return []byte(raw), nil }

func TestProxyPiFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-proxy.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		Model json.RawMessage
		Now   int64
		Cases []struct {
			Input struct {
				Name, Body, StatusText         string
				Status                         int
				Options                        map[string]any
				FailureAt                      string
				AbortOnFailure                 bool
				Failure                        *proxyFailureSpec
				RequestFailure                 *proxyFailureSpec
				RequestFailureMode             string
				RequestClockAdvance            bool
				RawMetadata                    *string
				RawMetadataCallback            bool
				RequestCycle                   string
				RequestMapMutation             string
				RequestSharedValue             bool
				RawRequest, RequestOptionOrder bool
				RequestNumber                  *struct {
					Kind, Value string
					Nested      bool
				}
				RequestModelMutation, ClockModelMutation bool
				AbortOnRequest                           bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixtures.Cases {
		t.Run(test.Input.Name, func(t *testing.T) {
			defer func() {
				if failure := recover(); failure != nil {
					t.Fatalf("proxy request preparation panicked (%T)", failure)
				}
			}()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			runCtx, abort := context.WithCancel(ctx)
			defer abort()
			var request any
			bodyClosed := false
			client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
				var body any
				rawBody, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if err := json.Unmarshal(rawBody, &body); err != nil {
					return nil, err
				}
				request = map[string]any{"url": r.URL.String(), "method": r.Method, "headers": map[string]string{"Authorization": r.Header.Get("Authorization"), "Content-Type": r.Header.Get("Content-Type")}, "body": body}
				if test.Input.RawRequest {
					request.(map[string]any)["rawBody"] = string(rawBody)
				}
				if test.Input.Failure != nil {
					if test.Input.AbortOnFailure {
						abort()
					}
					failure := test.Input.Failure.Value
					if test.Input.Failure.Kind == "error" {
						failure = errors.New(test.Input.Failure.Message)
					}
					if test.Input.FailureAt == "fetch" {
						panic(failure)
					}
					return &http.Response{StatusCode: 200, Status: "200 OK", Body: proxyPanicReader{failure, &bodyClosed}, Header: make(http.Header), Request: r}, nil
				}
				status := test.Input.Status
				if status == 0 {
					status = 200
				}
				return &http.Response{StatusCode: status, Status: statusLine(status, test.Input.StatusText), Body: io.NopCloser(strings.NewReader(test.Input.Body)), Header: make(http.Header), Request: r}, nil
			})}
			if spec := test.Input.RequestFailure; spec != nil {
				failure := spec.Value
				if spec.Kind == "error" {
					failure = errors.New(spec.Message)
				}
				if test.Input.Options == nil {
					test.Input.Options = map[string]any{}
				}
				var before func()
				if test.Input.AbortOnRequest {
					before = abort
				}
				test.Input.Options["metadata"] = proxyRequestFailure{failure: failure, mode: test.Input.RequestFailureMode, before: before}
			}
			requestedModel := append(json.RawMessage(nil), fixtures.Model...)
			mutateModel := func() {
				copy(requestedModel, bytes.Replace(requestedModel, []byte("proxy-model"), []byte("other-model"), 1))
			}
			clock := fixtures.Now
			if test.Input.RequestClockAdvance {
				if test.Input.Options == nil {
					test.Input.Options = map[string]any{}
				}
				test.Input.Options["metadata"] = proxyJSONMutation(func() { clock += 1000 })
			}
			if test.Input.RequestModelMutation {
				if test.Input.Options == nil {
					test.Input.Options = map[string]any{}
				}
				test.Input.Options["metadata"] = proxyJSONMutation(mutateModel)
			}
			if spec := test.Input.RequestNumber; spec != nil {
				var value any
				switch spec.Kind {
				case "int64":
					n, err := strconv.ParseInt(spec.Value, 10, 64)
					if err != nil {
						t.Fatal(err)
					}
					value = n
				case "uint64":
					n, err := strconv.ParseUint(spec.Value, 10, 64)
					if err != nil {
						t.Fatal(err)
					}
					value = n
				default:
					n, err := strconv.ParseFloat(spec.Value, 64)
					if err != nil {
						t.Fatal(err)
					}
					value = n
					if spec.Kind == "float32" {
						value = float32(n)
					}
				}
				if test.Input.Options == nil {
					test.Input.Options = map[string]any{}
				}
				if spec.Nested {
					test.Input.Options["metadata"] = map[string]any{"nested": []any{value}}
				} else {
					test.Input.Options["temperature"] = value
				}
			}
			serializerOrder := []string{}
			if test.Input.RequestOptionOrder {
				test.Input.Options = map[string]any{}
				for _, key := range []string{"temperature", "samplingParams", "maxTokens", "reasoning", "cacheRetention", "sessionId", "headers", "metadata", "transport", "thinkingBudgets", "maxRetryDelayMs", "ignored"} {
					test.Input.Options[key] = proxyJSONRecord{key, func(name string) { serializerOrder = append(serializerOrder, name) }}
				}
			}
			if test.Input.RequestMapMutation != "" {
				metadata := map[string]any{"b": 1}
				metadata["a"] = proxyJSONMutation(func() {
					switch test.Input.RequestMapMutation {
					case "replace":
						metadata["b"] = 2
					case "delete":
						delete(metadata, "b")
					case "insert":
						metadata["c"] = 3
					}
				})
				test.Input.Options = map[string]any{"metadata": metadata}
			}
			if test.Input.RequestSharedValue {
				child := map[string]any{"value": 1}
				test.Input.Options = map[string]any{"metadata": map[string]any{"a": child, "b": child}}
			}
			if test.Input.RequestCycle != "" {
				var value any
				if test.Input.RequestCycle == "array" {
					array := make([]any, 1)
					array[0] = array
					value = array
				} else {
					object := map[string]any{}
					object["self"] = object
					if test.Input.RequestCycle == "mutating-map" {
						object["unused"] = true
						object["a"] = proxyJSONMutation(func() { serializerOrder = append(serializerOrder, "a"); delete(object, "unused") })
					}
					value = object
				}
				test.Input.Options = map[string]any{"metadata": value}
			}
			if test.Input.RawMetadata != nil {
				var value any = json.RawMessage(*test.Input.RawMetadata)
				if test.Input.RawMetadataCallback {
					value = proxyRawJSON(*test.Input.RawMetadata)
				}
				test.Input.Options = map[string]any{"metadata": value}
			}
			stream := engine.StreamProxy(runCtx, requestedModel, ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example.com", AuthToken: "test-token", Client: client, Now: func() int64 {
				if test.Input.ClockModelMutation {
					mutateModel()
				}
				return clock
			}, Options: test.Input.Options})
			events := []ai.AssistantMessageEvent{}
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				events = append(events, event)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			if test.Input.FailureAt == "read" && !bodyClosed {
				t.Fatal("reader panic leaked response body")
			}
			values := map[string]any{"events": events, "result": result}
			if test.Input.RequestOptionOrder || test.Input.RequestCycle == "mutating-map" {
				values["serializerOrder"] = serializerOrder
			}
			if request != nil {
				values["request"] = request
			}
			actual, err := json.Marshal(values)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(test.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("proxy differs from Pi\ngot: %s\nwant: %s", actual, test.Expected)
			}
			// Engine consumers use isolated snapshots. The representation must
			// retain every field of the settled live payload, including nulls.
			snapshots := make([]ai.AssistantMessageEvent, len(events))
			for i, event := range events {
				snapshots[i], err = stream.SnapshotEvent(event)
				if err != nil {
					t.Fatal(err)
				}
			}
			final, err := stream.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
			}
			values["events"], values["result"] = snapshots, final
			copied, err := json.Marshal(values)
			if err != nil || !bytes.Equal(actual, copied) {
				t.Fatalf("snapshot changed settled payload: %v\nlive: %s\nsnapshot: %s", err, actual, copied)
			}
		})
	}
}

func statusLine(code int, text string) string {
	if text == "" {
		text = http.StatusText(code)
	}
	return fmt.Sprintf("%d %s", code, text)
}

func TestProxyCancellationAndLiveSnapshots(t *testing.T) {
	entered := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"type\":\"start\"}\n\ndata: {\"type\":\"text_start\",\"contentIndex\":0}\n\n")
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m","provider":"p","api":"a"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: server.URL})
	<-entered
	first, ok, err := stream.Next(wait)
	if err != nil || !ok {
		t.Fatalf("start: %v %v", ok, err)
	}
	if first.Partial == nil {
		t.Fatal("missing live partial")
	}
	_, ok, err = stream.Next(wait)
	if err != nil || !ok {
		t.Fatalf("text_start: %v %v", ok, err)
	}
	snapshot, err := stream.SnapshotEvent(first)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if err := stream.WaitForEnd(wait); err != nil {
		t.Fatal(err)
	}
	result, err := stream.Result(wait)
	if err != nil {
		t.Fatal(err)
	}
	if first.Partial != result || result.StopReason != "aborted" || result.ErrorMessage == nil || *result.ErrorMessage != "Request aborted by user" {
		t.Fatalf("abort lost identity/reason: %+v", result)
	}
	if snapshot.Partial.StopReason != "pending" {
		t.Fatal("snapshot changed with live payload")
	}
}

func TestProxyConcurrentEngineSnapshots(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"type\":\"start\"}\ndata: {\"type\":\"text_start\",\"contentIndex\":0}\n")
		for i := 0; i < 100; i++ {
			io.WriteString(w, "data: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"x\"}\n")
			w.(http.Flusher).Flush()
		}
		io.WriteString(w, "data: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{}}\n")
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var mu sync.Mutex
	var snapshots []*ai.Message
	messages, err := engine.Run(ctx, engine.NewList([]*ai.Message{{Role: "user", Content: ai.TextContent("test")}}...), engine.Context{}, engine.Config{ConvertToLLM: func(messages *engine.MessageList) (*engine.MessageList, error) { return messages, nil }, Model: json.RawMessage(`{"id":"m","provider":"p","api":"a"}`)}, func(e engine.Event) error {
		if e.Message != nil {
			mu.Lock()
			snapshots = append(snapshots, e.Message)
			mu.Unlock()
		}
		return nil
	}, func(ctx context.Context, m json.RawMessage, c ai.TranscriptContext, _ map[string]any) (*ai.AssistantMessageEventStream, error) {
		return engine.StreamProxy(ctx, m, c, engine.ProxyStreamOptions{ProxyURL: server.URL}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	last := messages.Get(messages.Len() - 1)
	if last.Content.Blocks.Get(0).Text != strings.Repeat("x", 100) {
		t.Fatalf("lost deltas: %+v", last)
	}
	// Consumer-owned snapshots are safe to retain and serialize independently.
	if _, err := json.Marshal(snapshots); err != nil {
		t.Fatal(err)
	}
}

// One-byte reads split every UTF-8 sequence and every protocol delimiter.
type proxyByteReader struct{ data []byte }

func (r *proxyByteReader) Read(dst []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	if len(dst) == 0 {
		return 0, nil
	}
	dst[0] = r.data[0]
	r.data = r.data[1:]
	return 1, nil
}
func (*proxyByteReader) Close() error { return nil }

func TestProxyFragmentedUTF8AndLongFrames(t *testing.T) {
	text := strings.Repeat("Chào 🌏", 10000)
	delta, _ := json.Marshal(map[string]any{"type": "text_delta", "contentIndex": 0, "delta": text})
	body := "\uFEFFdata: {\"type\":\"start\"}\r\ndata: {\"type\":\"text_start\",\"contentIndex\":0}\r\ndata: " + string(delta) + "\r\ndata: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{}}"
	client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Status: "200 OK", Body: &proxyByteReader{data: []byte(body)}, Header: make(http.Header), Request: r}, nil
	})}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example.com", Client: client})
	result, err := stream.SnapshotResult(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "stop" || result.Content.Blocks.Len() != 1 || result.Content.Blocks.Get(0).Text != text {
		t.Fatal("fragmented/long frame corrupted")
	}
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestProxyMalformedJSONSettles(t *testing.T) {
	client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Status: "200 OK", Body: io.NopCloser(strings.NewReader("data: {invalid}\n")), Header: make(http.Header), Request: r}, nil
	})}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example.com", Client: client})
	result, err := stream.SnapshotResult(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "error" || result.ErrorMessage == nil {
		t.Fatalf("malformed JSON did not settle: %+v", result)
	}
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestProxyDrainsAfterTerminal(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	finish := func() { once.Do(func() { close(release) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"type\":\"start\"}\ndata: {\"type\":\"text_start\",\"contentIndex\":0}\ndata: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{}}\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		io.WriteString(w, "data: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"after terminal\"}\n")
	}))
	defer server.Close()
	defer finish()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: server.URL})
	raw, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	before, err := stream.SnapshotResult(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if before.Content.Blocks.Get(0).Text != "" {
		t.Fatal("server passed the gate")
	}
	finish()
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	after, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if raw != after || after.Content.Blocks.Get(0).Text != "after terminal" || before.Content.Blocks.Get(0).Text != "" {
		t.Fatal("live result stopped updating at terminal")
	}
	events := []string{}
	for {
		event, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		events = append(events, event.Type)
	}
	if !reflect.DeepEqual(events, []string{"start", "text_start", "done"}) {
		t.Fatalf("post-terminal event leaked: %v", events)
	}
}

// An injected transport can return a body whose reads do not observe the
// request context. Pi's abort handler still cancels that body explicitly.
func TestProxyAbortClosesIndependentBody(t *testing.T) {
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	admitted := make(chan struct{})
	client := &http.Client{Transport: proxyTransport(func(*http.Request) (*http.Response, error) {
		close(admitted)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: reader}, nil
	})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"api":"test","provider":"test","id":"test"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "http://proxy.test", Client: client})
	<-admitted
	cancel()
	wait, done := context.WithTimeout(context.Background(), time.Second)
	defer done()
	result, err := stream.SnapshotResult(wait)
	if err != nil {
		t.Fatalf("abort did not release the independent response reader: %v", err)
	}
	if result.StopReason != "aborted" || result.ErrorMessage == nil || *result.ErrorMessage != "Request aborted by user" {
		t.Fatalf("unexpected abort result: %+v", result)
	}
	if err := stream.WaitForEnd(wait); err != nil {
		t.Fatal(err)
	}
}

func TestProxySharesRelayPayloadLock(t *testing.T) {
	relay := ai.NewAssistantMessageEventStream()
	ctx := ai.WithAssistantStreamSynchronization(context.Background(), relay)
	client := &http.Client{Transport: proxyTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("data: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{}}\n"))}, nil
	})}
	stream := engine.StreamProxy(ctx, json.RawMessage(`{"api":"test","provider":"test","id":"test"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "http://proxy.test", Client: client})
	started, acquired := make(chan struct{}), make(chan struct{})
	relay.Synchronize(func() {
		go func() { close(started); stream.Synchronize(func() { close(acquired) }) }()
		<-started
		select {
		case <-acquired:
			t.Error("proxy payload access bypassed the relay lock")
		case <-time.After(50 * time.Millisecond):
		}
	})
	wait, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	select {
	case <-acquired:
	case <-wait.Done():
		t.Fatal("payload access did not resume")
	}
	if err := stream.WaitForEnd(wait); err != nil {
		t.Fatal(err)
	}
}
