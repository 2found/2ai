package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPiAnthropicClientOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-anthropic-client.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, Success string
		Model                   map[string]json.RawMessage
		Cases                   []struct {
			Input struct {
				Name, Fail     string
				Model, Options map[string]json.RawMessage
				Context        Context
				Replacement    json.RawMessage
				Status         int
				Body           *string
				Failures       []struct {
					Message string
					Status  *int
					Headers map[string]string
				}
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 20 {
		t.Fatal("unexpected injected-client oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			model := map[string]json.RawMessage{}
			for k, v := range fixture.Model {
				model[k] = v
			}
			for k, v := range tc.Input.Model {
				model[k] = v
			}
			rawModel, _ := json.Marshal(model)
			rawOptions, _ := json.Marshal(tc.Input.Options)
			phases := []string{}
			calls := []any{}
			client := &AnthropicMessageClient{CreateResponse: func(ctx context.Context, params json.RawMessage, options AnthropicRequestOptions) (*http.Response, error) {
				phases = append(phases, "request")
				calls = append(calls, map[string]any{"params": append(json.RawMessage(nil), params...), "options": options})
				if len(calls) <= len(tc.Input.Failures) {
					failure := tc.Input.Failures[len(calls)-1]
					if failure.Status == nil {
						return nil, errors.New(failure.Message)
					}
					headers := http.Header{}
					for k, v := range failure.Headers {
						headers.Set(k, v)
					}
					return nil, &AnthropicClientError{Status: *failure.Status, Headers: headers, Message: failure.Message}
				}
				text := fixture.Success
				if tc.Input.Body != nil {
					text = *tc.Input.Body
				}
				status := tc.Input.Status
				if status == 0 {
					status = 200
				}
				var body io.ReadCloser = io.NopCloser(strings.NewReader(text))
				if status == 204 {
					body = nil
				}
				return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": []string{"text/event-stream"}, "X-Client": []string{"custom"}}, Body: body}, nil
			}}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			options := AnthropicStreamOptions{Options: rawOptions, Now: func() int64 { return 100 }, Client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				t.Error("builtin HTTP called")
				return nil, errors.New("builtin fetch called")
			})},
				OnPayload: func(context.Context, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
					phases = append(phases, "payload")
					if tc.Input.Fail == "payload" {
						return nil, errors.New("payload failed")
					}
					return tc.Input.Replacement, nil
				},
				OnResponse: func(context.Context, CompletionsResponse, json.RawMessage) error {
					phases = append(phases, "response")
					if tc.Input.Fail == "response" {
						return errors.New("response failed")
					}
					return nil
				},
				OnProviderStreamEvent: func(context.Context, *json.RawMessage, json.RawMessage) error {
					phases = append(phases, "event")
					if tc.Input.Fail == "event" {
						return errors.New("event failed")
					}
					return nil
				},
			}
			stream := StreamAnthropicWithClient(ctx, rawModel, NormalizeContext(tc.Input.Context), options, client)
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.SnapshotResult(ctx)
			if err != nil {
				t.Fatal(err)
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
			actual, err := json.Marshal(map[string]any{"calls": calls, "phases": phases, "events": events, "result": result})
			if err != nil {
				t.Fatal(err)
			}
			decode := func(raw []byte) any {
				var value any
				d := json.NewDecoder(bytes.NewReader(raw))
				d.UseNumber()
				if err := d.Decode(&value); err != nil {
					t.Fatal(err)
				}
				return value
			}
			if !reflect.DeepEqual(decode(actual), decode(tc.Expected)) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}

type anthropicClientTestBody struct {
	io.Reader
	closed bool
}

func (b *anthropicClientTestBody) Close() error { b.closed = true; return nil }

func TestAnthropicInjectedClientClosesBodyOnCallbackFailure(t *testing.T) {
	body := &anthropicClientTestBody{Reader: strings.NewReader("unread")}
	client := &AnthropicMessageClient{CreateResponse: func(ctx context.Context, _ json.RawMessage, _ AnthropicRequestOptions) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: body}, nil
	}}
	options := AnthropicStreamOptions{OnResponse: func(context.Context, CompletionsResponse, json.RawMessage) error { return errors.New("stop") }}
	stream := StreamAnthropicWithClient(context.Background(), json.RawMessage(`{"id":"test","api":"anthropic-messages","provider":"anthropic"}`), NormalizeContext(Context{}), options, client)
	if err := stream.WaitForEnd(context.Background()); err != nil {
		t.Fatal(err)
	}
	result, err := stream.SnapshotResult(context.Background())
	if err != nil || result.StopReason != "error" || !body.closed {
		t.Fatalf("response lifecycle: %+v closed=%v err=%v", result, body.closed, err)
	}
}

func TestAnthropicInjectedClientReceivesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	entered := make(chan struct{})
	client := &AnthropicMessageClient{CreateResponse: func(ctx context.Context, _ json.RawMessage, _ AnthropicRequestOptions) (*http.Response, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	stream := StreamAnthropicWithClient(ctx, json.RawMessage(`{"id":"test","api":"anthropic-messages","provider":"anthropic"}`), NormalizeContext(Context{}), AnthropicStreamOptions{}, client)
	select {
	case <-entered:
	case <-wait.Done():
		t.Fatal("injected client was not called")
	}
	cancel()
	if err := stream.WaitForEnd(wait); err != nil {
		t.Fatal(err)
	}
	result, err := stream.SnapshotResult(wait)
	if err != nil || result.StopReason != "aborted" {
		t.Fatalf("cancelled client: %+v err=%v", result, err)
	}
}
