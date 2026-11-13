package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPiAnthropicSimpleOracle(t *testing.T) {
	clearAnthropicFederationEnv(t)
	updates := readPiReconUpdates(t)
	raw, err := os.ReadFile("testdata/pi-anthropic-simple.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          map[string]json.RawMessage
		Cases          []struct {
			Input struct {
				Name           string
				Model, Options map[string]json.RawMessage
				Context        Context
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 122 {
		t.Fatal("unexpected Anthropic simple oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			expected := updates.simpleExpectation(t, "pi-anthropic-simple", tc.Input.Name, tc.Expected)
			before, _ := json.Marshal(tc.Input)
			model := map[string]json.RawMessage{}
			for key, value := range fixture.Model {
				model[key] = value
			}
			for key, value := range tc.Input.Model {
				model[key] = value
			}
			controls := map[string]json.RawMessage{"apiKey": json.RawMessage(`"fixture-key"`)}
			for key, value := range tc.Input.Options {
				controls[key] = value
			}
			rawModel, _ := json.Marshal(model)
			rawOptions, _ := json.Marshal(controls)
			transcript := NormalizeContext(tc.Input.Context)
			var params json.RawMessage
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path != "/v1/messages" {
					t.Errorf("unexpected endpoint: %s", r.URL)
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				var sent, expected any
				_ = json.Unmarshal(body, &sent)
				_ = json.Unmarshal(params, &expected)
				if fields, ok := expected.(map[string]any); ok {
					delete(fields, "betas")
				}
				if !reflect.DeepEqual(sent, expected) {
					t.Error("simple payload changed in transport")
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")), Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream, admissionErr := StreamAnthropicSimple(ctx, rawModel, transcript, AnthropicStreamOptions{Options: rawOptions, Client: client, Now: func() int64 { return 100 }, OnPayload: func(_ context.Context, payload, _ json.RawMessage) (json.RawMessage, error) {
				params = append(json.RawMessage(nil), payload...)
				return nil, nil
			}})
			var result *Message
			var failure *string
			if admissionErr != nil {
				message := admissionErr.Error()
				failure = &message
				if stream != nil || params != nil {
					t.Fatal("credential error admitted a stream")
				}
			} else {
				if err := stream.WaitForEnd(ctx); err != nil {
					t.Fatal(err)
				}
				result, err = stream.SnapshotResult(ctx)
				if err != nil {
					t.Fatal(err)
				}
			}
			actual, err := json.Marshal(map[string]any{"params": params, "result": result, "error": failure})
			if err != nil {
				t.Fatal(err)
			}
			decode := func(raw []byte) any {
				t.Helper()
				var value any
				d := json.NewDecoder(bytes.NewReader(raw))
				d.UseNumber()
				if err := d.Decode(&value); err != nil {
					t.Fatal(err)
				}
				return value
			}
			if !reflect.DeepEqual(decode(actual), decode(expected)) {
				t.Fatalf("simple mismatch\nGo: %s\nPi: %s", actual, expected)
			}
			after, _ := json.Marshal(tc.Input)
			if !bytes.Equal(before, after) {
				t.Fatal("simple provider mutated inputs")
			}
		})
	}
}
