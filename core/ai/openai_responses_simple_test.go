package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPiResponsesSimpleOracle(t *testing.T) {
	testResponsesSimpleOracle(t, "testdata/pi-responses-simple.json", 103, BuildOpenAIResponsesSimpleOptions, StreamOpenAIResponsesSimple)
}
func testResponsesSimpleOracle(t *testing.T, fixturePath string, count int, build func(json.RawMessage, TranscriptContext, json.RawMessage) (json.RawMessage, error), streamFn func(context.Context, json.RawMessage, TranscriptContext, OpenAIResponsesStreamOptions) (*AssistantMessageEventStream, error)) {
	updates := readPiReconUpdates(t)
	fixtureName := strings.TrimSuffix(filepath.Base(fixturePath), ".json")
	for _, key := range []string{"AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"} {
		t.Setenv(key, "")
	}

	t.Setenv("PI_CACHE_RETENTION", "")
	raw, err := os.ReadFile(fixturePath)
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
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != count {
		t.Fatal("unexpected Responses simple oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			expected := updates.simpleExpectation(t, fixtureName, tc.Input.Name, tc.Expected)
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
			prepared, err := build(rawModel, transcript, rawOptions)
			if err != nil {
				t.Fatal(err)
			}
			var params json.RawMessage
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path != "/v1/responses" {
					t.Errorf("unexpected endpoint: %s", r.URL)
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				var sent, expected any
				_ = json.Unmarshal(body, &sent)
				_ = json.Unmarshal(params, &expected)
				if !reflect.DeepEqual(sent, expected) {
					t.Error("simple payload changed in transport")
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"status\":\"completed\"}}\n\ndata: [DONE]\n\n")), Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream, admissionErr := streamFn(ctx, rawModel, transcript, OpenAIResponsesStreamOptions{Options: rawOptions, Client: client, Now: func() int64 { return 100 }, OnPayload: func(_ context.Context, payload, _ json.RawMessage) (json.RawMessage, error) {
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
			actual, err := json.Marshal(map[string]any{"options": prepared, "params": params, "result": result, "error": failure})
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
