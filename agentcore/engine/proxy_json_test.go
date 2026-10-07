package engine_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func TestPiProxyJSONAdmission(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-proxy-json.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input    struct{ Body, Stage string }
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 186 {
		t.Fatal("unexpected proxy JSON coverage")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i)+"/"+tc.Input.Stage, func(t *testing.T) {
			closed := false
			client := &http.Client{Transport: proxyTransport(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Status: "200 OK", Body: &proxyJSONBody{Reader: strings.NewReader(tc.Input.Body), closed: &closed}, Header: http.Header{}}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"test","api":"test","provider":"test"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.test", AuthToken: "key", Client: client, Now: func() int64 { return 1000 }})
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
			if !closed {
				t.Fatal("proxy body was not closed")
			}
			encoded, err := json.Marshal(map[string]any{"events": events, "result": result})
			if err != nil {
				t.Fatal(err)
			}
			// Preserve lone UTF-16 code units in diagnostics during comparison.
			actual, err := jsonjs.DecodeJSON(encoded)
			if err != nil {
				t.Fatal(err)
			}
			expected, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("Go: %s\nPi: %s", encoded, tc.Expected)
			}
		})
	}
}

type proxyJSONBody struct {
	io.Reader
	closed *bool
}

func (r *proxyJSONBody) Close() error { *r.closed = true; return nil }
