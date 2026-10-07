package engine_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func TestPiProxyModelIdentity(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-proxy-identity.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Mode, Body string
			Model, Expected  json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 39 {
		t.Fatal("unexpected proxy identity oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var requestModel json.RawMessage
			client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
				var request struct{ Model json.RawMessage }
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					return nil, err
				}
				requestModel = request.Model
				status, line := 200, "200 OK"
				if tc.Mode == "http-error" {
					status, line = 502, "502 Bad Gateway"
				}
				return &http.Response{StatusCode: status, Status: line, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(tc.Body)), Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream := engine.StreamProxy(ctx, tc.Model, ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example", Client: client, Now: func() int64 { return 123 }})
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
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
			actual, err := json.Marshal(map[string]any{"events": events, "result": result, "requestModel": requestModel})
			if err != nil {
				t.Fatal(err)
			}
			got, err := jsonjs.DecodeJSON(actual)
			if err != nil {
				t.Fatal(err)
			}
			want, err := jsonjs.DecodeJSON(tc.Expected)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
