package engine_test

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

func TestPiProxyUnicode(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-proxy-unicode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Body string
			Expected   json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 32 {
		t.Fatal("unexpected proxy unicode oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Status: "200 OK", Header: make(http.Header), Body: &proxyByteReader{data: []byte(tc.Body)}, Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m","api":"test","provider":"test"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example", Client: client, Now: func() int64 { return 123 }})
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
			actual, err := json.Marshal(map[string]any{"events": events, "result": result})
			if err != nil {
				t.Fatal(err)
			}
			// Decode with UTF-16 retention: encoding/json would replace both lone
			// surrogates and could hide data loss in the production implementation.
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
