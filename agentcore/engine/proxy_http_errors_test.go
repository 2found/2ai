package engine_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

type proxyErrorBody struct{ mode string }

func (r proxyErrorBody) Read([]byte) (int, error) {
	if r.mode == "read-panic" {
		panic("unreadable body")
	}
	return 0, errors.New("unreadable body")
}
func (proxyErrorBody) Close() error { return nil }

func TestPiProxyHTTPErrorBodies(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-proxy-http-errors.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, ReadFailure string
			BodyBase64        []byte
			Expected          json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 31 {
		t.Fatal("unexpected proxy HTTP oracle")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			client := &http.Client{Transport: proxyTransport(func(r *http.Request) (*http.Response, error) {
				var body io.ReadCloser = io.NopCloser(bytes.NewReader(tc.BodyBase64))
				if tc.ReadFailure != "" {
					body = proxyErrorBody{tc.ReadFailure}
				}
				return &http.Response{StatusCode: 502, Status: "502 Bad Gateway", Header: make(http.Header), Body: body, Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream := engine.StreamProxy(ctx, json.RawMessage(`{"id":"m","api":"test","provider":"test"}`), ai.NormalizeContext(ai.Context{}), engine.ProxyStreamOptions{ProxyURL: "https://proxy.example", Client: client})
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			if result.StopReason != "error" || result.ErrorMessage == nil {
				t.Fatalf("unexpected result: %+v", result)
			}
			actual := jsonjs.QuoteString(*result.ErrorMessage)
			if !bytes.Equal(actual, tc.Expected) {
				t.Fatalf("Go: %s\nPi: %s", actual, tc.Expected)
			}
		})
	}
}
