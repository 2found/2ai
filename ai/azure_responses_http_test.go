package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestPiAzureResponsesHTTP(t *testing.T) {
	testOpenAIHTTPOracle(t, "testdata/pi-azure-responses-http.json", 68, StreamAzureResponses)
}
func TestAzureResponsesHTTPTimeoutAndAbort(t *testing.T) {
	testOpenAIHTTPTimeoutAndAbort(t, "azure-openai-responses", StreamAzureResponses)
}

func TestPiAzureResponsesSimple(t *testing.T) {
	testResponsesSimpleOracle(t, "testdata/pi-azure-responses-simple.json", 106, BuildAzureResponsesSimpleOptions, StreamAzureResponsesSimple)
}

func TestPiAzureResponsesEndpoints(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-azure-responses-endpoints.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, SDKVersion string
		Cases                      []struct {
			Model, Options, Result json.RawMessage
			Requests               []struct{ URL, Redirect string }
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.SDKVersion != "7.19.0" || len(fixture.Cases) != 36 {
		t.Fatal("unexpected endpoint oracle")
	}
	for _, key := range []string{"AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"} {
		t.Setenv(key, "")
	}
	for i, tc := range fixture.Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			requests := []string{}
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				requests = append(requests, r.URL.String())
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")), Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			stream := StreamAzureResponses(ctx, tc.Model, NormalizeContext(Context{}), AzureResponsesStreamOptions{Options: tc.Options, Client: client, Now: func() int64 { return 100 }})
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := json.Marshal(result)
			if err != nil {
				t.Fatal(err)
			}
			if len(requests) != len(tc.Requests) {
				t.Fatalf("requests %v; result %s", requests, actual)
			}
			for j, request := range tc.Requests {
				if requests[j] != request.URL {
					t.Errorf("Go URL %q; Pi %q", requests[j], request.URL)
				}
			}
			if !reflect.DeepEqual(diagnosticValue(t, actual), diagnosticValue(t, tc.Result)) {
				t.Fatalf("Go %s; Pi %s", actual, tc.Result)
			}
		})
	}
}

func TestAzureResponsesRedirectPolicy(t *testing.T) {
	for _, key := range []string{"AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"} {
		t.Setenv(key, "")
	}
	for _, tc := range []struct {
		name, headers string
		follow        bool
	}{{"default", "{}", false}, {"override", `{"api-key":"override"}`, false}, {"omitted", `{"api-key":null}`, true}, {"empty with bearer", `{"api-key":"","authorization":"Bearer custom"}`, false}} {
		t.Run(tc.name, func(t *testing.T) {
			var initial, target, redirects atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/target" {
					target.Add(1)
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
					return
				}
				initial.Add(1)
				w.Header().Set("Location", "/target")
				w.WriteHeader(http.StatusTemporaryRedirect)
			}))
			defer server.Close()
			client := server.Client()
			client.CheckRedirect = func(*http.Request, []*http.Request) error { redirects.Add(1); return nil }
			model, _ := json.Marshal(map[string]any{"id": "test", "api": "azure-openai-responses", "provider": "azure-openai-responses", "baseUrl": server.URL})
			options := json.RawMessage(`{"apiKey":"key","headers":` + tc.headers + `}`)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			stream := StreamAzureResponses(ctx, model, NormalizeContext(Context{}), AzureResponsesStreamOptions{Options: options, Client: client})
			if err := stream.WaitForEnd(ctx); err != nil {
				t.Fatal(err)
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			want := int32(0)
			reason := "error"
			if tc.follow {
				want = 1
				reason = "stop"
			}
			if initial.Load() != 1 || target.Load() != want || redirects.Load() != want || result.StopReason != reason {
				t.Fatalf("initial=%d target=%d redirects=%d result=%+v", initial.Load(), target.Load(), redirects.Load(), result)
			}
		})
	}
}
