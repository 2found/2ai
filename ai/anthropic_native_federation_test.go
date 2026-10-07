package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func clearAnthropicFederationEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID", "PI_CACHE_RETENTION"} {
		t.Setenv(key, "")
	}
}

func TestPiAnthropicFederationOracle(t *testing.T) {
	clearAnthropicFederationEnv(t)
	raw, err := os.ReadFile("testdata/pi-anthropic-federation.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, SDKVersion string
		Model                      map[string]json.RawMessage
		IgnoredHeaders             []string
		Cases                      []struct {
			Input struct {
				Name           string
				Calls          int
				Model, Options map[string]json.RawMessage
				Identity       *string
				IdentityRepeat int
				TokenStatus    int
				TokenBody      *string
				APIStatuses    []int
				Oversized      bool
				RotateIdentity bool
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.SDKVersion != "0.129.0" || len(fixture.Cases) != 33 {
		t.Fatal("unexpected federation oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "identity")
			identity := "  fixture-assertion\n"
			if tc.Input.Identity != nil {
				identity = *tc.Input.Identity
			}
			if tc.Input.IdentityRepeat > 0 {
				identity = strings.Repeat("x", tc.Input.IdentityRepeat)
			}
			if err := os.WriteFile(path, []byte(identity), 0600); err != nil {
				t.Fatal(err)
			}
			model := map[string]json.RawMessage{}
			for k, v := range fixture.Model {
				model[k] = v
			}
			for k, v := range tc.Input.Model {
				model[k] = v
			}
			headers, _ := samplingObject(fixture.Model["headers"])
			extra, _ := samplingObject(tc.Input.Model["headers"])
			for k, v := range extra {
				headers[k] = v
			}
			model["headers"], _ = json.Marshal(headers)
			controls := map[string]json.RawMessage{}
			for k, v := range tc.Input.Options {
				controls[k] = v
			}
			env := map[string]json.RawMessage{"ANTHROPIC_FEDERATION_RULE_ID": json.RawMessage(`"rule"`), "ANTHROPIC_ORGANIZATION_ID": json.RawMessage(`"organization"`)}
			env["ANTHROPIC_IDENTITY_TOKEN_FILE"], _ = json.Marshal(path)
			overrides, _ := samplingObject(controls["env"])
			for k, v := range overrides {
				env[k] = v
			}
			controls["env"], _ = json.Marshal(env)
			rawModel, _ := json.Marshal(model)
			rawOptions, _ := json.Marshal(controls)
			requests := []any{}
			results := []any{}
			exchanges, apiCalls := 0, 0
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				headers := map[string]string{}
				for k, v := range r.Header {
					if len(v) > 0 {
						headers[strings.ToLower(k)] = strings.Join(v, ", ")
					}
				}
				for _, key := range fixture.IgnoredHeaders {
					delete(headers, key)
				}
				requests = append(requests, map[string]any{"path": r.URL.Path, "query": r.URL.RawQuery, "headers": headers, "body": json.RawMessage(body)})
				status := 200
				text := ""
				responseHeaders := http.Header{}
				if strings.HasSuffix(r.URL.Path, "/oauth/token") {
					exchanges++
					text = fmt.Sprintf(`{"access_token":"access-%d","expires_in":3600}`, exchanges)
					if tc.Input.TokenBody != nil {
						text = *tc.Input.TokenBody
					}
					if tc.Input.Oversized {
						text = strings.Repeat("x", (1<<20)+1)
					}
					if tc.Input.TokenStatus != 0 {
						status = tc.Input.TokenStatus
					}
					responseHeaders.Set("Request-Id", "token-request")
				} else {
					if apiCalls < len(tc.Input.APIStatuses) {
						status = tc.Input.APIStatuses[apiCalls]
					}
					apiCalls++
					text = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
					if status >= 300 {
						text = `{"error":{"message":"try again"}}`
					}
					responseHeaders.Set("Content-Type", "text/event-stream")
					responseHeaders.Set("Retry-After-Ms", "0")
				}
				return &http.Response{StatusCode: status, Header: responseHeaders, Body: io.NopCloser(strings.NewReader(text)), Request: r}, nil
			})}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			for i := 0; i < max(1, tc.Input.Calls); i++ {
				if i == 1 && tc.Input.RotateIdentity {
					if err := os.WriteFile(path, []byte("rotated-assertion"), 0600); err != nil {
						t.Fatal(err)
					}
				}
				stream, err := StreamAnthropicSimple(ctx, rawModel, NormalizeContext(Context{}), AnthropicStreamOptions{Options: rawOptions, Client: client, Now: func() int64 { return 1000000 }})
				var result *Message
				var failure *string
				if err != nil {
					value := err.Error()
					failure = &value
				} else {
					if err := stream.WaitForEnd(ctx); err != nil {
						t.Fatal(err)
					}
					result, err = stream.SnapshotResult(ctx)
					if err != nil {
						t.Fatal(err)
					}
				}
				results = append(results, map[string]any{"result": result, "error": failure})
			}
			actual, err := json.Marshal(map[string]any{"requests": requests, "results": results})
			if err != nil {
				t.Fatal(err)
			}
			actual = bytes.ReplaceAll(actual, []byte(path), []byte("<identity-file>"))
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
