package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
)

type radiusConfigHTTPCase struct {
	Input struct {
		Name, Body, Action, Gateway string
		Status                      int
		Key                         json.RawMessage
	}
	Requests, Output json.RawMessage
}
type radiusCatalogFixture struct {
	UpstreamCommit                                  string
	Baseline, Identity, Native, Wiring, Integration json.RawMessage
	Cases                                           []struct{ Input, Output json.RawMessage }
	HTTP                                            []radiusConfigHTTPCase
	ProviderCases                                   []struct {
		Input struct {
			Gateway *string
			Mode    string
		}
		Initial, Identity, Log, Output, Models json.RawMessage
	}
}

func readRadiusCatalogFixture(t *testing.T) radiusCatalogFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-radius-catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	var f radiusCatalogFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(f.Cases) != 86 || len(f.HTTP) != 76 || len(f.ProviderCases) != 80 || len(f.Integration) == 0 {
		t.Fatal("unexpected Radius catalog coverage")
	}
	return f
}
func radiusFixtureModel(t *testing.T, f radiusCatalogFixture) *Object {
	return authSpread(catalogProperty(catalogDecode(t, f.Baseline), "a"))
}
func TestPiRadiusConfig(t *testing.T) {
	f := readRadiusCatalogFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			credential := NewObject(Property{Name: "gatewayConfig", Value: catalogDecode(t, tc.Input)})
			var config any = Undefined
			if value := GetRadiusCredentialConfig(credential); value != nil {
				config = value
			}
			catalogCompare(t, NewObject(Property{Name: "config", Value: config}, Property{Name: "models", Value: GetRadiusModels("custom", credential)}), tc.Output)
		})
	}
}
func TestPiRadiusConfigIdentityAndNativeValues(t *testing.T) {
	f := readRadiusCatalogFixture(t)
	original := radiusFixtureModel(t, f)
	original.Delete("api")
	original.Delete("provider")
	original.Delete("baseUrl")
	original.Set("input", NewArray("text"))
	original.Set("cost", NewObject(Property{Name: "input", Value: 1}))
	original.Set("extra", NewObject(Property{Name: "keep", Value: true}))
	credential := NewObject(Property{Name: "gatewayConfig", Value: NewObject(Property{Name: "baseUrl", Value: "https://inference.test/messages"}, Property{Name: "models", Value: NewArray(original)})})
	c1, c2 := GetRadiusCredentialConfig(credential), GetRadiusCredentialConfig(credential)
	first, second := c1.Get("models").(*Array), c2.Get("models").(*Array)
	row, row2 := first.Get(0).(*Object), second.Get(0).(*Object)
	projected, err := GetRadiusModelsFromConfig("p", c1)
	if err != nil {
		t.Fatal(err)
	}
	mapped := projected.Get(0).(*Object)
	identity := map[string]any{"freshConfig": c1 != c2, "freshArray": first != second, "freshRows": row != original && row != row2, "sharedInput": row.Get("input") == original.Get("input"), "sharedCost": row.Get("cost") == original.Get("cost"), "sharedExtra": row.Get("extra") == original.Get("extra"), "projectedCopy": mapped != row, "projectedSharedCost": mapped.Get("cost") == row.Get("cost")}
	row.Set("name", "changed")
	original.Get("cost").(*Object).Set("input", 9)
	identity["after"] = map[string]any{"source": original, "first": row, "second": row2, "projected": mapped}
	catalogCompare(t, identity, f.Identity)
	native := NewArray()
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		model := radiusFixtureModel(t, f)
		model.Set("contextWindow", value)
		model.Set("maxTokens", value)
		models := GetRadiusModels("p", NewObject(Property{Name: "gatewayConfig", Value: NewObject(Property{Name: "baseUrl", Value: "x"}, Property{Name: "models", Value: NewArray(model)})}))
		number, _ := catalogKey(value, make(map[*Array]bool))
		native.Append(NewObject(Property{Name: "kind", Value: number}, Property{Name: "count", Value: models.Len()}, Property{Name: "context", Value: number}, Property{Name: "max", Value: number}))
	}
	sparse := NewArray()
	sparse.SetLength(3)
	model := radiusFixtureModel(t, f)
	model.Delete("api")
	model.Delete("provider")
	model.Delete("baseUrl")
	sparse.Set(1, model)
	models, err := GetRadiusModelsFromConfig("p", NewObject(Property{Name: "baseUrl", Value: "x"}, Property{Name: "models", Value: sparse}))
	if err != nil {
		t.Fatal(err)
	}
	native.Append(NewObject(Property{Name: "kind", Value: "sparse"}, Property{Name: "length", Value: models.Len()}, Property{Name: "keys", Value: NewArray("1")}, Property{Name: "value", Value: models}))
	catalogCompare(t, native, f.Native)
}
func TestPiRadiusConfigHTTP(t *testing.T) {
	for i, tc := range readRadiusCatalogFixture(t).HTTP {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runRadiusConfigHTTP(t, tc, false) })
	}
}
func TestPiRadiusConfigRealHTTP(t *testing.T) {
	for i, tc := range readRadiusCatalogFixture(t).HTTP {
		if tc.Input.Name != "success" && !strings.HasPrefix(tc.Input.Name, "status:") && !strings.HasPrefix(tc.Input.Name, "raw:") {
			continue
		}
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) { runRadiusConfigHTTP(t, tc, true) })
	}
}
func runRadiusConfigHTTP(t *testing.T, tc radiusConfigHTTPCase, realHTTP bool) {
	t.Helper()
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	failure, reason := errors.New("injected failure"), errors.New("original cause")
	if tc.Input.Action == "preabort" {
		cancel(reason)
	}
	requests := NewArray()
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		headers := NewObject(Property{Name: "accept", Value: r.Header.Get("Accept")})
		if value, exists := r.Header["Authorization"]; exists {
			headers.Set("authorization", value[0])
		}
		requests.Append(NewObject(Property{Name: "url", Value: r.URL.String()}, Property{Name: "method", Value: r.Method}, Property{Name: "headers", Value: headers}, Property{Name: "aborted", Value: r.Context().Err() != nil}))
		if strings.HasPrefix(tc.Input.Action, "fetch-abort") {
			cancel(reason)
		}
		if tc.Input.Action == "fetch-error" || tc.Input.Action == "fetch-abort-error" {
			return nil, failure
		}
		reader := strings.NewReader(tc.Input.Body)
		return &http.Response{StatusCode: tc.Input.Status, Header: make(http.Header), Body: &oauthTestBody{read: func(p []byte) (int, error) {
			if strings.HasPrefix(tc.Input.Action, "read-abort") {
				cancel(reason)
			}
			if tc.Input.Action == "read-error" || tc.Input.Action == "read-abort-error" {
				return 0, failure
			}
			return reader.Read(p)
		}}}, nil
	})
	client := &http.Client{Transport: transport}
	if realHTTP {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.URL.Scheme = "https"
			r.URL.Host = "gateway.test"
			response, err := transport.RoundTrip(r)
			if err != nil {
				t.Error(err)
				w.WriteHeader(500)
				return
			}
			defer response.Body.Close()
			w.WriteHeader(response.StatusCode)
			if _, err = io.Copy(w, response.Body); err != nil {
				t.Error(err)
			}
		}))
		defer server.Close()
		endpoint, err := url.Parse(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		local := server.Client()
		client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			clone := r.Clone(r.Context())
			copyURL := *r.URL
			copyURL.Scheme, copyURL.Host = endpoint.Scheme, endpoint.Host
			clone.URL = &copyURL
			return local.Transport.RoundTrip(clone)
		})}
	}
	config, err := LoadRadiusGatewayConfig(ctx, tc.Input.Gateway, catalogDecode(t, tc.Input.Key), client)
	output := map[string]any{"value": config}
	if err != nil {
		output = map[string]any{"error": err.Error(), "same": err == failure}
	}
	catalogCompare(t, output, tc.Output)
	catalogCompare(t, requests, tc.Requests)
}
