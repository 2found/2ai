package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Each vendor authenticates its list-models call differently and reads a
// different response shape. Getting the header wrong yields a 401 that looks
// like a bad user key, so pin both halves per vendor.
func TestListModelsForVendor_PerVendorWireContract(t *testing.T) {
	cases := []struct {
		vendor    string
		wantPath  string
		checkAuth func(*testing.T, *http.Request)
		body      string
		want      []string
	}{
		{
			vendor:   "openai",
			wantPath: "/models",
			checkAuth: func(t *testing.T, r *http.Request) {
				if got := r.Header.Get("Authorization"); got != "Bearer key-1" {
					t.Errorf("Authorization = %q", got)
				}
			},
			body: `{"data":[{"id":"gpt-5"},{"id":"  "},{"id":"o3"}]}`,
			want: []string{"gpt-5", "o3"}, // blank ids are dropped
		},
		{
			vendor:   "anthropic",
			wantPath: "/v1/models",
			checkAuth: func(t *testing.T, r *http.Request) {
				if got := r.Header.Get("x-api-key"); got != "key-1" {
					t.Errorf("x-api-key = %q", got)
				}
				if got := r.Header.Get("anthropic-version"); got != anthropicVersion {
					t.Errorf("anthropic-version = %q", got)
				}
			},
			body: `{"data":[{"id":"claude-sonnet-4-6"}]}`,
			want: []string{"claude-sonnet-4-6"},
		},
		{
			vendor:   "google",
			wantPath: "/v1beta/models",
			checkAuth: func(t *testing.T, r *http.Request) {
				if got := r.URL.Query().Get("key"); got != "key-1" {
					t.Errorf("key query param = %q", got)
				}
				if got := r.Header.Get("x-goog-api-key"); got != "key-1" {
					t.Errorf("x-goog-api-key = %q", got)
				}
			},
			// Gemini returns fully-qualified resource names; the "models/" prefix
			// is not part of the id callers use.
			body: `{"models":[{"name":"models/gemini-3-pro","outputTokenLimit":65536},{"name":"gemini-3-flash"}]}`,
			want: []string{"gemini-3-pro", "gemini-3-flash"},
		},
	}

	for _, c := range cases {
		t.Run(c.vendor, func(t *testing.T) {
			var gotPath string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				c.checkAuth(t, r)
				w.Write([]byte(c.body))
			}))
			defer srv.Close()

			got, err := listModelsForVendor(context.Background(), srv.Client(), c.vendor, srv.URL, "key-1")
			if err != nil {
				t.Fatalf("listModelsForVendor: %v", err)
			}
			if gotPath != c.wantPath {
				t.Errorf("path = %q, want %q", gotPath, c.wantPath)
			}
			if strings.Join(modelIDs(got), ",") != strings.Join(c.want, ",") {
				t.Fatalf("models = %v, want %v", modelIDs(got), c.want)
			}
		})
	}
}

func TestListGoogleModelsCarriesOutputTokenLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"name":"models/gemini-3-pro","inputTokenLimit":1000000,"outputTokenLimit":65536}]}`))
	}))
	defer srv.Close()
	models, err := listGoogleModels(context.Background(), srv.Client(), srv.URL, "k")
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ContextWindow != 1000000 || models[0].Capabilities.MaxOutputTokens != 65536 {
		t.Fatalf("models = %+v", models)
	}
}

// An unknown vendor is assumed OpenAI-compatible — that is the whole premise of
// the compat wire — so it must reach GET /models rather than error out.
func TestListModelsForVendor_UnknownVendorUsesTheOpenAICompatWire(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Write([]byte(`{"data":[{"id":"plus"}]}`))
	}))
	defer srv.Close()

	got, err := listModelsForVendor(context.Background(), srv.Client(), "9router", srv.URL, "k")
	if err != nil {
		t.Fatalf("listModelsForVendor: %v", err)
	}
	if path != "/models" || len(got) != 1 || got[0].ID != "plus" {
		t.Fatalf("path=%q models=%v; want the OpenAI-compat wire", path, got)
	}
}

// modelIDs projects a listing to its ids, for assertions that are only about
// which models came back.
func modelIDs(ms []listedModel) []string {
	out := make([]string, 0, len(ms))
	for _, m := range ms {
		out = append(out, m.ID)
	}
	return out
}

// A Google base URL already pointing at the OpenAI-compatible surface must not
// have /v1beta/models appended — that path does not exist there.
func TestListGoogleModels_OpenAICompatBaseURLUsesTheCompatWire(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Write([]byte(`{"data":[{"id":"gemini-3-pro"}]}`))
	}))
	defer srv.Close()

	got, err := listGoogleModels(context.Background(), srv.Client(), srv.URL+"/v1beta/openai", "k")
	if err != nil {
		t.Fatalf("listGoogleModels: %v", err)
	}
	if !strings.HasSuffix(path, "/openai/models") {
		t.Fatalf("path = %q, want the compat /models endpoint", path)
	}
	if len(got) != 1 || got[0].ID != "gemini-3-pro" {
		t.Fatalf("models = %v", got)
	}
}

// A base URL that is already a list-models URL must not gain a second /models.
func TestListGoogleModels_DoesNotDoubleAppendModels(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Write([]byte(`{"models":[]}`))
	}))
	defer srv.Close()

	if _, err := listGoogleModels(context.Background(), srv.Client(), srv.URL+"/v1beta/models", "k"); err != nil {
		t.Fatalf("listGoogleModels: %v", err)
	}
	if strings.Contains(path, "/models/models") {
		t.Fatalf("path = %q, want no doubled /models segment", path)
	}
}

// An error status must carry the server's explanation. Returning a bare status
// leaves an operator guessing at a bad key versus a wrong base URL.
func TestListModels_ErrorStatusIncludesTheServerMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"invalid api key"}`))
	}))
	defer srv.Close()

	_, err := listOpenAIModels(context.Background(), srv.Client(), srv.URL, "bad")
	if err == nil {
		t.Fatal("401 returned no error")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Fatalf("err = %v, want the status and the server message", err)
	}
}

// A non-JSON body (an HTML error page from a proxy, say) must be reported as a
// decode failure rather than silently yielding an empty catalog, which reads as
// "this provider has no models".
func TestListModels_NonJSONBodyIsADecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`<html>502 Bad Gateway</html>`))
	}))
	defer srv.Close()

	_, err := listOpenAIModels(context.Background(), srv.Client(), srv.URL, "k")
	if err == nil {
		t.Fatal("an HTML body decoded as an empty model list")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Fatalf("err = %v, want a decode error", err)
	}
}

// doJSON must tolerate a nil client by falling back to the package default,
// since every list helper accepts an optional client.
func TestDoJSON_NilClientFallsBackToTheDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"data":[]}`))
	}))
	defer srv.Close()

	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	_, status, err := doJSON(context.Background(), nil, req)
	if err != nil {
		t.Fatalf("doJSON with a nil client: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("status = %d", status)
	}
}
