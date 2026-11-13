package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The compaction budget is capped by this number, and the two ways of being
// wrong are not equally bad: too low compacts early (wasteful), too high never
// compacts and the provider rejects the request outright. So an id this package
// has no opinion about must answer 0 — "ask someone else" — and never a guess.
func TestContextWindowForUnknownModelAdmitsIgnorance(t *testing.T) {
	for _, id := range []string{"", "   ", "llama-3-70b", "mistral-large", "some-local-gguf"} {
		if got := ContextWindowFor("openai-compat", id); got != 0 {
			t.Errorf("ContextWindowFor(%q) = %d, want 0 so the caller falls back to its own ceiling", id, got)
		}
	}
}

// gpt-4 is the trap this table exists to survive: the bare model is 8k while
// every later member of the family is 128k or more. Shortest-prefix or
// map-iteration-order matching would hand a gpt-4o run an 8k budget (compacts
// constantly) or a gpt-4 run a 128k budget (never compacts, then dies).
func TestContextWindowForPicksTheLongestMatchingPrefix(t *testing.T) {
	cases := []struct {
		model string
		want  int
	}{
		{"gpt-4", 8_192},
		{"gpt-4-0613", 8_192},
		{"gpt-4-32k", 32_768},
		{"gpt-4-turbo-2024-04-09", 128_000},
		{"gpt-4o-mini", 128_000},
		{"gpt-4.1", 1_047_576},
		{"claude-sonnet-4-20250514", 200_000},
		{"claude-2.1", 100_000},
		{"gemini-1.5-pro-002", 2_000_000},
		{"gemini-1.0-pro", 32_768},
		{"grok-4-latest", 256_000},
		{"grok-3", 131_072},
		{"swe-2", 262_000},
		{"swe-2-high", 262_000},
		{"swe-2-medium", 262_000},
		{"swe-2-max", 262_000},
	}
	for _, c := range cases {
		if got := ContextWindowFor("", c.model); got != c.want {
			t.Errorf("ContextWindowFor(%q) = %d, want %d", c.model, got, c.want)
		}
	}
}

// A router rewrites ids on the way through ("anthropic/claude-sonnet-4") and
// Gemini returns resource names ("models/gemini-2.5-pro"). Matching the raw
// string would silently drop the window for every model reached that way — a
// miss that looks like nothing at all, because 0 is a legal answer.
func TestContextWindowForSeesThroughRoutedIDs(t *testing.T) {
	cases := map[string]int{
		"anthropic/claude-sonnet-4": 200_000,
		"models/gemini-1.5-pro":     2_000_000,
		"openai/gpt-4o":             128_000,
		"google/gemini-1.5-pro-002": 2_000_000,
		"ANTHROPIC/Claude-Sonnet-4": 200_000,
	}
	for id, want := range cases {
		if got := ContextWindowFor("", id); got != want {
			t.Errorf("ContextWindowFor(%q) = %d, want %d", id, got, want)
		}
	}
}

// Each vendor spells the window differently (or not at all). Reading only one
// name would leave the others at 0, which silently downgrades a correctly
// reported window into a fallback guess.
func TestListModelsReadsEachVendorsWindowField(t *testing.T) {
	cases := []struct {
		name   string
		vendor string
		body   string
		want   int
	}{
		{"gemini inputTokenLimit", "google",
			`{"models":[{"name":"models/x","inputTokenLimit":1048576,"outputTokenLimit":8192}]}`, 1_048_576},
		{"openrouter context_length", "openai",
			`{"data":[{"id":"x","context_length":131072}]}`, 131_072},
		{"vllm max_model_len", "openai",
			`{"data":[{"id":"x","max_model_len":32768}]}`, 32_768},
		{"openrouter top_provider", "openai",
			`{"data":[{"id":"x","top_provider":{"context_length":64000}}]}`, 64_000},
		{"max_context_length", "openai",
			`{"data":[{"id":"x","max_context_length":8000}]}`, 8_000},
		{"anthropic reports none", "anthropic",
			`{"data":[{"id":"x"}]}`, 0},
		{"openai reports none", "openai",
			`{"data":[{"id":"x"}]}`, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Write([]byte(c.body))
			}))
			defer srv.Close()

			got, err := listModelsForVendor(context.Background(), srv.Client(), c.vendor, srv.URL, "k")
			if err != nil {
				t.Fatalf("listModelsForVendor: %v", err)
			}
			if len(got) != 1 {
				t.Fatalf("got %d models, want 1", len(got))
			}
			if got[0].ContextWindow != c.want {
				t.Fatalf("ContextWindow = %d, want %d", got[0].ContextWindow, c.want)
			}
		})
	}
}

// The precedence rule: a vendor that reports its own window must beat the
// table, so a table entry can never go stale in a way that overrides live
// truth. Here the wire says 42 for an id the table would call 200k.
func TestListModelsPrefersTheVendorsOwnWindowOverTheTable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"data":[{"id":"claude-sonnet-4","context_length":42}]}`))
	}))
	defer srv.Close()

	p, err := New(Spec{Vendor: "openai-compat", BaseURL: srv.URL, APIKey: "k", HTTP: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 || models[0].ContextWindow != 42 {
		t.Fatalf("ContextWindow = %v, want the wire's 42 to beat the table's 200000", models)
	}
}

// ...and the other half of that rule: when the vendor says nothing, the table
// fills the gap rather than leaving 0. Anthropic's /v1/models never reports a
// window, so without this every Claude tier would fall through to the caller's
// raw ceiling.
func TestListModelsFallsBackToTheTableWhenTheVendorIsSilent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"data":[{"id":"claude-sonnet-4-20250514"},{"id":"who-knows"}]}`))
	}))
	defer srv.Close()

	p, err := New(Spec{Vendor: "anthropic", BaseURL: srv.URL, APIKey: "k", HTTP: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].ContextWindow != 200_000 {
		t.Errorf("claude window = %d, want the table's 200000", models[0].ContextWindow)
	}
	if models[1].ContextWindow != 0 {
		t.Errorf("unknown model window = %d, want 0 rather than an invented number", models[1].ContextWindow)
	}
}
