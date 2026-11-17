package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

func TestClampReasoningEffortSendsTheNearestServedLevel(t *testing.T) {
	cases := []struct{ model, effort, want string }{
		{"gpt-5.5", "max", "xhigh"},
		{"gpt-5.5", "minimal", "low"},
		{"gpt-6-sol", "max", "max"},
		{"gpt-5.1-codex-mini", "low", "medium"},
		{"gpt-5.3-chat-latest", "high", ""},
		{"deepseek-v4-flash", "medium", "low"},
		{"deepseek-v4-flash", "xhigh", "high"},
		{"openrouter/deepseek/deepseek-v4-pro", "medium", "low"},
		{"gemini-3-pro-preview", "medium", "low"},
		{"gemini-3-pro-preview", "max", "high"},
		{"claude-opus-4-6", "xhigh", "high"},
		{"claude-sonnet-4-6", "max", "high"},
		{"claude-opus-5-5", "minimal", "low"},
		{"us.anthropic.claude-opus-4-6-v1", "xhigh", "high"},
		{"glm-5.2", "low", "high"},
		{"grok-4", "high", ""},
		{"grok-4.20-0309-reasoning", "low", ""},
		{"grok-4.7", "max", "xhigh"},
		{"deepseek-v4-flash", "none", "none"},
		{"deepseek-v4-flash", "off", "off"},
		{"deepseek-v4-flash", "", ""},
		{"llama-3-70b", "max", "max"},
	}
	for _, c := range cases {
		if got := ClampReasoningEffort("", c.model, c.effort); got != c.want {
			t.Errorf("ClampReasoningEffort(%q, %q) = %q, want %q", c.model, c.effort, got, c.want)
		}
	}
}

// Every list is weakest first and names only levels a host may ask for, so
// "the strongest below" is well defined.
func TestReasoningEffortsAreLevelsInOrder(t *testing.T) {
	for prefix, levels := range reasoningEfforts {
		last := -1
		for _, level := range levels {
			rank := slices.Index(ReasoningEffortLevels, level)
			if rank <= last || level == "none" {
				t.Errorf("%s: %v is not a weakest-first list of ReasoningEffortLevels", prefix, levels)
			}
			last = rank
		}
	}
}

func TestReasoningEffortsForIsACopy(t *testing.T) {
	levels, ok := ReasoningEffortsFor("", "gpt-6-sol")
	if !ok || len(levels) != 5 {
		t.Fatalf("gpt-6-sol: %v %v", levels, ok)
	}
	levels[0] = "max"
	if again, _ := ReasoningEffortsFor("", "gpt-6-sol"); again[0] != "low" {
		t.Fatalf("the table was changed through a result: %v", again)
	}
	if levels, ok := ReasoningEffortsFor("", "grok-build"); !ok || levels == nil || len(levels) != 0 {
		t.Fatalf("grok-build reasons without a level: %#v %v", levels, ok)
	}
	if levels, ok := ReasoningEffortsFor("", "my-local-model"); ok || levels != nil {
		t.Fatalf("an unknown model is unknown: %v %v", levels, ok)
	}
}

func TestLongContextThresholdFor(t *testing.T) {
	for model, want := range map[string]int{
		"gpt-5.5": 272_000, "gpt-5-codex": 272_000, "openai/gpt-5.4": 272_000,
		"gemini-2.5-pro": 200_000, "google/gemini-3-pro-preview": 200_000, "gemini-1.5-pro-002": 128_000,
		"gpt-6-sol": 272_000, "claude-haiku-5-5": 100_000, "us.anthropic.claude-haiku-5-5": 100_000,
		"grok-4.20-0309-reasoning": 200_000, "grok-4.7": 200_000, "grok-build": 200_000, "grok-4": 0, "grok-4-1-fast": 0,
		"MiniMax-M3": 512_000, "qwen3.7-plus": 256_000,
		"gemini-2.5-flash": 0, "gpt-4.1": 0, "deepseek-flash": 0, "claude-opus-5-5": 0, "claude-haiku-4-5": 0, "": 0,
	} {
		if got := LongContextThresholdFor("", model); got != want {
			t.Errorf("%s: %d, want %d", model, got, want)
		}
	}
}

// A listed model carries the table's levels and price step; a model the table
// does not know says null, not "no levels".
func TestListModelsCarriesEffortsAndThreshold(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"data":[{"id":"gpt-5.5"},{"id":"who-knows"}]}`))
	}))
	defer srv.Close()
	p, err := New(Spec{Vendor: "openai", BaseURL: srv.URL, APIKey: "k", HTTP: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) != 2 {
		t.Fatalf("%v %v", models, err)
	}
	if !slices.Equal(models[0].ReasoningEfforts, []string{"low", "medium", "high", "xhigh"}) || models[0].LongContextThreshold != 272_000 {
		t.Errorf("gpt-5.5: %+v", models[0])
	}
	if models[1].ReasoningEfforts != nil || models[1].LongContextThreshold != 0 {
		t.Errorf("unknown model: %+v", models[1])
	}
}
