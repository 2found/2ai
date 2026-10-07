package ask_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/ask"
)

func TestAskToolProperties(t *testing.T) {
	tool := ask.Tool{}
	if tool.Name() != "ask" {
		t.Errorf("expected name 'ask', got %q", tool.Name())
	}
	if !tool.SelfGated() {
		t.Error("expected ask to be SelfGated")
	}
	if !tool.RetrySafe() {
		t.Error("expected ask to be RetrySafe")
	}
	schema := tool.Schema()
	if schema.Name != "ask" {
		t.Errorf("schema name: %q", schema.Name)
	}
	req, ok := schema.Parameters["required"].([]string)
	if !ok || len(req) != 1 || req[0] != "question" {
		t.Errorf("expected required [question], got %v", schema.Parameters["required"])
	}
}

func TestAskToolClamping(t *testing.T) {
	tool := ask.Tool{}

	// Oversized question + too many options + oversized labels.
	longQ := strings.Repeat("q", 3000)
	opts := make([]map[string]any, 15)
	for i := range opts {
		opts[i] = map[string]any{
			"label":       strings.Repeat("l", 300),
			"description": strings.Repeat("d", 300),
		}
	}
	raw, err := json.Marshal(map[string]any{
		"question": longQ,
		"options":  opts,
		"multi":    true,
	})
	if err != nil {
		t.Fatal(err)
	}

	prepped := tool.PrepareArguments(string(raw))
	var out struct {
		Question string `json:"question"`
		Options  []struct {
			Label       string `json:"label"`
			Description string `json:"description"`
		} `json:"options"`
		Multi bool `json:"multi"`
	}
	if err := json.Unmarshal([]byte(prepped), &out); err != nil {
		t.Fatalf("prepared arguments invalid: %v", err)
	}

	if len(out.Question) != 2000 {
		t.Errorf("expected question clamped to 2000, got %d", len(out.Question))
	}
	if len(out.Options) != 8 {
		t.Errorf("expected options clamped to 8, got %d", len(out.Options))
	}
	for i, o := range out.Options {
		if len(o.Label) != 200 {
			t.Errorf("opt %d: label len %d, want 200", i, len(o.Label))
		}
		if len(o.Description) != 200 {
			t.Errorf("opt %d: desc len %d, want 200", i, len(o.Description))
		}
	}
	if !out.Multi {
		t.Error("expected multi=true preserved")
	}
}

func TestAskClampsAtUTF8BoundariesAndIsIdempotent(t *testing.T) {
	raw, err := json.Marshal(map[string]any{
		"question": strings.Repeat("q", 1999) + "三",
		"options":  []any{map[string]string{"label": strings.Repeat("l", 199) + "🙂", "description": strings.Repeat("d", 199) + "ü"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tool := ask.Tool{}
	prepared := tool.PrepareArguments(string(raw))
	if strings.Contains(prepared, `\ufffd`) || tool.PrepareArguments(prepared) != prepared {
		t.Fatalf("normalization damaged Unicode or changed twice: %s", prepared)
	}
	var out struct {
		Question string
		Options  []struct{ Label, Description string }
	}
	if err := json.Unmarshal([]byte(prepared), &out); err != nil {
		t.Fatal(err)
	}
	if !utf8.ValidString(out.Question) || len(out.Question) != 1999 || len(out.Options[0].Label) != 199 || len(out.Options[0].Description) != 199 {
		t.Fatal("clamp split a code point")
	}
}

func TestAskToolRunReturnsErrParked(t *testing.T) {
	tool := ask.Tool{}
	ctx := context.Background()

	// Empty question is rejected before park.
	_, err := tool.Run(ctx, `{"question": "   "}`)
	if err == nil || errors.Is(err, agentcore.ErrParked) {
		t.Errorf("expected validation error for empty question, got %v", err)
	}

	// Valid question returns ErrParked.
	_, err = tool.Run(ctx, `{"question": "Which tier?", "options": [{"label": "pro"}]}`)
	if !errors.Is(err, agentcore.ErrParked) {
		t.Errorf("expected ErrParked, got %v", err)
	}
}
