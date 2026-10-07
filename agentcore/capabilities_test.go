package agentcore

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
)

type capabilityProbeProvider struct {
	name     string
	caps     ModelCapabilities
	response ChatResponse
	calls    int32
	recorded []ChatRequest
}

func (p *capabilityProbeProvider) Name() string        { return p.name }
func (p *capabilityProbeProvider) SupportsTools() bool { return true }
func (p *capabilityProbeProvider) ModelCapabilities(string) ModelCapabilities {
	return p.caps
}
func (p *capabilityProbeProvider) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	atomic.AddInt32(&p.calls, 1)
	p.recorded = append(p.recorded, req)
	return p.response, nil
}
func (p *capabilityProbeProvider) Stream(context.Context, ChatRequest) (<-chan ChatDelta, error) {
	return nil, errors.New("unused")
}

type capabilityProbeTool struct{}

func (capabilityProbeTool) Name() string { return "probe" }
func (capabilityProbeTool) Schema() ToolSchema {
	return ToolSchema{Name: "probe", Parameters: map[string]any{"type": "object"}}
}
func (capabilityProbeTool) Run(context.Context, string) (string, error) { return "ok", nil }


func TestRequestForCapabilitiesStripsOnlyExplicitlyUnsupportedHints(t *testing.T) {
	schema := &OutputSchema{Name: "answer", Schema: map[string]any{"type": "object"}}
	req := ChatRequest{
		ReasoningEffort: "high", OutputSchema: schema,
		CacheKey: "session-1", CacheRetention: "long",
	}
	unsupported := &capabilityProbeProvider{name: "limited", caps: ModelCapabilities{
		ReasoningEffort:  CapabilityUnsupported,
		StructuredOutput: CapabilityUnsupported,
		PromptCaching:    CapabilityUnsupported,
	}}
	got, err := requestForCapabilities(unsupported, "m", ModelCapabilities{}, req)
	if err != nil {
		t.Fatalf("requestForCapabilities: %v", err)
	}
	if got.ReasoningEffort != "" || got.OutputSchema != nil || got.CacheKey != "" || got.CacheRetention != "" {
		t.Fatalf("unsupported hints were not stripped: %+v", got)
	}

	unknown := &capabilityProbeProvider{name: "unknown"}
	got, err = requestForCapabilities(unknown, "m", ModelCapabilities{}, req)
	if err != nil {
		t.Fatalf("unknown requestForCapabilities: %v", err)
	}
	if got.ReasoningEffort != req.ReasoningEffort || got.OutputSchema != schema ||
		got.CacheKey != req.CacheKey || got.CacheRetention != req.CacheRetention {
		t.Fatalf("unknown capabilities changed a backward-compatible request: %+v", got)
	}
}

func TestRequestForCapabilitiesValidatesAndDegradesToolChoice(t *testing.T) {
	tools := []ToolSchema{{Name: "query", Parameters: map[string]any{"type": "object"}}}
	supported := &capabilityProbeProvider{name: "supported", caps: ModelCapabilities{ToolChoice: CapabilitySupported}}
	named := ChatRequest{Tools: tools, ToolChoice: ToolChoice{Mode: ToolChoiceNamed, Name: "query"}}
	got, err := requestForCapabilities(supported, "m", ModelCapabilities{}, named)
	if err != nil || got.ToolChoice != named.ToolChoice {
		t.Fatalf("supported named choice = %+v err=%v", got.ToolChoice, err)
	}

	missing := named
	missing.ToolChoice.Name = "absent"
	if _, err := requestForCapabilities(supported, "m", ModelCapabilities{}, missing); err == nil || !strings.Contains(err.Error(), "not present") {
		t.Fatalf("missing named choice error = %v", err)
	}
	if _, err := requestForCapabilities(supported, "m", ModelCapabilities{}, ChatRequest{
		ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
	}); err == nil || !strings.Contains(err.Error(), "at least one tool") {
		t.Fatalf("required-without-tools error = %v", err)
	}

	unsupported := &capabilityProbeProvider{name: "legacy", caps: ModelCapabilities{ToolChoice: CapabilityUnsupported}}
	parallel := false
	got, err = requestForCapabilities(unsupported, "m", ModelCapabilities{}, ChatRequest{
		Tools: tools, ToolChoice: ToolChoice{Mode: ToolChoiceNone}, ParallelToolCalls: &parallel,
	})
	if err != nil || len(got.Tools) != 0 || got.ToolChoice != (ToolChoice{}) || got.ParallelToolCalls != nil {
		t.Fatalf("unsupported none choice was not locally enforced: %+v err=%v", got, err)
	}
	unsupported.caps.Tools = CapabilityUnsupported
	got, err = requestForCapabilities(unsupported, "m", ModelCapabilities{}, ChatRequest{
		Tools: tools, ToolChoice: ToolChoice{Mode: ToolChoiceNone}, ParallelToolCalls: &parallel,
	})
	if err != nil || len(got.Tools) != 0 {
		t.Fatalf("text-only none choice was not locally enforced: %+v err=%v", got, err)
	}
	unsupported.caps.Tools = CapabilityUnknown
	if _, err := requestForCapabilities(unsupported, "m", ModelCapabilities{}, named); err == nil {
		t.Fatal("unsupported forced choice should fail for escalation")
	} else {
		var capabilityErr *ModelCapabilityError
		if !errors.As(err, &capabilityErr) || capabilityErr.Feature != "forced tool choice" {
			t.Fatalf("forced choice error = %T %v", err, err)
		}
	}
}

func TestRequestForCapabilitiesRejectsUnknownToolStrictness(t *testing.T) {
	provider := &capabilityProbeProvider{name: "supported", caps: ModelCapabilities{Tools: CapabilitySupported}}
	_, err := requestForCapabilities(provider, "m", ModelCapabilities{}, ChatRequest{Tools: []ToolSchema{{
		Name: "query", Strict: ToolStrictness("sometimes"), Parameters: map[string]any{"type": "object"},
	}}})
	if err == nil || !strings.Contains(err.Error(), `tool "query"`) || !strings.Contains(err.Error(), "unknown tool strictness") {
		t.Fatalf("invalid strictness error = %v", err)
	}
}


func TestToolChoiceValidation(t *testing.T) {
	cases := []struct {
		choice  ToolChoice
		wantErr bool
	}{
		{ToolChoice{}, false},
		{ToolChoice{Mode: ToolChoiceAuto}, false},
		{ToolChoice{Mode: ToolChoiceNone}, false},
		{ToolChoice{Mode: ToolChoiceRequired}, false},
		{ToolChoice{Mode: ToolChoiceNamed, Name: "query"}, false},
		{ToolChoice{Mode: ToolChoiceNamed}, true},
		{ToolChoice{Mode: ToolChoiceAuto, Name: "query"}, true},
		{ToolChoice{Mode: "sometimes"}, true},
	}
	for _, tc := range cases {
		if err := tc.choice.Validate(); (err != nil) != tc.wantErr {
			t.Fatalf("Validate(%+v) = %v, wantErr=%v", tc.choice, err, tc.wantErr)
		}
	}
}

func TestRequestForCapabilitiesClampsOnlyExplicitOutputLimit(t *testing.T) {
	limited := &capabilityProbeProvider{name: "limited", caps: ModelCapabilities{MaxOutputTokens: 4096}}
	got, err := requestForCapabilities(limited, "m", ModelCapabilities{}, ChatRequest{MaxTokens: 32000})
	if err != nil || got.MaxTokens != 4096 {
		t.Fatalf("clamped request = %+v err=%v, want 4096", got, err)
	}
	got, err = requestForCapabilities(limited, "m", ModelCapabilities{}, ChatRequest{})
	if err != nil || got.MaxTokens != 0 {
		t.Fatalf("provider-default request = %+v err=%v, want MaxTokens=0", got, err)
	}
}

func TestRequestForCapabilitiesDegradesImagesCopyOnWrite(t *testing.T) {
	original := []Message{{
		Role: RoleTool, Content: "plot", ToolCallID: "c1",
		ContentParts: []ContentPart{
			{Type: ContentPartText, Text: "caption"},
			{Type: ContentPartImage, MIMEType: "image/png", Data: "aW1hZ2U="},
		},
	}}
	req := ChatRequest{Messages: original}
	limited := &capabilityProbeProvider{name: "text-only", caps: ModelCapabilities{ImageInput: CapabilityUnsupported}}
	got, err := requestForCapabilities(limited, "m", ModelCapabilities{}, req)
	if err != nil {
		t.Fatalf("requestForCapabilities: %v", err)
	}
	if len(got.Messages[0].ContentParts) != 1 || got.Messages[0].ContentParts[0].Type != ContentPartText {
		t.Fatalf("degraded parts = %+v", got.Messages[0].ContentParts)
	}
	if got.Messages[0].Content != "plot\n[1 image output(s) omitted: provider/model path does not support image input]" {
		t.Fatalf("degraded content = %q", got.Messages[0].Content)
	}
	if len(original[0].ContentParts) != 2 || original[0].Content != "plot" {
		t.Fatalf("capability filtering mutated escalation source: %+v", original[0])
	}

	unknown := &capabilityProbeProvider{name: "unknown"}
	kept, err := requestForCapabilities(unknown, "m", ModelCapabilities{}, req)
	if err != nil || len(kept.Messages[0].ContentParts) != 2 {
		t.Fatalf("unknown image capability should preserve request: %+v err=%v", kept, err)
	}
}

func TestRequestForCapabilitiesClampsOldestImagesToRungLimit(t *testing.T) {
	image := func(data string) ContentPart {
		return ContentPart{Type: ContentPartImage, MIMEType: "image/png", Data: data}
	}
	original := []Message{
		{Role: RoleTool, ToolCallID: "c1", Content: "first", ContentParts: []ContentPart{image("1"), {Type: ContentPartText, Text: "keep"}, image("2")}},
		{Role: RoleTool, ToolCallID: "c2", Content: "second", ContentParts: []ContentPart{image("3"), image("4")}},
		{Role: RoleTool, ToolCallID: "c3", Content: "third", ContentParts: []ContentPart{image("5"), image("6")}},
	}
	provider := &capabilityProbeProvider{name: "limited-vision", caps: ModelCapabilities{
		ImageInput: CapabilitySupported, MaxInputImages: 3,
	}}
	got, err := requestForCapabilities(provider, "m", ModelCapabilities{}, ChatRequest{Messages: original})
	if err != nil {
		t.Fatal(err)
	}
	count := func(messages []Message) int {
		n := 0
		for _, message := range messages {
			for _, part := range message.ContentParts {
				if part.Type == ContentPartImage {
					n++
				}
			}
		}
		return n
	}
	if count(got.Messages) != 3 || count(original) != 6 {
		t.Fatalf("image counts: clamped=%d original=%d", count(got.Messages), count(original))
	}
	if len(got.Messages[0].ContentParts) != 1 || got.Messages[0].ContentParts[0].Type != ContentPartText {
		t.Fatalf("non-image part was not preserved: %+v", got.Messages[0].ContentParts)
	}
	if !strings.Contains(got.Messages[0].Content, "2 older image attachment(s) omitted") ||
		!strings.Contains(got.Messages[1].Content, "1 older image attachment(s) omitted") {
		t.Fatalf("omissions were not visible: %+v", got.Messages)
	}
	if got.Messages[1].ContentParts[0].Data != "4" || got.Messages[2].ContentParts[0].Data != "5" {
		t.Fatalf("did not retain the newest images in order: %+v", got.Messages)
	}
}

func TestRequestForCapabilitiesUsesSafeImageFloorWhenLimitUnknown(t *testing.T) {
	parts := make([]ContentPart, 6)
	for i := range parts {
		parts[i] = ContentPart{Type: ContentPartImage, MIMEType: "image/png", Data: string(rune('a' + i))}
	}
	original := []Message{{Role: RoleUser, ContentParts: parts}}
	got, err := requestForCapabilities(&capabilityProbeProvider{name: "unknown"}, "m", ModelCapabilities{}, ChatRequest{Messages: original})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages[0].ContentParts) != unknownProviderImageLimit || got.Messages[0].ContentParts[0].Data != "b" {
		t.Fatalf("unknown provider floor did not drop the oldest image: %+v", got.Messages[0])
	}
	if len(original[0].ContentParts) != 6 {
		t.Fatal("image clamp mutated canonical history")
	}
}

func TestCapabilitiesOfBridgesLegacyProviderToolsFlag(t *testing.T) {
	legacy := NewFauxProvider(AssistantText("ok"))
	if got := CapabilitiesOf(legacy, "m").Tools; got != CapabilitySupported {
		t.Fatalf("legacy tools capability = %q, want supported", got)
	}
}
