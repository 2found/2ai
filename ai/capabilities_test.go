package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestWiredListModelsOverlaysAndRetainsLiveCapabilities(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{
			"id":"text-only",
			"supports_tools":false,
			"supports_reasoning":true,
			"max_input_images":7,
			"max_output_tokens":8192,
			"capabilities":{"prompt_caching":"supported"},
			"supported_parameters":["response_format"],
			"architecture":{"input_modalities":["text"]}
		}]}`))
	}))
	defer srv.Close()

	provider, err := New(Spec{ID: "row", Vendor: "openai", BaseURL: srv.URL, HTTP: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models = %d, want 1", len(models))
	}
	caps := models[0].Capabilities
	if caps.Tools != protocol.CapabilityUnsupported ||
		caps.ReasoningEffort != protocol.CapabilitySupported ||
		caps.StructuredOutput != protocol.CapabilitySupported ||
		caps.PromptCaching != protocol.CapabilitySupported ||
		caps.ImageInput != protocol.CapabilityUnsupported || caps.MaxInputImages != 7 || caps.MaxOutputTokens != 8192 {
		t.Fatalf("listed capabilities = %+v", caps)
	}

	capable, ok := provider.(protocol.ModelCapabilityProvider)
	if !ok {
		t.Fatal("wired provider lost ModelCapabilityProvider")
	}
	retained := capable.ModelCapabilities("text-only")
	if retained.Tools != protocol.CapabilityUnsupported || retained.ReasoningEffort != protocol.CapabilitySupported {
		t.Fatalf("retained capabilities = %+v", retained)
	}
}

func TestDiscoveredCapabilitiesNeverTreatsMissingMetadataAsUnsupported(t *testing.T) {
	got := (discoveredCapabilities{}).modelCapabilities()
	if got != (protocol.ModelCapabilities{}) {
		t.Fatalf("empty discovery = %+v, want entirely unknown", got)
	}
}

func TestDedicatedCapabilityBooleanWinsOverGenericHints(t *testing.T) {
	no := false
	got := (discoveredCapabilities{
		SupportsTools:       &no,
		SupportedParameters: []string{"tools"},
	}).modelCapabilities()
	if got.Tools != protocol.CapabilityUnsupported {
		t.Fatalf("tools = %q, explicit supports_tools:false must win", got.Tools)
	}
}

func TestCapabilityOverlayUsesOnlyKnownNewValues(t *testing.T) {
	base := protocol.ModelCapabilities{
		Tools:           protocol.CapabilitySupported,
		PromptCaching:   protocol.CapabilitySupported,
		MaxInputImages:  90,
		MaxOutputTokens: 8192,
	}
	got := base.Overlay(protocol.ModelCapabilities{Tools: protocol.CapabilityUnsupported})
	if got.Tools != protocol.CapabilityUnsupported || got.PromptCaching != protocol.CapabilitySupported || got.MaxInputImages != 90 || got.MaxOutputTokens != 8192 {
		t.Fatalf("overlay = %+v", got)
	}
	got = got.Overlay(protocol.ModelCapabilities{MaxInputImages: 12})
	if got.MaxInputImages != 12 {
		t.Fatalf("numeric capability did not overlay: %+v", got)
	}
	got = got.Overlay(protocol.ModelCapabilities{MaxOutputTokens: 4096})
	if got.MaxOutputTokens != 4096 {
		t.Fatalf("output-token capability did not overlay: %+v", got)
	}
}

func TestDiscoveredImageLimitAcceptsDedicatedAndCapabilityMapShapes(t *testing.T) {
	twelve := 12
	got := (discoveredCapabilities{
		MaxImages: &twelve,
		Capabilities: map[string]json.RawMessage{
			"max_input_images": json.RawMessage(`9`),
		},
	}).modelCapabilities()
	if got.MaxInputImages != 12 {
		t.Fatalf("dedicated max_images should win over generic map: %+v", got)
	}
}

func TestModelCapabilitiesRejectsNegativeImageLimit(t *testing.T) {
	if err := (protocol.ModelCapabilities{MaxInputImages: -1}).Validate(); err == nil {
		t.Fatal("negative image limit must be rejected")
	}
	if err := (protocol.ModelCapabilities{MaxOutputTokens: -1}).Validate(); err == nil {
		t.Fatal("negative output-token limit must be rejected")
	}
}

func TestAntigravityAdvertisesModelOutputCeilings(t *testing.T) {
	if got := CapabilitiesFor(VendorGoogleAntigravity, "claude-sonnet-4-6").MaxOutputTokens; got != 64000 {
		t.Fatalf("Claude Antigravity max output = %d, want 64000", got)
	}
	if got := CapabilitiesFor(VendorGoogleAntigravity, "gemini-3-pro").MaxOutputTokens; got != 65536 {
		t.Fatalf("Gemini Antigravity max output = %d, want 65536", got)
	}
}

func TestOpenAICompatSelectsConfiguredMaxTokenField(t *testing.T) {
	completion := NewOpenAIProvider("k", "http://local", Compat{
		MaxTokensField: "max_completion_tokens", SupportsTools: true,
	}).encode(protocol.ChatRequest{Model: "reasoning", MaxTokens: 1234})
	if completion.MaxCompletionTokens != 1234 || completion.MaxTokens != 0 {
		t.Fatalf("completion-token request = %+v", completion)
	}
	standard := NewOpenAIProvider("k", "http://local", DefaultCompat()).encode(
		protocol.ChatRequest{Model: "chat", MaxTokens: 5678},
	)
	if standard.MaxTokens != 5678 || standard.MaxCompletionTokens != 0 {
		t.Fatalf("standard-token request = %+v", standard)
	}
}

func TestAntigravityHonorsSmallerRequestedOutputLimit(t *testing.T) {
	p := NewAntigravityProvider()
	body := p.encode(protocol.ChatRequest{Model: "claude-sonnet-4-6", MaxTokens: 2048})
	if body.Request.GenerationConfig == nil || body.Request.GenerationConfig.MaxOutputTokens != 2048 {
		t.Fatalf("generation config = %+v, want maxOutputTokens=2048", body.Request.GenerationConfig)
	}
}

func TestOpenAIResponsesAdvertisesStatefulWireWithoutChangingChatCompletions(t *testing.T) {
	responses := CapabilitiesFor(VendorOpenAIResponses, "gpt-test")
	if responses.StatefulResponses != protocol.CapabilitySupported ||
		responses.StructuredOutput != protocol.CapabilitySupported ||
		responses.Tools != protocol.CapabilitySupported || responses.MaxInputImages != 200 {
		t.Fatalf("responses capabilities = %+v", responses)
	}
	if chat := CapabilitiesFor("openai", "gpt-test"); chat.StatefulResponses != protocol.CapabilityUnsupported {
		t.Fatalf("chat-completions stateful capability = %q, want unsupported", chat.StatefulResponses)
	}
}

func TestProviderImageBudgetsArePortableAndAntigravityIsExplicitlyTextOnly(t *testing.T) {
	for vendor, want := range map[string]int{
		"openai": 200, VendorOpenAIResponses: 200, "anthropic": 90,
		VendorClaudeCode: 90, VendorOpenAICodex: 200, "google": 200, "openrouter": 90,
	} {
		caps := CapabilitiesFor(vendor, "m")
		if caps.ImageInput != protocol.CapabilitySupported || caps.MaxInputImages != want {
			t.Errorf("%s image capabilities = %+v, want supported/%d", vendor, caps, want)
		}
	}
	if got := CapabilitiesFor(VendorGoogleAntigravity, "m"); got.ImageInput != protocol.CapabilityUnsupported {
		t.Fatalf("antigravity image capability = %+v", got)
	}
}
