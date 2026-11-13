package ai

import (
	"strings"
	"testing"
)

// TestGeminiProviderIsGoogleOnOpenAIWire pins the provider-breadth constructor:
// Gemini rides the shared OpenAI-compatible implementation, with vendor
// identity and Google's compat base URL.
func TestGeminiProviderIsGoogleOnOpenAIWire(t *testing.T) {
	p := NewGeminiProvider("k")
	if p.Name() != "google" {
		t.Fatalf("Name() = %q, want google", p.Name())
	}
	if !strings.Contains(p.BaseURL, "generativelanguage.googleapis.com") {
		t.Fatalf("BaseURL = %q", p.BaseURL)
	}
	if !p.SupportsTools() {
		t.Fatal("gemini compat must advertise tool support")
	}
	p.UpdateAPIKey("k2")
	if p.APIKey != "k2" {
		t.Fatal("KeyUpdater must swap the key")
	}
	// The stock provider's identity is unchanged by the vendor field's existence.
	if NewOpenAIProvider("k", "", DefaultCompat()).Name() != "openai" {
		t.Fatal("stock provider must stay \"openai\"")
	}
}
