package ai

import (
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// TestNewClientResolvesVendors checks NewClient maps config→wire client with
// zero loop edits: built-in vendors resolve, OpenAI-compatible vendors route
// through the OpenAI provider when given a base_url + compat, and an unknown
// vendor with neither is a hard error (§12 AC).
func TestNewClientResolvesVendors(t *testing.T) {
	cases := []struct {
		name     string
		spec     ClientSpec
		wantName string
		wantErr  bool
	}{
		{"default empty -> openai", ClientSpec{}, "openai", false},
		{"explicit openai", ClientSpec{Name: "openai", APIKey: "k"}, "openai", false},
		{"openai responses", ClientSpec{Name: "openai-responses", APIKey: "k"}, VendorOpenAIResponses, false},
		{"case-insensitive", ClientSpec{Name: "OpenAI"}, "openai", false},
		{"anthropic", ClientSpec{Name: "anthropic", APIKey: "k"}, "anthropic", false},
		{"google", ClientSpec{Name: "google", APIKey: "k"}, "google", false},
		{"gemini alias", ClientSpec{Name: "Gemini", APIKey: "k"}, "google", false},
		{
			// Compat vendors keep their own identity so traces and per-turn key
			// refresh attribute to the vendor's tier, not "openai".
			"compatible vendor via base_url+compat",
			ClientSpec{Name: "groq", BaseURL: "https://api.groq.com/openai/v1", Compat: Compat{MaxTokensField: "max_tokens"}},
			"groq", false,
		},
		{"unknown vendor, no compat", ClientSpec{Name: "mystery"}, "", true},
		{"unknown vendor, compat but no base_url", ClientSpec{Name: "mystery", Compat: Compat{MaxTokensField: "max_tokens"}}, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewClient(tc.spec)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for %+v", tc.spec)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if p.Name() != tc.wantName {
				t.Errorf("provider name = %q, want %q", p.Name(), tc.wantName)
			}
		})
	}
}

func TestNewClientSelectsResponsesWireWithoutChangingOpenAIIdentity(t *testing.T) {
	p, err := NewClient(ClientSpec{
		Name: "openai", APIKey: "sk-test", BaseURL: "https://gateway.example/v1",
		OpenAIWire: OpenAIWireResponses,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	responses, ok := p.(*OpenAIResponsesProvider)
	if !ok {
		t.Fatalf("provider = %T, want *OpenAIResponsesProvider", p)
	}
	if responses.Name() != "openai" {
		t.Fatalf("Name() = %q, want provider identity openai", responses.Name())
	}
	if responses.APIKey != "sk-test" || responses.BaseURL != "https://gateway.example/v1" {
		t.Fatalf("responses provider = %+v", responses)
	}
	if got := responses.ModelCapabilities("gpt-test").StatefulResponses; got != protocol.CapabilitySupported {
		t.Fatalf("stateful responses capability = %q, want supported", got)
	}
}

func TestNewClientRejectsUnknownOpenAIWire(t *testing.T) {
	_, err := NewClient(ClientSpec{Name: "openai", OpenAIWire: OpenAIWire("mystery")})
	if err == nil || !strings.Contains(err.Error(), "unknown OpenAI wire") {
		t.Fatalf("err = %v, want unknown OpenAI wire", err)
	}
}
