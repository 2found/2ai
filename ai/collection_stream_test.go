package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

// sseChatServer serves one streamed assistant message saying label, so two
// providers registered under the same model id stay tellable apart.
func sseChatServer(t *testing.T, label string, hit *[]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*hit = append(*hit, label)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + label + "\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"))
	}))
}

// StreamOn exists because a tier can name both a provider id and a model: when
// two providers publish the same model id, routing by model alone would send the
// request to whichever registered first. Pinning the provider must beat that.
func TestStreamOn_RoutesToTheNamedProviderNotTheModelOwner(t *testing.T) {
	var hits []string
	srvA := sseChatServer(t, "A", &hits)
	defer srvA.Close()
	srvB := sseChatServer(t, "B", &hits)
	defer srvB.Close()

	a, err := New(Spec{ID: "pa", Vendor: "openai-compat", APIKey: "ka", BaseURL: srvA.URL, HTTP: srvA.Client()})
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(Spec{ID: "pb", Vendor: "openai-compat", APIKey: "kb", BaseURL: srvB.URL, HTTP: srvB.Client()})
	if err != nil {
		t.Fatal(err)
	}
	col := NewCollection()
	col.Register(a)
	col.Register(b)

	ch, err := col.StreamOn(context.Background(), "pb", protocol.ChatRequest{
		Model:    "same-id",
		Messages: []protocol.Message{{Role: protocol.RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatalf("StreamOn: %v", err)
	}
	var got strings.Builder
	for d := range ch {
		if d.Err != nil {
			t.Fatalf("delta error: %v", d.Err)
		}
		got.WriteString(d.ContentDelta)
	}
	if got.String() != "B" {
		t.Fatalf("streamed %q, want B", got.String())
	}
	for _, h := range hits {
		if h == "A" {
			t.Fatalf("stream leaked to provider A: %v", hits)
		}
	}
}

// An unknown provider id must fail loudly. Falling back to model-based routing
// would silently defeat the point of naming a provider.
func TestStreamOn_UnknownProviderIsAnError(t *testing.T) {
	col := NewCollection()
	_, err := col.StreamOn(context.Background(), "nope", protocol.ChatRequest{Model: "m"})
	if err == nil {
		t.Fatal("StreamOn on an unregistered provider returned no error")
	}
	if !strings.Contains(err.Error(), "nope") {
		t.Fatalf("err = %v, want it to name the unknown provider", err)
	}
}

// Collection satisfies protocol.LLMProvider so it can be handed to the loop
// directly. Both identity methods are part of that contract: the loop uses
// SupportsTools to decide whether to advertise a toolset at all.
func TestCollection_SatisfiesTheProviderIdentityContract(t *testing.T) {
	var _ protocol.LLMProvider = NewCollection()
	col := NewCollection()
	if col.Name() != "collection" {
		t.Fatalf("Name() = %q, want collection", col.Name())
	}
	if !col.SupportsTools() {
		t.Fatal("SupportsTools() = false; the loop would drop the toolset")
	}
}

// A rotated BYO token must reach the Anthropic wire, same as the OpenAI one —
// this is the protocol.KeyUpdater contract the loop calls between turns.
func TestAnthropicProvider_UpdateAPIKey(t *testing.T) {
	p := NewAnthropicProvider("old", "")
	var _ protocol.KeyUpdater = p

	p.UpdateAPIKey("new")
	if p.APIKey != "new" {
		t.Fatalf("APIKey = %q, want new", p.APIKey)
	}
	// An empty key is a no-op; treating it as a set would leave the provider
	// unauthenticated after a failed refresh.
	p.UpdateAPIKey("")
	if p.APIKey != "new" {
		t.Fatalf("empty update erased the key: %q", p.APIKey)
	}
}
