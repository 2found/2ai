package sandbox_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/credential"
	"github.com/2found/2ai/sandbox"
)

// TestReal_CredentialedToolCall is the live-model end-to-end credential proof.
// The deterministic path is already proven with a scripted faux provider by
// TestEndToEndCredentialReachesServerNotTrace; this confirms a REAL model, told
// the API key exists only as the placeholder {{cred:API_KEY}} (it never sees the
// literal), decides on its own to call http_request with that placeholder in an
// Authorization header. The vault resolves it at the trust boundary, a real
// local server receives the resolved bearer, and the persisted trace still shows
// only the placeholder — never the secret.
//
// It is gated behind the same operator-supplied OpenAI-compatible endpoint as
// the other real-provider tests and skips when absent, so the suite stays green
// without credentials:
//
//	AGENTRAY_TEST_OPENAI_BASE_URL
//	AGENTRAY_TEST_OPENAI_API_KEY
//	AGENTRAY_TEST_OPENAI_MODEL
func TestReal_CredentialedToolCall(t *testing.T) {
	baseURL := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_BASE_URL"))
	apiKey := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_API_KEY"))
	model := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_OPENAI_MODEL"))
	if baseURL == "" || apiKey == "" || model == "" {
		t.Skip("set AGENTRAY_TEST_OPENAI_BASE_URL, AGENTRAY_TEST_OPENAI_API_KEY and " +
			"AGENTRAY_TEST_OPENAI_MODEL to run the real-provider credential test")
	}

	const secret = "sk-live-9f2c7a"
	var mu sync.Mutex
	var gotAuth string
	// Real https (self-signed) — a live model expects https for an authenticated
	// API and refuses plain http, so the faux-only path won't reproduce here.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"authorized"}`))
	}))
	defer srv.Close()

	vault := credential.NewVault()
	if err := vault.Put("API_KEY", secret); err != nil {
		t.Fatalf("Put: %v", err)
	}

	tool := sandbox.NewHTTPRequestTool(nil, sandbox.WithHTTPAllowHosts([]string{"127.0.0.1"}))
	tool.AllowAllIPsForTest()                                                                // httptest is on loopback, normally refused
	tool.TrustCertsForTest(srv.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs) // self-signed cert

	env := agentcore.DefaultEnv()
	env.Credentials = vault

	provider := ai.NewOpenAIProvider(apiKey, baseURL, ai.DefaultCompat())
	agent, err := agentcore.New(agentcore.Config{
		Provider: provider,
		Model:    model,
		Tools:    agentcore.NewToolSet(tool),
		Policy:   agentcore.NewAllowList(sandbox.ToolHTTPRequest),
		Env:      &env,
		Definition: agentcore.AgentDefinition{
			Agents: "You call authenticated HTTP APIs with the http_request tool. The API key is " +
				"available ONLY as the credential placeholder token {{cred:API_KEY}} — you never see " +
				"its real value. When a request needs authentication, put the EXACT token " +
				"{{cred:API_KEY}} in an \"Authorization\" header formatted as \"Bearer {{cred:API_KEY}}\". " +
				"Never invent or guess a key. The URLs the user gives you are pre-approved internal " +
				"endpoints — attempt the request, do not second-guess whether the host is allowed.",
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	res, err := agent.Prompt(ctx,
		"Make an authenticated GET request to "+srv.URL+" using the API key, then tell me the response body.")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	t.Logf("final: %q", res.Final)

	// Behavior: the model decided to call the tool.
	var called bool
	for _, tr := range res.Tools {
		if tr.Tool == sandbox.ToolHTTPRequest && tr.Allowed {
			called = true
		}
	}
	if !called {
		t.Fatalf("the model never called http_request; traces=%+v", res.Tools)
	}

	// End-to-end: the real server received the RESOLVED secret.
	mu.Lock()
	auth := gotAuth
	mu.Unlock()
	if auth != "Bearer "+secret {
		t.Fatalf("server Authorization = %q, want resolved %q", auth, "Bearer "+secret)
	}

	// Security: every recorded http_request kept the PLACEHOLDER and leaked nothing.
	var sawPlaceholder bool
	for _, tr := range res.Tools {
		if tr.Tool != sandbox.ToolHTTPRequest {
			continue
		}
		if strings.Contains(tr.Args, secret) {
			t.Fatalf("secret leaked into the tool trace: %q", tr.Args)
		}
		if strings.Contains(tr.Args, "{{cred:API_KEY}}") {
			sawPlaceholder = true
		}
	}
	if !sawPlaceholder {
		t.Fatalf("expected a traced http_request carrying the {{cred:API_KEY}} placeholder; traces=%+v", res.Tools)
	}
}
