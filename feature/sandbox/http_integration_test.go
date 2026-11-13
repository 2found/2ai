package sandbox_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/feature/credential"
	"github.com/2found/2ai/feature/sandbox"
)

// TestEndToEndCredentialReachesServerNotTrace is the worked-consumer proof: the
// model emits {{cred:API_KEY}} in an Authorization header, the vault resolves it
// at the trust boundary, the http_request tool sends the *real* bearer to the
// destination — and the persisted trace still shows only the placeholder.
func TestEndToEndCredentialReachesServerNotTrace(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	vault := credential.NewVault()
	if err := vault.Put("API_KEY", "sk-real-secret"); err != nil {
		t.Fatalf("Put: %v", err)
	}

	tool := sandbox.NewHTTPRequestTool(nil,
		sandbox.WithHTTPAllowHosts([]string{"127.0.0.1"}),
		sandbox.WithHTTPAllowPlain(true), // httptest serves plain http
	)
	tool.AllowAllIPsForTest() // httptest is on loopback, normally refused

	args, _ := json.Marshal(map[string]any{
		"method":  "GET",
		"url":     srv.URL,
		"headers": map[string]string{"Authorization": "Bearer {{cred:API_KEY}}"},
	})

	env := agentcore.DefaultEnv()
	env.Credentials = vault
	env.CredentialGrants = []agentcore.CredentialGrant{{
		Name: "API_KEY", Tools: []string{sandbox.ToolHTTPRequest}, Hosts: []string{"127.0.0.1"},
	}}
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: httpCallProvider(args),
		Model:          "test",
		Tools:          agentcore.NewToolSet(tool),
		Policy:         agentcore.NewAllowList(sandbox.ToolHTTPRequest),
		Env:            &env,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "fetch it")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// The destination received the resolved secret.
	if gotAuth != "Bearer sk-real-secret" {
		t.Fatalf("server Authorization = %q, want resolved bearer", gotAuth)
	}
	// The trace kept the placeholder — the secret never persisted.
	if len(res.Tools) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(res.Tools))
	}
	if !strings.Contains(res.Tools[0].Args, "{{cred:API_KEY}}") {
		t.Fatalf("trace should keep the placeholder, got %q", res.Tools[0].Args)
	}
	if strings.Contains(res.Tools[0].Args, "sk-real-secret") {
		t.Fatal("secret leaked into the tool trace")
	}
}

// TestEndToEndBlocksNonAllowlistedHost confirms the tool refuses an off-allowlist
// host and feeds the reason back to the model rather than making the call.
func TestEndToEndBlocksNonAllowlistedHost(t *testing.T) {
	tool := sandbox.NewHTTPRequestTool(nil, sandbox.WithHTTPAllowHosts([]string{"api.allowed.com"}))
	args, _ := json.Marshal(map[string]any{"url": "https://evil.example.com/steal"})
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: httpCallProvider(args),
		Model:          "test",
		Tools:          agentcore.NewToolSet(tool),
		Policy:         agentcore.NewAllowList(sandbox.ToolHTTPRequest),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	var sawErr bool
	for _, m := range res.Messages {
		if m.Role == agentcore.RoleTool && strings.Contains(m.Content, "allowlist") {
			sawErr = true
		}
	}
	if !sawErr {
		t.Fatal("expected the allowlist refusal to be returned to the model")
	}
}

func httpCallProvider(args json.RawMessage) *ai.FallbackProvider {
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{
		Model: json.RawMessage(`{"id":"test","contextWindow":128000}`),
		Stream: ai.ScriptedStream(
			ai.Message{Role: "assistant", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{
				Type: "toolCall", ID: "c1", Name: sandbox.ToolHTTPRequest, Arguments: args,
			})},
			ai.Message{Role: "assistant", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "done"})},
		),
	}}}
}

// A host-scoped grant does not follow the call elsewhere: the same credential
// aimed at a host the grant does not name is refused before it is resolved.
func TestCredentialGrantRefusesOtherHosts(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	vault := credential.NewVault()
	if err := vault.Put("API_KEY", "sk-real-secret"); err != nil {
		t.Fatalf("Put: %v", err)
	}
	tool := sandbox.NewHTTPRequestTool(nil,
		sandbox.WithHTTPAllowHosts([]string{"127.0.0.1"}),
		sandbox.WithHTTPAllowPlain(true),
	)
	tool.AllowAllIPsForTest()
	args, _ := json.Marshal(map[string]any{
		"method":  "GET",
		"url":     srv.URL,
		"headers": map[string]string{"Authorization": "Bearer {{cred:API_KEY}}"},
	})
	env := agentcore.DefaultEnv()
	env.Credentials = vault
	env.CredentialGrants = []agentcore.CredentialGrant{{
		Name: "API_KEY", Tools: []string{sandbox.ToolHTTPRequest}, Hosts: []string{"api.example.com"},
	}}
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: httpCallProvider(args),
		Model:          "test",
		Tools:          agentcore.NewToolSet(tool),
		Policy:         agentcore.NewAllowList(sandbox.ToolHTTPRequest),
		Env:            &env,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, _ := agent.Prompt(context.Background(), "fetch it")
	if hits != 0 {
		t.Fatalf("the request reached a host the grant does not name (%d hits)", hits)
	}
	for _, tr := range res.Tools {
		if strings.Contains(tr.Error, "sk-real-secret") || strings.Contains(tr.Args, "sk-real-secret") {
			t.Fatal("secret leaked into the trace")
		}
	}
}
