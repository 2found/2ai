package ai

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestDevinAuthURLShape(t *testing.T) {
	var seen string
	interaction := ProviderAuthInteraction{
		Context: context.Background(),
		Notify: func(value *Object) {
			if u, _ := value.Get("url").(string); u != "" {
				seen = u
			}
		},
		Prompt: func(ctx context.Context, value *Object) (any, error) {
			return "", context.Canceled // bail after URL is emitted
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	options := DevinOAuthOptions{
		PKCE: func() (PKCE, error) { return PKCE{Verifier: "v", Challenge: "ch"}, nil },
		StartCallback: func(o *OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
			return &OAuthCallbackServer{RedirectURI: "http://127.0.0.1:59653/callback", Wait: func() (any, error) { <-ctx.Done(); return nil, ctx.Err() }, Cancel: func() {}, Close: func() {}}, nil
		},
		Now: func() float64 { return float64(time.Now().UnixMilli()) },
	}
	interaction.Context = ctx
	go func() { time.Sleep(300 * time.Millisecond); cancel() }()
	_, _ = loginDevinOAuth(interaction, options)
	for _, want := range []string{"response_type=code", "redirect_uri=", "code_challenge=ch", "code_challenge_method=S256", "state=", "prompt=select_account"} {
		if !strings.Contains(seen, want) {
			t.Fatalf("auth URL missing %s: %s", want, seen)
		}
	}
	t.Log("url:", seen)
}
