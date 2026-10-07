package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func devinTestJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(raw []byte) string { return base64.RawURLEncoding.EncodeToString(raw) }
	return encode([]byte(`{"alg":"HS256","typ":"JWT"}`)) + "." + encode(payload) + "." + encode([]byte("signature"))
}

func TestDevinOAuthExchangeMinimalBodyAndJWTExpiry(t *testing.T) {
	expiry := time.Now().Add(48 * time.Hour).Truncate(time.Second)
	token := devinTestJWT(t, map[string]any{"exp": expiry.Unix()})

	var captured []byte
	var contentType string
	options := DevinOAuthOptions{
		Client: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			raw, err := io.ReadAll(request.Body)
			if err != nil {
				return nil, err
			}
			captured, contentType = raw, request.Header.Get("Content-Type")
			body, _ := json.Marshal(map[string]string{"token": token})
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(string(body))), Request: request}, nil
		})},
		tokenURL:     "https://token.test/exchange",
		Now:          func() float64 { return float64(time.Now().UnixMilli()) },
		PKCE:         GeneratePKCE,
		authorizeURL: "https://authorize.test/cli",
	}
	value, err := devinOAuthExchange(context.Background(), "the-code", "the-verifier", options)
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "application/json" {
		t.Fatalf("content type %q", contentType)
	}
	// The CLI exchange carries only the code and its PKCE verifier: no
	// grant_type, client_id, redirect_uri or state.
	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatal(err)
	}
	if len(body) != 2 || body["code"] != "the-code" || body["code_verifier"] != "the-verifier" {
		t.Fatalf("exchange body = %s", captured)
	}

	object, ok := value.(*Object)
	if !ok {
		t.Fatalf("credential type %T", value)
	}
	if got := object.Get("access"); got != token {
		t.Fatalf("access = %v", got)
	}
	if got := object.Get("refresh"); got != token {
		t.Fatalf("refresh = %v", got)
	}
	if got := object.Get("type"); got != "oauth" {
		t.Fatalf("type = %v", got)
	}
	if got, _ := object.Get("expires").(float64); int64(got) != expiry.UnixMilli()-5*60*1000 {
		t.Fatalf("expires = %v, want %d", got, expiry.UnixMilli())
	}
}

func TestDevinOAuthExpiryFallback(t *testing.T) {
	now := float64(time.Now().UnixMilli())
	options := DevinOAuthOptions{Now: func() float64 { return now }}
	got := devinTokenExpiryMS("not-a-jwt", options)
	if int64(got-now) != devinOAuthFallbackLifetimeMS {
		t.Fatalf("fallback lifetime = %d, want %d", int64(got-now), devinOAuthFallbackLifetimeMS)
	}
	// A JWT with no exp claim also falls back rather than expiring immediately.
	noExp := devinTestJWT(t, map[string]any{"session_id": "s"})
	if got := devinTokenExpiryMS(noExp, options); int64(got-now) != devinOAuthFallbackLifetimeMS {
		t.Fatalf("missing exp lifetime = %d", int64(got-now))
	}
}

func TestDevinOAuthToAuthAndRefresh(t *testing.T) {
	auth := DevinOAuth()
	if auth.Name != "Devin" {
		t.Fatalf("name %q", auth.Name)
	}
	credential := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "access", Value: "session-token"}, Property{Name: "expires", Value: float64(1)})
	value, err := auth.ToAuth(credential)
	if err != nil {
		t.Fatal(err)
	}
	object, ok := value.(*Object)
	if !ok || object.Get("apiKey") != "session-token" {
		t.Fatalf("ToAuth = %v", value)
	}
	refreshed, err := auth.Refresh(context.Background(), credential)
	if err != nil {
		t.Fatal(err)
	}
	if refreshed.(*Object).Get("access") != "session-token" {
		t.Fatal("refresh must return the session credential unchanged (no refresh grant)")
	}
}
