package ai

import (
	"encoding/base64"
	"testing"
)

func TestPiOAuthPKCE(t *testing.T) {
	for _, tc := range readOAuthDeviceCodeFixture(t).PKCE {
		t.Run(tc.Kind, func(t *testing.T) {
			if len(tc.Bytes) != 32 || tc.Calls != 1 {
				t.Fatal("unexpected source entropy contract")
			}
			got := pkceFromBytes([32]byte(tc.Bytes))
			if got != tc.Output {
				t.Fatalf("PKCE: %#v, expected %#v", got, tc.Output)
			}
		})
	}
}
func TestGeneratePKCEUsesFreshEntropy(t *testing.T) {
	first, err := GeneratePKCE()
	if err != nil {
		t.Fatal(err)
	}
	second, err := GeneratePKCE()
	if err != nil {
		t.Fatal(err)
	}
	if first.Verifier == second.Verifier {
		t.Fatal("verifier entropy reused")
	}
	for _, pair := range []PKCE{first, second} {
		for _, encoded := range []string{pair.Verifier, pair.Challenge} {
			decoded, err := base64.RawURLEncoding.DecodeString(encoded)
			if err != nil || len(decoded) != 32 || len(encoded) != 43 {
				t.Fatalf("invalid PKCE encoding: length %d, %v", len(decoded), err)
			}
		}
	}
}
