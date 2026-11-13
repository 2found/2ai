package ai

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
)

type PKCE struct {
	Verifier  string `json:"verifier"`
	Challenge string `json:"challenge"`
}

// GeneratePKCE uses 32 cryptographically random bytes and an S256 challenge,
// matching Pi's Web Crypto flow. The challenge hashes the encoded verifier.
func GeneratePKCE() (PKCE, error) {
	var bytes [32]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return PKCE{}, err
	}
	return pkceFromBytes(bytes), nil
}

func pkceFromBytes(bytes [32]byte) PKCE {
	verifier := base64.RawURLEncoding.EncodeToString(bytes[:])
	hash := sha256.Sum256([]byte(verifier))
	return PKCE{Verifier: verifier, Challenge: base64.RawURLEncoding.EncodeToString(hash[:])}
}
