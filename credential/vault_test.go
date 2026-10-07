package credential_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/credential"
)

func TestPublicVaultContractAndSafeFormatting(t *testing.T) {
	const secret = "test-value-only-not-a-real-key"
	v, err := credential.FromMap(map[string]string{"TOKEN": secret})
	if err != nil {
		t.Fatal(err)
	}
	var resolver agentcore.CredentialResolver = v
	value, err := resolver.Resolve(context.Background(), "{{cred:TOKEN}}")
	if err != nil || value != secret {
		t.Fatal("resolution failed")
	}
	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	for _, rendered := range []string{fmt.Sprint(v), fmt.Sprintf("%+v", v), fmt.Sprintf("%#v", v), string(encoded)} {
		if strings.Contains(rendered, secret) {
			t.Fatal("vault formatting disclosed a value")
		}
	}
	if value, err := resolver.Resolve(context.Background(), "{{cred:TOKEN}} {{cred:MISSING}}"); err == nil || value != "" {
		t.Fatal("resolution did not fail atomically")
	}
}
