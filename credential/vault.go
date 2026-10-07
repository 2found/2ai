// Package credential provides the small, in-memory host vault shared by AgentRay
// and consumers of agentcore.CredentialResolver. It has no infrastructure dependencies.
package credential

import (
	"context"
	"fmt"
	"regexp"
	"sync"
)

// placeholderRegex matches the {{cred:NAME}} reference syntax an agent emits in
// a tool argument. NAME is the only capture group.
var placeholderRegex = regexp.MustCompile(`\{\{\s*cred:([A-Za-z0-9_.\-]{1,128})\s*\}\}`)

// nameRegex validates a credential name on the way *in* (Put), so a malformed
// name can never be stored and later fail to resolve.
var nameRegex = regexp.MustCompile(`^[A-Za-z0-9_.\-]{1,128}$`)

// Vault holds named secrets and resolves {{cred:NAME}} placeholders against
// them. Values go in via Put and only ever come back out substituted into a
// tool's argument string by Resolve. Formatting reports only the count; JSON
// cannot serialize the private map.
//
// Safe for concurrent use: a run resolves placeholders while the host may still
// be loading the vault at startup.
type Vault struct {
	mu      sync.RWMutex
	secrets map[string]string
}

// NewVault returns an empty in-memory vault.
func NewVault() *Vault {
	return &Vault{secrets: make(map[string]string)}
}

// ValidName reports whether name is a legal credential name. It is the single
// source of truth for the {{cred:NAME}} naming rule, shared by the host-side
// secret store (storage.UpsertAgentSecret) so a name that cannot be stored can
// never become one that silently fails to resolve at run time.
func ValidName(name string) bool { return nameRegex.MatchString(name) }

// FromMap builds a Vault from a name→value map, validating every entry through
// Put. It fails closed: a single invalid name or empty value rejects the whole
// map, so a misconfigured per-agent secret store surfaces at run start rather
// than as a silently-missing injection mid-run.
func FromMap(m map[string]string) (*Vault, error) {
	v := NewVault()
	for name, value := range m {
		if err := v.Put(name, value); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// Put stores (or overwrites) a secret under name. The name must match
// [A-Za-z0-9_.-]{1,128}; an empty value is rejected so a misconfigured secret
// surfaces at load time, not as a silently-empty injection at run time.
func (v *Vault) Put(name, value string) error {
	if !nameRegex.MatchString(name) {
		return fmt.Errorf("credential: invalid name %q (must match [A-Za-z0-9_.-]{1,128})", name)
	}
	if value == "" {
		return fmt.Errorf("credential: empty value for %q", name)
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	v.secrets[name] = value
	return nil
}

// Names returns the stored credential names (never the values), sorted-stable
// only by map iteration — for startup logging and tests. It is safe to log.
func (v *Vault) Names() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	names := make([]string, 0, len(v.secrets))
	for name := range v.secrets {
		names = append(names, name)
	}
	return names
}

// Len reports how many credentials are loaded.
func (v *Vault) Len() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return len(v.secrets)
}

// Resolve substitutes every {{cred:NAME}} placeholder in args with its secret
// value. It implements agentcore.CredentialResolver and fails closed: if a
// referenced credential is unknown, it returns an error (naming the missing
// credential, never a value) so the tool call is blocked and the model is told
// to correct course, rather than a bare placeholder reaching the tool. Args
// with no placeholder are returned unchanged with no allocation of the map.
func (v *Vault) Resolve(_ context.Context, args string) (string, error) {
	matches := placeholderRegex.FindAllStringSubmatch(args, -1)
	if len(matches) == 0 {
		return args, nil
	}

	v.mu.RLock()
	defer v.mu.RUnlock()

	// Validate every reference before substituting so an unknown name fails the
	// whole call atomically (no partially-resolved args reach the tool).
	for _, m := range matches {
		if _, ok := v.secrets[m[1]]; !ok {
			return "", fmt.Errorf("credential: unknown credential %q referenced in tool arguments", m[1])
		}
	}

	resolved := placeholderRegex.ReplaceAllStringFunc(args, func(token string) string {
		name := placeholderRegex.FindStringSubmatch(token)[1]
		return v.secrets[name]
	})
	return resolved, nil
}

// String and GoString prevent accidental value disclosure through diagnostic formatting.
func (v *Vault) String() string   { return fmt.Sprintf("credential.Vault(%d names)", v.Len()) }
func (v *Vault) GoString() string { return v.String() }
