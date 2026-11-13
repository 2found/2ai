package ai

import (
	"context"
	"os"
	"strings"
)

// DefaultProviderAuthContext reads process environment lazily. Whitespace-only
// values are absent, while nonempty values retain their original whitespace.
// FileExists follows Pi's literal leading-tilde expansion, including ~suffix.
func DefaultProviderAuthContext() *AuthContext {
	return &AuthContext{
		Env: func(_ context.Context, name string) (any, error) {
			value, exists := os.LookupEnv(name)
			if !exists || strings.TrimFunc(value, jsWhitespace) == "" {
				return Undefined, nil
			}
			return value, nil
		},
		FileExists: func(_ context.Context, path string) (bool, error) {
			if strings.HasPrefix(path, "~") {
				home, err := os.UserHomeDir()
				if err != nil {
					return false, nil
				}
				path = home + path[1:]
			}
			_, err := os.Stat(path)
			return err == nil, nil
		},
	}
}
