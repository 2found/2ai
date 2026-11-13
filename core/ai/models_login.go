package ai

import (
	"context"
	"sync"
)

// Login preserves the provider's credential object and waits for its write to
// settle. Cancellation stops waiting for a queued mutation. Once the store
// invokes the mutation callback, it owns the commit/abort boundary.
func (m *Models) Login(providerID, authType string, interaction ProviderAuthInteraction, options ...*OAuthLoginOptions) (any, error) {
	ctx := interaction.Context
	if ctx == nil {
		ctx = context.Background()
		interaction.Context = ctx
	}
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	provider := m.GetProvider(providerID)
	if provider == nil {
		return nil, NewModelsError("provider", "Unknown provider: "+providerID, nil)
	}
	var login func() (any, error)
	var settings *OAuthLoginOptions
	if len(options) > 0 {
		settings = options[0]
	}
	if authType == "oauth" {
		if method := provider.authConfig().OAuth; method != nil && method.Login != nil {
			login = func() (any, error) { return method.Login(interaction, settings) }
		}
	} else if method := provider.authConfig().APIKey; method != nil && method.Login != nil {
		login = func() (any, error) { return method.Login(interaction, settings) }
	}
	if login == nil {
		return nil, NewModelsError("auth", provider.Name+" does not support "+authType+" login", nil)
	}
	credential, err := raceAuthOperation(ctx, login)
	if err != nil {
		return nil, err
	}
	started := make(chan struct{})
	settled := make(chan error, 1)
	var once sync.Once
	go func() {
		_, err := invokeAuth(func() (any, error) {
			return m.credentials.Modify(ctx, providerID, func(any) (any, error) {
				once.Do(func() { close(started) })
				return credential, nil
			})
		})
		settled <- err
	}()
	select {
	case <-started:
		err = <-settled
	case err = <-settled:
	case <-ctx.Done():
		select {
		case <-started:
			err = <-settled
		default:
			return nil, context.Cause(ctx)
		}
	}
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return nil, cause
		}
		return nil, NewModelsError("auth", "Credential store modify failed for "+providerID, err)
	}
	return credential, nil
}

// Logout also accepts an unknown provider ID: stored credentials outlive
// provider registration. An admitted delete follows the persistence callback's
// settlement, including stores that finish a deletion after caller cancellation.
func (m *Models) Logout(ctx context.Context, providerID string) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	_, err := invokeAuth(func() (any, error) { return Undefined, m.credentials.Delete(ctx, providerID) })
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return cause
		}
		return NewModelsError("auth", "Credential store delete failed for "+providerID, err)
	}
	return nil
}
