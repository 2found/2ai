package ai

import (
	"context"
	"sync"
)

// LazyOAuthOptions advertises auth metadata without loading the flow. Load is
// read on first use. Its returned error is a rejected load and remains cached;
// a panic models a synchronous throw before Pi has assigned the load promise.
// Input/implementation callback mutations require caller synchronization.
type LazyOAuthOptions struct {
	Name           string
	IsSubscription *bool
	LoginLabel     *string
	Load           func() (*OAuthAuth, error)
}

// LazyOAuth shares one load across login, refresh and toAuth, including
// concurrent callers. Waiting for that load does not inspect the operation's
// cancellation signal; the loaded flow owns cancellation and its result.
func LazyOAuth(input *LazyOAuthOptions) *OAuthAuth {
	type loading struct {
		done chan struct{}
		auth *OAuthAuth
		err  error
	}
	var mu sync.Mutex
	var pending *loading
	loaded := func() (*OAuthAuth, error) {
		mu.Lock()
		flight := pending
		first := flight == nil
		if first {
			flight = &loading{done: make(chan struct{})}
			pending = flight
		}
		mu.Unlock()
		if first {
			returned := false
			value, err := invokeAuth(func() (any, error) {
				auth, err := input.Load()
				returned = true
				return auth, err
			})
			mu.Lock()
			flight.auth, _ = value.(*OAuthAuth)
			flight.err = err
			if !returned {
				pending = nil
			}
			close(flight.done)
			mu.Unlock()
		}
		<-flight.done
		return flight.auth, flight.err
	}
	metadata := &OAuthAuth{Name: input.Name}
	if input.IsSubscription != nil {
		value := *input.IsSubscription
		metadata.IsSubscription = &value
	}
	if input.LoginLabel != nil {
		value := *input.LoginLabel
		metadata.LoginLabel = &value
	}
	metadata.Login = func(interaction ProviderAuthInteraction, options *OAuthLoginOptions) (any, error) {
		return invokeAuth(func() (any, error) {
			auth, err := loaded()
			if err != nil {
				return nil, err
			}
			return auth.Login(interaction, options)
		})
	}
	metadata.Refresh = func(ctx context.Context, credential any) (any, error) {
		return invokeAuth(func() (any, error) {
			auth, err := loaded()
			if err != nil {
				return nil, err
			}
			return auth.Refresh(ctx, credential)
		})
	}
	metadata.ToAuth = func(credential any) (any, error) {
		return invokeAuth(func() (any, error) {
			auth, err := loaded()
			if err != nil {
				return nil, err
			}
			return auth.ToAuth(credential)
		})
	}
	return metadata
}
