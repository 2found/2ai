package ai

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOAuthRefreshCancellationPreservesStartedTransaction(t *testing.T) {
	for _, caller := range []string{"auth", "catalog"} {
		for _, phase := range []string{"refresh", "commit"} {
			t.Run(caller+"/"+phase, func(t *testing.T) {
				memory := NewInMemoryCredentialStore()
				_, err := memory.Modify(context.Background(), "p", func(any) (any, error) {
					return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: "old"}, Property{Name: "expires", Value: 0}), nil
				})
				if err != nil {
					t.Fatal(err)
				}
				entered, release, settled := make(chan struct{}), make(chan struct{}), make(chan struct{})
				var once sync.Once
				defer once.Do(func() { close(release) })
				var refreshCalls, networkCalls atomic.Int32
				var retained context.Context
				store := &CredentialPersistence{Read: memory.Read, Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
					defer close(settled)
					return memory.Modify(ctx, id, func(current any) (any, error) {
						next, err := modify(current)
						if phase == "commit" {
							close(entered)
							<-release
						}
						return next, err
					})
				}}
				models := NewModels(ModelsOptions{Credentials: store})
				models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{OAuth: &OAuthAuth{
					Refresh: func(ctx context.Context, current any) (any, error) {
						refreshCalls.Add(1)
						retained = ctx
						if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > OAuthRefreshTimeout {
							t.Error("refresh lost independent timeout")
						}
						if phase == "refresh" {
							close(entered)
							<-release
							if ctx.Err() != nil {
								return nil, ctx.Err()
							}
						}
						return NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: "rotated"}, Property{Name: "expires", Value: 900000}), nil
					},
					ToAuth: func(any) (any, error) { return NewObject(), nil },
				}}, RefreshModels: func(call ModelRefreshContext) error {
					if call.AllowNetwork {
						networkCalls.Add(1)
					}
					return nil
				}})
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				done := make(chan error, 1)
				go func() {
					if caller == "auth" {
						_, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
						done <- err
					} else {
						result := models.Refresh(ctx, ModelsRefreshOptions{Now: func() int64 { return 1000 }})
						if result.Aborted {
							done <- ctx.Err()
						} else {
							done <- errors.New("catalog refresh did not report cancellation")
						}
					}
				}()
				publicationAwait(t, entered)
				cancel()
				if err := publicationAwait(t, done); !errors.Is(err, context.Canceled) {
					t.Fatal("caller did not return promptly on cancellation", err)
				}
				once.Do(func() { close(release) })
				publicationAwait(t, settled)
				stored, err := memory.Read(context.Background(), "p")
				if err != nil || catalogProperty(stored, "refresh") != "rotated" || refreshCalls.Load() != 1 || retained.Err() != nil {
					t.Fatal("cancelled waiter discarded or repeated started refresh", stored, err, refreshCalls.Load(), retained.Err())
				}
				// Credential persistence settles before the catalogue phase does.
				// The phase itself refuses network work with a cancelled context.
				if networkCalls.Load() != 0 {
					t.Fatal("cancelled catalogue refresh admitted network work")
				}
			})
		}
	}
}

func TestOAuthCancellationBeforeCredentialLockSkipsRefresh(t *testing.T) {
	memory := NewInMemoryCredentialStore()
	credential := NewObject(Property{Name: "type", Value: "oauth"}, Property{Name: "refresh", Value: "old"}, Property{Name: "expires", Value: 0})
	if _, err := memory.Modify(context.Background(), "p", func(any) (any, error) { return credential, nil }); err != nil {
		t.Fatal(err)
	}
	locked, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	go func() {
		_, _ = memory.Modify(context.Background(), "p", func(any) (any, error) {
			close(locked)
			<-release
			return Undefined, nil
		})
	}()
	publicationAwait(t, locked)
	queued, settled := make(chan struct{}), make(chan struct{})
	store := &CredentialPersistence{Read: memory.Read, Modify: func(ctx context.Context, id string, modify func(any) (any, error)) (any, error) {
		defer close(settled)
		close(queued)
		return memory.Modify(ctx, id, modify)
	}}
	var calls atomic.Int32
	models := NewModels(ModelsOptions{Credentials: store})
	models.SetProvider(&ModelProvider{ID: "p", Auth: &ProviderAuth{OAuth: &OAuthAuth{Refresh: func(context.Context, any) (any, error) {
		calls.Add(1)
		return nil, errors.New("cancelled queued refresh ran")
	}}}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := models.GetAuth(ctx, "p", AuthResolutionOverrides{Now: func() int64 { return 1000 }})
		done <- err
	}()
	publicationAwait(t, queued)
	cancel()
	if err := publicationAwait(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal("queued caller did not cancel", err)
	}
	publicationAwait(t, settled)
	once.Do(func() { close(release) })
	// A succeeding transaction is a barrier after the cancelled queue entry.
	stored, err := memory.Modify(context.Background(), "p", func(any) (any, error) { return Undefined, nil })
	if err != nil || stored != credential || calls.Load() != 0 {
		t.Fatal("queued cancellation changed credentials", stored, err, calls.Load())
	}
}
