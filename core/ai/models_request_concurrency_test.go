package ai

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestPiModelsRequestCancellationAfterAuth(t *testing.T) {
	for _, tc := range readModelsRequestFixture(t).Races {
		t.Run(fmt.Sprintf("%s/%s/fail=%v", tc.Operation, tc.Stage, tc.Fail), func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			started, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(release) })
			calls := 0
			var providerAborted any = Undefined
			models := NewModels()
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
				return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "key"})}), nil
			}}}}
			dispatch := func(context.Context, any, any, *Object) (any, error) {
				calls++
				if tc.Stage == "provider" {
					close(started)
					<-release
				}
				providerAborted = ctx.Err() != nil
				if tc.Stage == "provider" && tc.Fail {
					return nil, errors.New("provider failed")
				}
				return NewObject(Property{Name: "result", Value: "ok"}), nil
			}
			provider.CancelDeferred = func(c context.Context, m, h any, o *Object) error { _, err := dispatch(c, m, h, o); return err }
			provider.GenerateImages = dispatch
			provider.Classify = dispatch
			models.SetProvider(provider)
			kind := "chat"
			if tc.Operation == "images" {
				kind = "image"
			}
			if tc.Operation == "classify" {
				kind = "classifier"
			}
			model := NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}, Property{Name: "api", Value: "test"}, Property{Name: "type", Value: kind})
			options := &ModelsRequestOptions{Now: func() int64 { return 1000 }, TransformHeaders: func(headers any) (any, error) {
				if tc.Stage == "transform" {
					close(started)
					<-release
					if tc.Fail {
						return nil, errors.New("transform failed")
					}
				}
				return headers, nil
			}}
			pending := make(chan any, 1)
			go func() {
				var value any
				var err error
				switch tc.Operation {
				case "cancel":
					value = Undefined
					err = models.CancelDeferred(ctx, model, NewObject(), options)
				case "images":
					value, err = models.GenerateImages(ctx, model, NewObject(), options)
				case "classify":
					value, err = models.Classify(ctx, model, NewObject(), options)
				}
				pending <- authResolveCapture(value, err)
			}()
			publicationAwait(t, started)
			cancel(errors.New("cancelled"))
			if tc.SettledBeforeRelease {
				t.Fatal("unexpected source cancellation contract")
			}
			select {
			case value := <-pending:
				t.Fatalf("returned before callback settled: %v", value)
			case <-time.After(25 * time.Millisecond):
			}
			once.Do(func() { close(release) })
			catalogCompare(t, publicationAwait(t, pending), tc.Result)
			catalogCompare(t, authResolveCapture(providerAborted, nil), tc.ProviderAborted)
			if calls != tc.DispatchCalls {
				t.Fatalf("provider calls %d, want %d", calls, tc.DispatchCalls)
			}
		})
	}
}

func TestModelsRequestCancelDeferredRetainsCause(t *testing.T) {
	for _, stage := range []string{"resolve", "transform", "provider"} {
		t.Run(stage, func(t *testing.T) {
			cause := errors.New("original failure")
			models := NewModels()
			provider := &ModelProvider{ID: "p", Auth: &ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(APIKeyAuthInput) (any, error) {
				if stage == "resolve" {
					return nil, cause
				}
				return NewObject(Property{Name: "auth", Value: NewObject(Property{Name: "apiKey", Value: "key"})}), nil
			}}}, CancelDeferred: func(context.Context, any, any, *Object) error { return cause }}
			models.SetProvider(provider)
			settings := &ModelsRequestOptions{TransformHeaders: func(value any) (any, error) {
				if stage == "transform" {
					return nil, cause
				}
				return value, nil
			}}
			err := models.CancelDeferred(context.Background(), NewObject(Property{Name: "id", Value: "m"}, Property{Name: "provider", Value: "p"}), NewObject(), settings)
			if !errors.Is(err, cause) {
				t.Fatalf("lost cause: %v", err)
			}
			if stage == "resolve" {
				var typed *ModelsError
				if !errors.As(err, &typed) || typed.Code != "auth" {
					t.Fatal("missing auth error category")
				}
			} else if err != cause {
				t.Fatal("transform/provider error was wrapped")
			}
		})
	}
}
