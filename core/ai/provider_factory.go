package ai

import (
	"context"
	"fmt"
)

type ProviderImages struct {
	GenerateImages func(context.Context, any, any, *Object) (any, error)
}

type ProviderClassifier struct {
	Classify func(context.Context, any, any, *Object) (any, error)
}

// ProviderFactoryOptions composes a provider from concrete callbacks. API is a
// single implementation for every chat model; otherwise APIs dispatches by
// model.api. Maps, auth, baseline models and their entries retain identity.
// Callers synchronize mutations, as for ModelProvider itself.
type ProviderFactoryOptions struct {
	ID              string
	Name            *string
	BaseURL         any
	Headers         any
	Auth            *ProviderAuth
	Models          *Array
	FetchModels     func(ModelRefreshContext) (*Array, error)
	FilterModels    func(*Array, any) (*Array, error)
	FilterAllModels func(*Array, any) (*Array, error)
	API             *ProviderStreams
	APIs            map[string]*ProviderStreams
	Images          map[string]*ProviderImages
	Classifiers     map[string]*ProviderClassifier
	Now             func() int64
}

// NewModelProvider ports Pi's createProvider. Optional capabilities are fixed
// at construction, while dispatch observes replacements in the captured maps.
// Metadata does not implicitly override model or request options.
func NewModelProvider(input *ProviderFactoryOptions) (*ModelProvider, error) {
	single, byAPI, images, classifiers := input.API, input.APIs, input.Images, input.Classifiers
	streams := []*ProviderStreams{}
	if single != nil {
		streams = append(streams, single)
	} else {
		for _, implementation := range byAPI {
			if implementation != nil {
				streams = append(streams, implementation)
			}
		}
	}
	hasImages, hasClassifiers := false, false
	for _, implementation := range images {
		hasImages = hasImages || implementation != nil
	}
	for _, implementation := range classifiers {
		hasClassifiers = hasClassifiers || implementation != nil
	}
	if len(streams) == 0 && !hasImages && !hasClassifiers {
		return nil, fmt.Errorf("Provider %s: at least one of \"api\", \"images\", or \"classifiers\" is required.", input.ID)
	}
	keyFor := func(model any) (string, error) {
		api, err := modelProperty(model, "api")
		if err != nil {
			return "", err
		}
		return catalogKey(api, make(map[*Array]bool))
	}
	apiFor := func(model any) (*ProviderStreams, error) {
		if single != nil {
			return single, nil
		}
		if byAPI == nil {
			return nil, nil
		}
		key, err := keyFor(model)
		return byAPI[key], err
	}
	failure := func(model any, code, operation string) error {
		key, err := keyFor(model)
		if err != nil {
			return err
		}
		return NewModelsError(code, fmt.Sprintf("Provider %s %s for \"%s\"", input.ID, operation, key), nil)
	}
	now := input.Now
	catalog := newProviderModelCatalog(func() string { return input.ID }, input.Models, input.FetchModels, now)
	provider := &ModelProvider{ID: input.ID, Name: input.ID, BaseURL: input.BaseURL, Headers: input.Headers, Auth: input.Auth,
		GetModels: catalog.GetModels, GetAllModels: catalog.GetAllModels, RefreshModels: catalog.RefreshModels,
		FilterModels: input.FilterModels, FilterAllModels: input.FilterAllModels}
	if input.Name != nil {
		provider.Name = *input.Name
	}
	if provider.Auth == nil {
		provider.Auth = &ProviderAuth{}
	}
	dispatch := func(simple bool) ModelStreamFunc {
		return func(ctx context.Context, model any, transcript TranscriptContext, options *Object) (*ProviderEventSource, error) {
			implementation, err := apiFor(model)
			if err != nil {
				return nil, err
			}
			if implementation == nil {
				return SourceFromAssistantStream(LazyStream(ctx, model, func(context.Context) (*ProviderEventSource, error) {
					return nil, failure(model, "stream", "has no API implementation")
				}, now)), nil
			}
			if simple {
				return implementation.StreamSimple(ctx, model, transcript, options)
			}
			return implementation.Stream(ctx, model, transcript, options)
		}
	}
	provider.Stream, provider.StreamSimple = dispatch(false), dispatch(true)
	for _, implementation := range streams {
		if implementation.FetchDeferred != nil {
			provider.FetchDeferred = func(ctx context.Context, model, handle any, options *Object) (*ProviderEventSource, error) {
				return SourceFromAssistantStream(LazyStream(ctx, model, func(ctx context.Context) (*ProviderEventSource, error) {
					implementation, err := apiFor(model)
					if err != nil {
						return nil, err
					}
					if implementation == nil || implementation.FetchDeferred == nil {
						return nil, failure(model, "provider", "does not support deferred responses")
					}
					return implementation.FetchDeferred(ctx, model, handle, options)
				}, now)), nil
			}
		}
		if implementation.CancelDeferred != nil {
			provider.CancelDeferred = func(ctx context.Context, model, handle any, options *Object) error {
				implementation, err := apiFor(model)
				if err != nil {
					return err
				}
				if implementation == nil || implementation.CancelDeferred == nil {
					return failure(model, "provider", "cannot cancel deferred responses")
				}
				return implementation.CancelDeferred(ctx, model, handle, options)
			}
		}
	}
	if hasImages {
		provider.GenerateImages = func(ctx context.Context, model, request any, options *Object) (any, error) {
			key, err := keyFor(model)
			if err != nil {
				return nil, err
			}
			if implementation := images[key]; implementation != nil {
				return implementation.GenerateImages(ctx, model, request, options)
			}
			return modelOperationErrorResult(model, failure(model, "provider", "has no image generation implementation"), false, false, now)
		}
	}
	if hasClassifiers {
		provider.Classify = func(ctx context.Context, model, request any, options *Object) (any, error) {
			key, err := keyFor(model)
			if err != nil {
				return nil, err
			}
			if implementation := classifiers[key]; implementation != nil {
				return implementation.Classify(ctx, model, request, options)
			}
			return modelOperationErrorResult(model, failure(model, "provider", "has no classifier implementation"), false, true, now)
		}
	}
	return provider, nil
}
