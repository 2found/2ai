package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// OpenAIResponsesStreamOptions uses the same concrete HTTP client and callback
// contracts as Completions; Options carries Responses-specific request controls.
type OpenAIResponsesStreamOptions = OpenAICompletionsStreamOptions

// StreamOpenAIResponses runs Responses directly in Go. It freezes caller inputs
// and returns the live stream immediately; producer failures emit error events.
func StreamOpenAIResponses(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAIResponsesStreamOptions) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStreamFor(ctx)
	rawModel = append(json.RawMessage(nil), rawModel...)
	options.Options = append(json.RawMessage(nil), options.Options...)
	var model completionsModel
	prepareErr := json.Unmarshal(rawModel, &model)
	encoded, err := json.Marshal(transcript)
	if prepareErr == nil {
		prepareErr = err
	}
	var frozen Context
	if err = json.Unmarshal(encoded, &frozen); prepareErr == nil {
		prepareErr = err
	}
	compat, err := ResolveOpenAIResponsesCompat(rawModel)
	if prepareErr == nil {
		prepareErr = err
	}
	transcript = ResolveTranscript(NormalizeContext(frozen), compat.SupportsMidConvoSystemMessages)
	now := time.Now().UnixMilli()
	if options.Now != nil {
		now = options.Now()
	}
	options, callbackFailure := nativeFailureCallbacks(options)
	recordNativeFailure(ctx, model.Provider, nil, false)
	acc := newResponsesAccumulator(model, nil, stream, now)
	acc.applyServiceTierPricing = func(usage *Usage, tier json.RawMessage) { responsesServiceTierPricing(model.ID, usage, tier) }
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				recordNativeFailure(ctx, model.Provider, fmt.Errorf("%v", recovered), true)
				acc.fail(fmt.Sprint(recovered), ctx.Err() != nil)
			}
		}()
		err := prepareErr
		if err == nil {
			err = runResponsesHTTP(ctx, acc, compat, rawModel, transcript, options)
		}
		if err != nil {
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			acc.fail(formatResponsesError(err, model.Provider), ctx.Err() != nil)
		}
	}()
	return stream
}
func runResponsesHTTP(ctx context.Context, acc *responsesAccumulator, compat OpenAIResponsesCompat, rawModel json.RawMessage, transcript TranscriptContext, options OpenAIResponsesStreamOptions) error {
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return err
		}
	}
	key, err := completionsAPIKey(acc.model.Provider, controls)
	if err != nil {
		return err
	}
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(transcript.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return err
	}
	acc.grammar = grammar
	acc.serviceTier = controls["serviceTier"]
	headers, headerErr := openAIHeaders(rawModel, transcript, controls, compat.SessionAffinityFormat, true, false, key)
	params, err := BuildOpenAIResponsesParams(rawModel, transcript, options.Options)
	if err != nil {
		return err
	}
	return runOpenAIHTTP(ctx, rawModel, options, controls, headers, headerErr, params, openAIHTTPStream{
		path: "/responses", nonstreamObjectRequired: true, start: acc.start, chunk: acc.chunk, finish: acc.finish,
		nonIterableError: "undefined is not a function (near '...event of openaiStream...')",
	})
}
