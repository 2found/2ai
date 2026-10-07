package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const VendorAzureResponses = "azure-openai-responses"

type AzureResponsesStreamOptions = OpenAICompletionsStreamOptions

// StreamAzureResponses executes Pi's Azure Responses stream directly in Go.
// Inputs are frozen before the asynchronous producer starts; callbacks receive
// the original model, while deployment and endpoint settings affect only HTTP.
func StreamAzureResponses(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options AzureResponsesStreamOptions) *AssistantMessageEventStream {
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
	compat, err := azureResponsesCompat(model)
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
	model.API = "azure-openai-responses"
	acc := newResponsesAccumulator(model, nil, stream, now)
	acc.formatFailure = func(message string) string { return message }
	acc.missingStopReason = "Azure OpenAI Responses stream ended without a stop reason"
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				recordNativeFailure(ctx, model.Provider, fmt.Errorf("%v", recovered), true)
				acc.fail(fmt.Sprint(recovered), ctx.Err() != nil)
			}
		}()
		err := prepareErr
		if err == nil {
			err = runAzureResponsesHTTP(ctx, acc, compat, rawModel, transcript, options)
		}
		if err != nil {
			recordNativeFailure(ctx, model.Provider, err, *callbackFailure)
			acc.fail(formatOpenAIProviderError(err, "Azure OpenAI API error", false), ctx.Err() != nil)
		}
	}()
	return stream
}

func runAzureResponsesHTTP(ctx context.Context, acc *responsesAccumulator, compat OpenAIResponsesCompat, rawModel json.RawMessage, transcript TranscriptContext, options AzureResponsesStreamOptions) error {
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return err
		}
	}
	key := samplingString(controls["apiKey"])
	if key == "" {
		return fmt.Errorf("No API key for provider: %s", acc.model.Provider)
	}
	config, err := ResolveAzureResponsesConfig(rawModel, options.Options)
	if err != nil {
		return err
	}
	headers, headerErr := openAIHeadersFor(rawModel, transcript, controls, "", false, false, key, true)
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(transcript.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return err
	}
	acc.grammar = grammar
	params, err := BuildAzureResponsesParams(rawModel, transcript, options.Options)
	if err != nil {
		return err
	}
	endpoint, err := azureResponsesEndpoint(config)
	if err != nil {
		return err
	}
	_, manualRedirect := headers["Api-Key"]
	return runOpenAIHTTP(ctx, rawModel, options, controls, headers, headerErr, params, openAIHTTPStream{
		endpoint: endpoint, manualRedirect: manualRedirect, nonstreamObjectRequired: true, start: acc.start, chunk: acc.chunk, finish: acc.finish,
		nonIterableError: "undefined is not a function (near '...event of openaiStream...')",
	})
}

// Azure's SDK concatenates the endpoint path before merging query parameters.
// Existing duplicates collapse to their last value without changing key order.
func azureResponsesEndpoint(config AzureResponsesConfig) (string, error) {
	parsed, err := whatwg.Parse(strings.TrimSuffix(config.BaseURL, "/") + "/responses")
	if err != nil {
		return "", err
	}
	keys := []string{}
	values := map[string]string{}
	set := func(key, value string) {
		if _, exists := values[key]; !exists {
			keys = append(keys, key)
		}
		values[key] = value
	}
	for _, pair := range strings.Split(strings.TrimPrefix(parsed.Search(), "?"), "&") {
		if pair == "" {
			continue
		}
		key, value, _ := strings.Cut(pair, "=")
		set(urlQueryComponent(key), urlQueryComponent(value))
	}
	set("api-version", config.APIVersion)
	sort.SliceStable(keys, func(i, j int) bool {
		a, ai := jsonjs.ArrayIndex(keys[i])
		b, bi := jsonjs.ArrayIndex(keys[j])
		if ai && bi {
			return a < b
		}
		return ai && !bi
	})
	escape := func(value string) string { return strings.ReplaceAll(url.QueryEscape(value), "+", "%20") }
	pairs := make([]string, len(keys))
	for i, key := range keys {
		pairs[i] = escape(key) + "=" + escape(values[key])
	}
	parsed.SetSearch(strings.Join(pairs, "&"))
	return parsed.String(), nil
}
