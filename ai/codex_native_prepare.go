package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"

	"github.com/google/uuid"
)

type codexPreparedRequest struct {
	body                         json.RawMessage
	controls                     map[string]json.RawMessage
	account, session             string
	sseHeaders, websocketHeaders http.Header
	timeout, connectTimeout      float64
}

func prepareCodexRequest(ctx context.Context, acc *codexResponsesAccumulator, rawModel json.RawMessage, transcript TranscriptContext, options CodexResponsesStreamOptions) (*codexPreparedRequest, error) {
	controls := map[string]json.RawMessage{}
	if len(options.Options) > 0 {
		if err := json.Unmarshal(options.Options, &controls); err != nil {
			return nil, err
		}
	}
	key := samplingString(controls["apiKey"])
	if key == "" {
		return nil, fmt.Errorf("No API key for provider: %s", acc.model.Provider)
	}
	account, err := codexNativeAccountID(key)
	if err != nil {
		return nil, err
	}
	acc.grammar, err = CreateGrammarToolInputProperties(GetDeclaredTools(transcript.Messages()), samplingTruthy(acc.model.Compat["supportsOpenAIGrammarTools"]))
	if err != nil {
		return nil, err
	}
	acc.serviceTier = controls["serviceTier"]
	var session *string
	rawSession := ""
	if samplingString(controls["cacheRetention"]) != "none" {
		if raw, exists := controls["sessionId"]; exists {
			var value string
			if json.Unmarshal(raw, &value) != nil {
				return nil, errors.New("Codex sessionId must be a string")
			}
			rawSession = value
			chars := []rune(value)
			if len(chars) > 64 {
				value = string(chars[:64])
			}
			session = &value
		}
	}
	params, err := BuildCodexResponsesParams(rawModel, transcript, options.Options, session)
	if err != nil {
		return nil, err
	}
	if options.OnPayload != nil {
		next, err := options.OnPayload(ctx, params, rawModel)
		if err != nil {
			return nil, &codexProviderCallbackError{err}
		}
		if next != nil {
			params = next
		}
	}
	if !json.Valid(params) {
		return nil, errors.New("onPayload returned invalid JSON")
	}
	fields, _ := samplingObject(rawModel)
	headers, err := codexNativeHeaders(fields["headers"], controls["headers"], account, key, session, false)
	if err != nil {
		return nil, err
	}
	requestID := ""
	if session != nil {
		requestID = *session
	}
	if requestID == "" {
		id, err := uuid.NewV7()
		if err != nil {
			return nil, err
		}
		requestID = id.String()
	}
	wsHeaders, err := codexNativeHeaders(fields["headers"], controls["headers"], account, key, &requestID, true)
	if err != nil {
		return nil, err
	}
	timeout := 0.0
	if raw, exists := controls["timeoutMs"]; exists {
		if json.Unmarshal(raw, &timeout) != nil || !samplingNonNull(raw) || timeout < 0 || math.IsInf(timeout, 0) || math.IsNaN(timeout) {
			return nil, fmt.Errorf("Invalid timeoutMs: %s", completionsErrorString(raw))
		}
		timeout = math.Floor(timeout)
	}
	connectTimeout := 0.0
	if raw, exists := controls["websocketConnectTimeoutMs"]; exists {
		if json.Unmarshal(raw, &connectTimeout) != nil || !samplingNonNull(raw) || connectTimeout < 0 || math.IsInf(connectTimeout, 0) || math.IsNaN(connectTimeout) {
			return nil, fmt.Errorf("Invalid timeoutMs: %s", completionsErrorString(raw))
		}
		connectTimeout = math.Floor(connectTimeout)
	}
	params, err = stringifyCompletionsJSON(params)
	if err != nil {
		return nil, err
	}
	return &codexPreparedRequest{body: params, controls: controls, account: account, session: rawSession, sseHeaders: headers, websocketHeaders: wsHeaders, timeout: timeout, connectTimeout: connectTimeout}, nil
}
