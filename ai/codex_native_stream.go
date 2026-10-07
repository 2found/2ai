package ai

import (
	"encoding/json"
	"fmt"
)

// codexAPIError retains the code used by Codex's transport retry decisions.
// Payload is the original event, including fields unrecognized by the parser.
type codexAPIError struct {
	Code    string
	Payload json.RawMessage
	Message string
}

func (e *codexAPIError) Error() string { return e.Message }

type codexResponsesAccumulator struct{ *responsesAccumulator }

func newCodexResponsesAccumulator(model completionsModel, grammar map[string]string, stream *AssistantMessageEventStream, timestamp int64) *codexResponsesAccumulator {
	model.API = "openai-codex-responses"
	acc := newResponsesAccumulator(model, grammar, stream, timestamp)
	// Codex uses its own API-error wording and does not append the public
	// Responses provider's subscription-sharing help text.
	acc.formatFailure = func(message string) string { return message }
	acc.resolveServiceTier = func(response, request json.RawMessage) json.RawMessage {
		tier := samplingString(request)
		if samplingString(response) == "default" && (tier == "flex" || tier == "priority") {
			return request
		}
		if samplingNonNull(response) {
			return response
		}
		return request
	}
	acc.applyServiceTierPricing = func(usage *Usage, tier json.RawMessage) {
		// Unlike public Responses, Codex does not alias "fast" to "priority".
		if samplingString(tier) != "fast" {
			responsesServiceTierPricing(model.ID, usage, tier)
		}
	}
	return &codexResponsesAccumulator{acc}
}

// chunk maps the Codex protocol into shared Responses events. A terminal event
// ends admission immediately, even if the transport contains later frames.
func (a *codexResponsesAccumulator) chunk(raw json.RawMessage) (bool, error) {
	return a.chunkStarted(raw, nil)
}
func (a *codexResponsesAccumulator) chunkStarted(raw json.RawMessage, before func()) (bool, error) {
	event, valid := samplingObject(raw)
	if !valid {
		return false, fmt.Errorf("Codex event must be an object")
	}
	var kind string
	if json.Unmarshal(event["type"], &kind) != nil || kind == "" {
		return false, nil
	}
	switch kind {
	case "error":
		nested, _ := samplingObject(event["error"])
		stringField := func(name string) string {
			var value string
			if json.Unmarshal(event[name], &value) == nil && string(event[name]) != "null" {
				return value
			}
			_ = json.Unmarshal(nested[name], &value)
			return value
		}
		code, message := stringField("code"), stringField("message")
		if message == "" {
			message = code
		}
		if message == "" {
			encoded, err := stringifyCompletionsJSON(raw)
			if err != nil {
				return false, err
			}
			message = string(encoded)
		}
		return false, &codexAPIError{Code: code, Payload: append(json.RawMessage(nil), raw...), Message: "Codex error: " + message}
	case "response.failed":
		response, _ := samplingObject(event["response"])
		failure, _ := samplingObject(response["error"])
		message := samplingString(failure["message"])
		if message == "" {
			message = "Codex response failed"
		}
		return false, &codexAPIError{Code: samplingString(failure["code"]), Payload: append(json.RawMessage(nil), raw...), Message: message}
	case "response.done", "response.completed", "response.incomplete":
		if samplingNonNull(event["response"]) {
			response, valid := samplingObject(event["response"])
			if !valid {
				return false, fmt.Errorf("Codex response must be an object")
			}
			switch string(response["end_turn"]) {
			case "true", "false":
				end := string(response["end_turn"]) == "true"
				a.stream.Synchronize(func() { a.output.EndTurn = &end })
			}
			switch samplingString(response["status"]) {
			case "completed", "incomplete", "failed", "cancelled", "queued", "in_progress":
			default:
				delete(response, "status")
			}
			encoded, err := json.Marshal(response)
			event["response"] = encoded
			if err != nil {
				return false, err
			}
		}
		event["type"] = json.RawMessage(`"response.completed"`)
		mapped, err := json.Marshal(event)
		if err != nil {
			return false, err
		}
		if before != nil {
			before()
		}
		return true, a.responsesAccumulator.chunk(mapped)
	default:
		if before != nil {
			before()
		}
		return false, a.responsesAccumulator.chunk(raw)
	}
}
