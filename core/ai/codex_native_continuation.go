package ai

import (
	"encoding/json"
	"errors"
)

type codexWebSocketContinuation struct {
	LastRequestBody   json.RawMessage   `json:"lastRequestBody"`
	LastResponseID    string            `json:"lastResponseId"`
	LastResponseItems []json.RawMessage `json:"lastResponseItems"`
}

func codexBodyWithoutInput(raw json.RawMessage) (json.RawMessage, error) {
	fields, ok := samplingObject(raw)
	if !ok {
		return nil, errors.New("Codex request must be an object")
	}
	keys := []string{}
	for _, key := range samplingObjectKeys(raw) {
		if key != "input" && key != "previous_response_id" {
			keys = append(keys, key)
		}
	}
	delete(fields, "input")
	delete(fields, "previous_response_id")
	return stringifyCompletionsJSON(marshalSamplingObject(fields, keys))
}

// A continuation is usable only when both the request settings and complete
// previous input/output prefix retain JSON.stringify identity, including key
// order. A mismatch invalidates it and sends the full current request.
func codexCachedWebSocketBody(body json.RawMessage, continuation **codexWebSocketContinuation) (json.RawMessage, error) {
	if *continuation == nil {
		return body, nil
	}
	prior := *continuation
	currentSettings, err := codexBodyWithoutInput(body)
	if err != nil {
		return nil, err
	}
	oldSettings, err := codexBodyWithoutInput(prior.LastRequestBody)
	if err != nil {
		return nil, err
	}
	invalidate := func() (json.RawMessage, error) { *continuation = nil; return body, nil }
	if string(currentSettings) != string(oldSettings) || prior.LastResponseID == "" {
		return invalidate()
	}
	current, _ := samplingObject(body)
	previous, _ := samplingObject(prior.LastRequestBody)
	var input, baseline []json.RawMessage
	if samplingNonNull(current["input"]) {
		if err := json.Unmarshal(current["input"], &input); err != nil {
			return nil, err
		}
	}
	if samplingNonNull(previous["input"]) {
		if err := json.Unmarshal(previous["input"], &baseline); err != nil {
			return nil, err
		}
	}
	baseline = append(baseline, prior.LastResponseItems...)
	if len(input) < len(baseline) {
		return invalidate()
	}
	prefix, _ := json.Marshal(input[:len(baseline)])
	expected, _ := json.Marshal(baseline)
	// Empty null slices correspond to the source's empty array baselines.
	if len(baseline) == 0 {
		prefix, expected = json.RawMessage(`[]`), json.RawMessage(`[]`)
	}
	if !declarationJSONEqual(prefix, expected) {
		return invalidate()
	}
	delta := append([]json.RawMessage{}, input[len(baseline):]...)
	keys := samplingObjectKeys(body)
	for _, key := range []string{"previous_response_id", "input"} {
		if _, exists := current[key]; !exists {
			keys = append(keys, key)
		}
	}
	current["previous_response_id"], _ = json.Marshal(prior.LastResponseID)
	current["input"], _ = json.Marshal(delta)
	return marshalSamplingObject(current, keys), nil
}
