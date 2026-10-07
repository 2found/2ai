package ai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

const (
	maxProviderErrorBody   = 64 << 10
	maxInBandErrorDetail   = 4 << 10
	truncatedErrorBodyNote = "\n[provider error body truncated]"
)

// readProviderErrorBody bounds adversarial or accidental error pages. Provider
// errors are diagnostic metadata, not model output; retaining 64 KiB is ample
// while preventing a gateway from making one failed request consume unbounded
// process memory on either a laptop or a server worker.
func readProviderErrorBody(r io.Reader) []byte {
	data, _ := io.ReadAll(io.LimitReader(r, maxProviderErrorBody+1))
	if len(data) <= maxProviderErrorBody {
		return data
	}
	data = data[:maxProviderErrorBody]
	return append(data, truncatedErrorBodyNote...)
}

// inBandProviderError recognizes provider failures carried inside a successful
// HTTP/SSE envelope. It deliberately derives a synthetic status only from an
// explicit recognized status field or a known machine-readable error code;
// prose mentioning a status cannot accidentally enter an authentication or
// retry lane.
func inBandProviderError(provider string, resp *http.Response, payload []byte) (*protocol.ProviderError, bool) {
	trimmed := strings.TrimSpace(string(payload))
	if trimmed == "" {
		return nil, false
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(payload, &root); err != nil {
		status := leadingRetryableStatus(trimmed)
		if status == 0 {
			return nil, false
		}
		pe := protocol.NewProviderError(provider, resp, limitErrorDetail(trimmed))
		pe.Status = status
		return pe, true
	}

	inner, hasError := nestedErrorObject(root)
	rootType := rawString(root["type"])
	code := normalizeErrorCode(firstNonEmpty(rawScalar(inner["code"]), rawString(inner["type"]), rawScalar(root["code"])))
	if code == "" && rootType != "error" {
		code = normalizeErrorCode(rootType)
	}
	knownStatus := providerStatusForCode(code)
	explicitStatus := firstProviderStatus(
		rawStatus(inner["status"]), rawStatus(inner["status_code"]), rawStatus(inner["code"]),
		rawStatus(root["status"]), rawStatus(root["status_code"]), rawStatus(root["code"]),
	)
	if !hasError && rootType != "error" && knownStatus == 0 && explicitStatus == 0 {
		return nil, false
	}

	message := firstNonEmpty(rawString(inner["message"]), rawString(root["message"]))
	if message == "" && hasError && len(inner) == 0 {
		message = rawString(root["error"])
	}
	if code != "" && !strings.Contains(strings.ToLower(message), strings.ToLower(code)) {
		if message == "" {
			message = code
		} else {
			message = code + ": " + message
		}
	}
	if message == "" {
		message = "provider returned an in-band error"
	}

	status := explicitStatus
	if status == 0 {
		status = knownStatus
	}
	pe := protocol.NewProviderError(provider, resp, limitErrorDetail(message))
	if status != 0 {
		pe.Status = status
	} else if strictToolsRejectionMessage(message) {
		// Some OpenAI-compatible SSE servers omit status and code entirely.
		// The narrow message classifier makes this a request-shape rejection,
		// allowing strict-tool preflight to demote it exactly once.
		pe.Status = http.StatusUnprocessableEntity
	} else if pe.Status == 0 {
		// An in-band error is not a transport failure. A neutral 200 status lets
		// IsRetryable consult the message without treating every unknown envelope
		// as transient merely because no response object was supplied in a test.
		pe.Status = http.StatusOK
	}
	return pe, true
}

func nestedErrorObject(root map[string]json.RawMessage) (map[string]json.RawMessage, bool) {
	raw, ok := root["error"]
	if !ok {
		if responseRaw, exists := root["response"]; exists {
			var response map[string]json.RawMessage
			if json.Unmarshal(responseRaw, &response) == nil {
				raw, ok = response["error"]
			}
		}
	}
	if !ok {
		return map[string]json.RawMessage{}, false
	}
	var current map[string]json.RawMessage
	if json.Unmarshal(raw, &current) != nil {
		return map[string]json.RawMessage{}, true
	}
	if current == nil { // JSON null is the absence of an error, not an envelope.
		return map[string]json.RawMessage{}, false
	}
	// Some Azure-compatible gateways double-wrap the error object. Bound the
	// walk so hostile JSON cannot turn classification into recursive work.
	for depth := 0; depth < 2; depth++ {
		nextRaw, exists := current["error"]
		if !exists {
			break
		}
		var next map[string]json.RawMessage
		if json.Unmarshal(nextRaw, &next) != nil {
			break
		}
		current = next
	}
	return current, true
}

func rawString(raw json.RawMessage) string {
	var value string
	if json.Unmarshal(raw, &value) == nil {
		return strings.TrimSpace(value)
	}
	return ""
}

func rawScalar(raw json.RawMessage) string {
	if value := rawString(raw); value != "" {
		return value
	}
	var number json.Number
	if json.Unmarshal(raw, &number) == nil {
		return number.String()
	}
	return ""
}

func rawStatus(raw json.RawMessage) int {
	value := rawScalar(raw)
	status, err := strconv.Atoi(value)
	if err != nil || status < 100 || status > 599 {
		return 0
	}
	return status
}

func firstProviderStatus(values ...int) int {
	for _, status := range values {
		if status == http.StatusBadRequest || status == http.StatusUnprocessableEntity ||
			status == http.StatusUnauthorized || status == http.StatusForbidden ||
			status == http.StatusTooManyRequests || status >= 500 && status <= 599 {
			return status
		}
	}
	return 0
}

func providerStatusForCode(code string) int {
	switch normalizeErrorCode(code) {
	case "bad_request", "invalid_request", "invalid_request_error":
		return http.StatusBadRequest
	case "unprocessable_entity":
		return http.StatusUnprocessableEntity
	case "authentication_error", "invalid_api_key", "invalid_token", "token_expired",
		"unauthenticated", "unauthorized":
		return http.StatusUnauthorized
	case "access_denied", "forbidden", "permission_denied", "permission_error":
		return http.StatusForbidden
	case "rate_limit_error", "rate_limit_exceeded", "rate_limit", "rate_limited",
		"rate_limit_reached", "ratelimit", "too_many_requests", "request_throttled",
		"throttled", "throttling", "throttling_error", "throttling_exception",
		"throttling_allocation_quota", "request_limit_exceeded", "retry_later":
		return http.StatusTooManyRequests
	case "overloaded_error", "server_overloaded", "model_overloaded", "overloaded",
		"service_unavailable", "server_busy", "high_demand", "capacity_exceeded":
		return http.StatusServiceUnavailable
	default:
		return 0
	}
}

func normalizeErrorCode(code string) string {
	code = strings.TrimSpace(code)
	if code == "" {
		return ""
	}
	var normalized strings.Builder
	lastUnderscore := false
	previousWasLowerOrDigit := false
	for _, r := range code {
		original := r
		if r >= 'A' && r <= 'Z' {
			if previousWasLowerOrDigit && !lastUnderscore {
				normalized.WriteByte('_')
			}
			r += 'a' - 'A'
		}
		switch r {
		case '-', '.', ' ', '\t':
			r = '_'
		}
		if r == '_' {
			if normalized.Len() == 0 || lastUnderscore {
				continue
			}
			lastUnderscore = true
			previousWasLowerOrDigit = false
		} else {
			lastUnderscore = false
			previousWasLowerOrDigit = original >= 'a' && original <= 'z' || original >= '0' && original <= '9'
		}
		normalized.WriteRune(r)
	}
	return strings.Trim(normalized.String(), "_")
}

func leadingRetryableStatus(message string) int {
	fields := strings.Fields(message)
	if len(fields) == 0 {
		return 0
	}
	first := fields[0]
	if strings.HasPrefix(strings.ToUpper(first), "HTTP/") && len(fields) > 1 {
		first = fields[1]
	}
	status, err := strconv.Atoi(strings.TrimRight(first, ":"))
	if err != nil {
		return 0
	}
	status = firstProviderStatus(status)
	// A free-form non-JSON line is commonly a proxy failure page. Only promote
	// transient statuses here; auth rotation requires a structured status/code
	// so an arbitrary text frame cannot burn a pooled credential.
	if status == http.StatusUnauthorized || status == http.StatusForbidden {
		return 0
	}
	return status
}

// providerErrorWithStatus retains retry/reset headers from the HTTP response
// while applying the status carried by an HTTP-200 stream event.
func providerErrorWithStatus(provider string, resp *http.Response, status int, message string) *protocol.ProviderError {
	err := protocol.NewProviderError(provider, resp, message)
	err.Status = status
	return err
}

func limitErrorDetail(message string) string {
	message = strings.TrimSpace(message)
	if len(message) <= maxInBandErrorDetail {
		return message
	}
	return fmt.Sprintf("%s…", message[:maxInBandErrorDetail])
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
