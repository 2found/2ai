package ai

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

func codexNativeURL(base string) string {
	if strings.TrimFunc(base, jsWhitespace) == "" {
		base = codexBaseURL
	}
	base = strings.TrimRight(base, "/")
	if strings.HasSuffix(base, "/codex/responses") {
		return base
	}
	if strings.HasSuffix(base, "/codex") {
		return base + "/responses"
	}
	return base + "/codex/responses"
}

// The source decodes JWT payloads with atob (Latin-1), not a UTF-8 decoder.
// Validation here extracts the account identity; it does not verify the JWT.
func codexNativeAccountID(token string) (string, error) {
	invalid := errors.New("Failed to extract accountId from token")
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", invalid
	}
	payload := strings.Map(func(r rune) rune {
		if strings.ContainsRune(" \t\r\n\f", r) {
			return -1
		}
		return r
	}, parts[1])
	var decoded []byte
	var err error
	if strings.Contains(payload, "=") {
		decoded, err = base64.StdEncoding.DecodeString(payload)
	} else {
		decoded, err = base64.RawStdEncoding.DecodeString(payload)
	}
	if err != nil {
		return "", invalid
	}
	latin := make([]rune, len(decoded))
	for i, b := range decoded {
		latin[i] = rune(b)
	}
	claims, ok := samplingObject(json.RawMessage(string(latin)))
	if !ok {
		return "", invalid
	}
	auth, _ := samplingObject(claims["https://api.openai.com/auth"])
	account := auth["chatgpt_account_id"]
	if !samplingTruthy(account) {
		return "", invalid
	}
	return completionsErrorString(account), nil
}

// Defaults precede model and caller overrides. Case-insensitive duplicates use
// set/delete semantics; protocol/auth headers remain authoritative.
func codexNativeHeaders(initial, additional json.RawMessage, account, token string, session *string, websocket bool) (http.Header, error) {
	headers := http.Header{"Originator": {"pi"}, "User-Agent": {completionsUserAgent()}}
	assign := func(name, value string) error {
		if !completionsHeaderName.MatchString(name) {
			return fmt.Errorf("Invalid header name: %q", name)
		}
		value = strings.Trim(value, " \t\r\n")
		if strings.ContainsAny(value, "\r\n\x00") {
			return fmt.Errorf("Invalid header value: %q", value)
		}
		for _, r := range value {
			if r > 255 {
				return fmt.Errorf("Header value cannot be converted to a ByteString")
			}
		}
		headers.Set(name, value)
		return nil
	}
	for i, raw := range []json.RawMessage{initial, additional} {
		fields, _ := samplingObject(raw)
		for _, name := range samplingObjectKeys(raw) {
			value := fields[name]
			if i == 1 && !samplingNonNull(value) {
				headers.Del(name)
				continue
			}
			if err := assign(name, completionsErrorString(value)); err != nil {
				return nil, err
			}
		}
	}
	for _, pair := range [][2]string{{"Authorization", "Bearer " + token}, {"chatgpt-account-id", account}} {
		if err := assign(pair[0], pair[1]); err != nil {
			return nil, err
		}
	}
	if websocket {
		headers.Del("accept")
		headers.Del("content-type")
		headers.Del("OpenAI-Beta")
		headers.Set("OpenAI-Beta", "responses_websockets=2026-02-06")
	} else {
		headers.Set("OpenAI-Beta", "responses=experimental")
		headers.Set("Accept", "text/event-stream")
		headers.Set("Content-Type", "application/json")
	}
	if session != nil && (websocket || *session != "") {
		for _, name := range []string{"session-id", "x-client-request-id"} {
			if err := assign(name, *session); err != nil {
				return nil, err
			}
		}
	}
	return headers, nil
}
