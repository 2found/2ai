package ai

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"golang.org/x/text/encoding/unicode"
)

// readCompletionsSSE matches the SDK's event/data framing, including CR-only
// lines, multiline data, an unterminated final event, and the exact sentinel.
// It has no Scanner token limit and stops reading immediately at [DONE].
func readCompletionsSSE(ctx context.Context, body io.Reader, headers http.Header, consume func(json.RawMessage) error) error {
	return readNativeJSONSSE(ctx, body, headers, true, consume)
}

func readNativeJSONSSE(ctx context.Context, body io.Reader, headers http.Header, inspectErrors bool, consume func(json.RawMessage) error) error {
	reader := bufio.NewReader(body)
	var line strings.Builder
	var data []string
	event := ""
	hasEvent := false
	skipLF := false
	dispatch := func() (bool, error) {
		if event == "" && len(data) == 0 {
			return false, nil
		}
		payload := strings.Join(data, "\n")
		name, named := event, hasEvent
		event = ""
		hasEvent = false
		data = nil
		if payload == "[DONE]" {
			return true, nil
		}
		raw := json.RawMessage(payload)
		if !json.Valid(raw) {
			return false, errors.New("Error reading response: malformed server-sent event JSON.")
		}
		if named && strings.HasPrefix(name, "thread.") {
			raw, _ = json.Marshal(struct {
				Event string          `json:"event"`
				Data  json.RawMessage `json:"data"`
			}{name, raw})
		} else {
			fields, _ := samplingObject(raw)
			if inspectErrors && name == "error" {
				failure := fields["error"]
				if !samplingNonNull(failure) {
					failure = raw
				}
				return false, newCompletionsRequestError(0, failure, "", headers)
			}
			if inspectErrors && samplingTruthy(fields["error"]) {
				return false, newCompletionsRequestError(0, fields["error"], "", headers)
			}
		}
		return false, consume(raw)
	}
	process := func() (bool, error) {
		// The SDK decodes each complete line separately with TextDecoder,
		// which strips a leading BOM on each line and repairs invalid UTF-8.
		text, err := unicode.UTF8BOM.NewDecoder().String(line.String())
		line.Reset()
		if err != nil {
			return false, err
		}
		if text == "" {
			return dispatch()
		}
		if strings.HasPrefix(text, ":") {
			return false, nil
		}
		field, value, found := strings.Cut(text, ":")
		if !found {
			value = ""
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
			hasEvent = true
		case "data":
			data = append(data, value)
		}
		return false, nil
	}
	for {
		if ctx.Err() != nil {
			return nil
		} // SDK turns transport abort into EOF.
		b, err := reader.ReadByte()
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			if err != io.EOF {
				return err
			}
			if line.Len() > 0 {
				if done, err := process(); done || err != nil {
					return err
				}
			}
			_, err := dispatch()
			return err
		}
		if skipLF {
			skipLF = false
			if b == '\n' {
				continue
			}
		}
		if b == '\r' || b == '\n' {
			skipLF = b == '\r'
			if done, err := process(); done || err != nil {
				return err
			}
		} else {
			line.WriteByte(b)
		}
	}
}
