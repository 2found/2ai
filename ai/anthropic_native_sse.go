package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"
)

// Anthropic uses Pi's own SSE reader, not the OpenAI SDK decoder: only named
// message events are parsed, strings get JSON escape repair, and message_start
// requires a matching message_stop. Read boundaries are retained for CR handling.
func readAnthropicSSE(ctx context.Context, body io.Reader, consume func(json.RawMessage) error) error {
	if body == nil {
		return errors.New("Attempted to iterate over an Anthropic response with no body")
	}
	decoder := unicode.UTF8BOM.NewDecoder()
	var pending []byte
	buffer, event := "", ""
	data := []string{}
	rawLines := []string{}
	sawStart, sawEnd := false, false
	dispatch := func() error {
		if event == "" && len(data) == 0 {
			return nil
		}
		name, payload, lines := event, strings.Join(data, "\n"), rawLines
		event = ""
		data = nil
		rawLines = nil
		if name == "error" {
			return errors.New(payload)
		}
		switch name {
		case "message_start", "message_delta", "message_stop", "content_block_start", "content_block_delta", "content_block_stop":
		default:
			return nil
		}
		raw, err := ParseJSONWithRepair(payload)
		if err == nil && strings.TrimSpace(string(raw)) == "null" {
			err = errors.New("null is not an object (evaluating 'event.type')")
		}
		if err != nil {
			return fmt.Errorf("Could not parse Anthropic SSE event %s: %s; data=%s; raw=%s", name, err.Error(), payload, strings.Join(lines, `\n`))
		}
		fields, _ := samplingObject(raw)
		if samplingString(fields["type"]) == "message_start" {
			sawStart = true
		} else if samplingString(fields["type"]) == "message_stop" {
			sawEnd = true
		}
		return consume(raw)
	}
	line := func(value string) error {
		if value == "" {
			return dispatch()
		}
		rawLines = append(rawLines, value)
		if strings.HasPrefix(value, ":") {
			return nil
		}
		field, value, _ := strings.Cut(value, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
		case "data":
			data = append(data, value)
		}
		return nil
	}
	consumeLines := func() error {
		for {
			index := strings.IndexAny(buffer, "\r\n")
			if index < 0 {
				return nil
			}
			end := index + 1
			if buffer[index] == '\r' && end < len(buffer) && buffer[end] == '\n' {
				end++
			}
			text := buffer[:index]
			buffer = buffer[end:]
			if err := line(text); err != nil {
				return err
			}
		}
	}
	decode := func(chunk []byte, eof bool) error {
		pending = append(pending, chunk...)
		dest := make([]byte, len(pending)*3+4)
		written, read, err := decoder.Transform(dest, pending, eof)
		pending = pending[read:]
		if err != nil && err != transform.ErrShortSrc {
			return err
		}
		buffer += string(dest[:written])
		return consumeLines()
	}
	chunk := make([]byte, 16*1024)
	for {
		if ctx.Err() != nil {
			return errors.New("Request was aborted")
		}
		n, err := body.Read(chunk)
		if n > 0 {
			if failure := decode(chunk[:n], false); failure != nil {
				return failure
			}
		}
		if err != nil {
			if err != io.EOF {
				return err
			}
			break
		}
	}
	if err := decode(nil, true); err != nil {
		return err
	}
	if buffer != "" {
		if err := line(buffer); err != nil {
			return err
		}
	}
	if err := dispatch(); err != nil {
		return err
	}
	if sawStart && !sawEnd {
		return errors.New("Anthropic stream ended before message_stop")
	}
	return nil
}
