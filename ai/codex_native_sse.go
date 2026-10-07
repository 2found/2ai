package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"
)

// readCodexSSE follows Codex's own framing rather than the OpenAI SDK parser:
// only LF/LF terminates a frame, [DONE] is skipped, and EOF flushes a residual
// frame. consume returns true when the mapped terminal event ends admission.
// Closing the body releases its reader on termination, failure or cancellation.
func readCodexSSE(ctx context.Context, body io.ReadCloser, consume func(json.RawMessage) (bool, error)) error {
	if body == nil {
		return nil
	}
	defer body.Close()
	stop := context.AfterFunc(ctx, func() { _ = body.Close() })
	defer stop()
	reader := transform.NewReader(body, unicode.UTF8BOM.NewDecoder())
	buffer := ""
	bytes := make([]byte, 16*1024)
	for {
		if ctx.Err() != nil {
			return fmt.Errorf("Request was aborted")
		}
		n, err := reader.Read(bytes)
		if ctx.Err() != nil {
			return fmt.Errorf("Request was aborted")
		}
		buffer += string(bytes[:n])
		if err == io.EOF && strings.TrimFunc(buffer, jsWhitespace) != "" {
			buffer += "\n\n"
		}
		for {
			index := strings.Index(buffer, "\n\n")
			if index < 0 {
				break
			}
			frame := buffer[:index]
			buffer = buffer[index+2:]
			var data []string
			for _, line := range strings.Split(frame, "\n") {
				if strings.HasPrefix(line, "data:") {
					data = append(data, strings.TrimFunc(line[5:], jsWhitespace))
				}
			}
			payload := strings.TrimFunc(strings.Join(data, "\n"), jsWhitespace)
			if payload == "" || payload == "[DONE]" {
				continue
			}
			if !json.Valid([]byte(payload)) {
				var value any
				failure := json.Unmarshal([]byte(payload), &value)
				return fmt.Errorf("Invalid Codex SSE JSON: %v", failure)
			}
			done, failure := consume(json.RawMessage(payload))
			if failure != nil {
				return failure
			}
			if done {
				return nil
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}
