package ai

import (
	"errors"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// IsContextOverflow recognizes explicit context rejections, not transient
// errors, arbitrary payload limits or an ambiguous transport interruption.
func IsContextOverflow(err error) bool {
	if err == nil {
		return false
	}
	var provider *protocol.ProviderError
	if errors.As(err, &provider) && provider.Status != 0 && provider.Status != 400 && provider.Status != 413 {
		return false
	}
	text := strings.ToLower(err.Error())
	for _, phrase := range []string{"context_length_exceeded", "context_window_exceeded", "maximum context length", "context window exceeded", "prompt is too long"} {
		if strings.Contains(text, phrase) {
			return true
		}
	}
	return false
}
