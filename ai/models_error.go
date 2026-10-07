package ai

import "strings"

// ModelsError retains Pi's public error category and the underlying Go cause.
type ModelsError struct {
	Code    string
	Message string
	Cause   error
}

func (e *ModelsError) Error() string { return e.Message }
func (e *ModelsError) Unwrap() error { return e.Cause }

func NewModelsError(code, message string, cause error) *ModelsError {
	if cause != nil {
		detail := cause.Error()
		if detail == "" {
			detail = "Error"
			if _, ok := cause.(*ModelsError); ok {
				detail = "ModelsError"
			}
		}
		detail = strings.TrimFunc(detail, jsWhitespace)
		if detail != "" && !strings.Contains(message, detail) {
			message += ": " + detail
		}
	}
	return &ModelsError{Code: code, Message: message, Cause: cause}
}
