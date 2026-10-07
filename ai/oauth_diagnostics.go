package ai

import (
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// OAuthDiagnosticError carries the Error fields read by Pi's OAuth diagnostic
// formatter. Nil metadata means absent; Null represents an explicit null cause
// or errno. Stack is host-provided: a Go stack is not a JavaScript stack.
type OAuthDiagnosticError struct {
	Name    *string
	Message string
	Code    any
	Errno   any
	Cause   any
	Stack   string
	stack   func() string
}

func (e *OAuthDiagnosticError) Error() string { return e.Message }
func (e *OAuthDiagnosticError) Unwrap() error {
	cause, _ := e.Cause.(error)
	return cause
}

func formatOAuthErrorDetails(value any) (string, error) {
	failure, isError := value.(error)
	if !isError {
		return catalogKey(value, make(map[*Array]bool))
	}
	name := "Error"
	metadata, _ := failure.(*OAuthDiagnosticError)
	if metadata != nil && metadata.Name != nil {
		name = *metadata.Name
	}
	details := []string{name + ": " + failure.Error()}
	if metadata != nil {
		appendValue := func(label string, value any) error {
			text, err := catalogKey(value, make(map[*Array]bool))
			if err == nil {
				details = append(details, label+text)
			}
			return err
		}
		if catalogEntryTruthy(metadata.Code) {
			if err := appendValue("code=", metadata.Code); err != nil {
				return "", err
			}
		}
		if metadata.Errno != nil && !jsonjs.IsUndefined(metadata.Errno) {
			if err := appendValue("errno=", metadata.Errno); err != nil {
				return "", err
			}
		}
		if metadata.Cause != nil && !jsonjs.IsUndefined(metadata.Cause) {
			cause, err := formatOAuthErrorDetails(metadata.Cause)
			if err != nil {
				return "", err
			}
			details = append(details, "cause="+cause)
		}
		stack := metadata.Stack
		if stack == "" && metadata.stack != nil {
			stack = metadata.stack()
		}
		if stack != "" {
			details = append(details, "stack="+stack)
		}
	}
	return strings.Join(details, "; "), nil
}
