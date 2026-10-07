// Provider contracts are owned by ai/protocol. These aliases keep existing
// plugin and SDK implementations source-compatible during the engine migration.
package agentcore

import (
	"github.com/2found/2ai/ai/protocol"
	"net/http"
)

type CapabilitySupport = protocol.CapabilitySupport
type ModelCapabilities = protocol.ModelCapabilities
type Role = protocol.Role
type Message = protocol.Message
type ReasoningBlock = protocol.ReasoningBlock
type ContentPart = protocol.ContentPart
type ToolCall = protocol.ToolCall
type ToolSchema = protocol.ToolSchema
type ToolStrictness = protocol.ToolStrictness
type ToolChoiceMode = protocol.ToolChoiceMode
type ToolChoice = protocol.ToolChoice
type Usage = protocol.Usage
type ChatRequest = protocol.ChatRequest
type OutputSchema = protocol.OutputSchema
type ChatResponse = protocol.ChatResponse
type ChatDelta = protocol.ChatDelta
type LLMProvider = protocol.LLMProvider
type ModelCapabilityProvider = protocol.ModelCapabilityProvider
type KeyUpdater = protocol.KeyUpdater
type ProviderSessionState = protocol.ProviderSessionState
type AccountScopedProviderState = protocol.AccountScopedProviderState
type ProviderSession = protocol.ProviderSession
type ProviderError = protocol.ProviderError
type RetryPolicy = protocol.RetryPolicy

const (
	CapabilityUnknown      = protocol.CapabilityUnknown
	CapabilitySupported    = protocol.CapabilitySupported
	CapabilityUnsupported  = protocol.CapabilityUnsupported
	RoleSystem             = protocol.RoleSystem
	RoleUser               = protocol.RoleUser
	RoleAssistant          = protocol.RoleAssistant
	RoleTool               = protocol.RoleTool
	ReasoningBlockThinking = protocol.ReasoningBlockThinking
	ReasoningBlockRedacted = protocol.ReasoningBlockRedacted
	ContentPartText        = protocol.ContentPartText
	ContentPartImage       = protocol.ContentPartImage
	ToolStrictDefault      = protocol.ToolStrictDefault
	ToolStrictEnabled      = protocol.ToolStrictEnabled
	ToolStrictDisabled     = protocol.ToolStrictDisabled
	ToolChoiceDefault      = protocol.ToolChoiceDefault
	ToolChoiceAuto         = protocol.ToolChoiceAuto
	ToolChoiceNone         = protocol.ToolChoiceNone
	ToolChoiceRequired     = protocol.ToolChoiceRequired
	ToolChoiceNamed        = protocol.ToolChoiceNamed
)

func CapabilitiesOf(provider LLMProvider, model string) ModelCapabilities {
	return protocol.CapabilitiesOf(provider, model)
}
func NewProviderSession() *ProviderSession { return protocol.NewProviderSession() }
func NewProviderError(provider string, response *http.Response, message string) *ProviderError {
	return protocol.NewProviderError(provider, response, message)
}
func IsRetryable(err error) bool      { return protocol.IsRetryable(err) }
func DefaultRetryPolicy() RetryPolicy { return protocol.DefaultRetryPolicy() }

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
