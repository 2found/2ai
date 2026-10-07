package ai

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// Devin (Cognition SWE / Codeium Cascade) speaks a Connect-RPC protobuf surface
// at server.codeium.com, not an OpenAI-shaped wire. This file holds the
// protocol.LLMProvider adapter (Chat/Stream over the native event stream) and
// the model listing/uid resolution that the OAuth vendor registry expects; the
// wire producer itself lives in devin_native.go.
const (
	// DevinDefaultBaseURL is the Cascade origin used when a provider row sets none.
	DevinDefaultBaseURL = "https://server.codeium.com"

	devinSessionPrefix      = "devin-session-token$"
	devinChatPath           = "/exa.api_server_pb.ApiServerService/GetChatMessage"
	devinAssignModelPath    = "/exa.api_server_pb.ApiServerService/AssignModel"
	devinAuthPath           = "/exa.auth_pb.AuthService/GetUserJwt"
	devinCLIModelsPath      = "/exa.api_server_pb.ApiServerService/GetCliModelConfigs"
	devinModelRouterID      = "adaptive"
	devinCacheTypeEphemeral = 1
	devinRequestTypeCascade = 5
	devinPlannerDefault     = 1
	devinSourceUser         = 1
	devinSourceSystem       = 2
	devinSourceTool         = 4
	devinDefaultMaxTokens   = 64000
	devinDefaultTemperature = 0.4
)

var devinStopPatterns = []string{"<|user|>", "<|bot|>", "<|context_request|>", "<|endoftext|>", "<|end_of_turn|>"}

// DevinProvider is the wire client for the Devin Cascade surface. It is
// stateless per request apart from the account credential the pooled wrapper
// installs via applyOAuthToken.
type DevinProvider struct {
	// BaseURL overrides the Cascade origin; empty uses DevinDefaultBaseURL.
	BaseURL string
	// HTTP serves listing and unary calls.
	HTTP *http.Client
	// StreamHTTP serves the chat stream; nil falls back to HTTP.
	StreamHTTP *http.Client

	// tok is the account credential applied per request by pooledProvider.
	// AccessToken is the Devin session token (the `devin-session-token$`
	// prefix is added at wire time).
	tok OAuthToken
}

func NewDevinProvider() *DevinProvider {
	return &DevinProvider{HTTP: NewChatHTTPClient(0), StreamHTTP: NewStreamHTTPClient(0)}
}

// applyOAuthToken returns a per-call clone carrying the account credential
// (pooledProvider's oauthTokenApplier seam) so concurrent calls never share
// the token field.
func (p *DevinProvider) applyOAuthToken(tok OAuthToken) protocol.LLMProvider {
	c := *p
	c.tok = tok
	return &c
}

func (p *DevinProvider) Name() string        { return VendorDevin }
func (p *DevinProvider) SupportsTools() bool { return true }

func (p *DevinProvider) ModelCapabilities(model string) protocol.ModelCapabilities {
	return CapabilitiesFor(p.Name(), model)
}

func (p *DevinProvider) streamHTTP() *http.Client {
	if p.StreamHTTP != nil {
		return p.StreamHTTP
	}
	return p.HTTP
}

func (p *DevinProvider) baseURL() string {
	base := strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
	if base == "" {
		return DevinDefaultBaseURL
	}
	return base
}

func devinNormalizeSessionToken(token string) string {
	token = strings.TrimSpace(token)
	if token == "" || strings.HasPrefix(token, devinSessionPrefix) {
		return token
	}
	return devinSessionPrefix + token
}

// devinCascadeID returns the thread id shared by every call in one turn. It
// comes from the caller's session identity when present so a resumed
// conversation keeps threading.
func (p *DevinProvider) cascadeID(req protocol.ChatRequest) string {
	if id := strings.TrimSpace(req.SessionID); id != "" {
		return id
	}
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("cascade-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// Stream adapts the native Devin event stream onto the neutral ChatDelta
// contract.
func (p *DevinProvider) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	session := devinNormalizeSessionToken(p.tok.AccessToken)
	if session == "" {
		return nil, &protocol.ProviderError{Provider: p.Name(), Status: http.StatusUnauthorized,
			Message: "devin: no session credential"}
	}
	model, controls, err := devinModelAndControls(req, session, p.baseURL())
	if err != nil {
		return nil, err
	}
	options := OpenAICompletionsStreamOptions{Client: p.streamHTTP()}
	options.Options, _ = json.Marshal(controls)
	transcript := NormalizeContext(Context{
		SystemPrompt: devinSystemPrompt(req.Messages),
		Messages:     devinTranscriptMessages(req.Messages),
		Tools:        devinTranscriptTools(req.Tools),
	})
	// The capture carries the attempt's typed error to the adapter; the event
	// itself can only carry a message string.
	attemptCtx, capture := WithNativeProviderFailure(ctx)
	stream := NewAssistantMessageEventStreamFor(attemptCtx)
	go devinAgentStream(attemptCtx, model, transcript, options, stream)

	out := make(chan protocol.ChatDelta, 32)
	go func() {
		defer close(out)
		for {
			event, ok, err := stream.Next(context.WithoutCancel(ctx))
			if err != nil {
				out <- protocol.ChatDelta{Err: err}
				return
			}
			if !ok {
				return
			}
			switch event.Type {
			case "text_delta":
				out <- protocol.ChatDelta{ContentDelta: event.Delta}
			case "thinking_delta":
				// Devin thinking is opaque reasoning; carry it as a provider
				// replay block at block end rather than as visible content.
			case "toolcall_end":
				if event.ToolCall != nil {
					out <- protocol.ChatDelta{ToolCall: &protocol.ToolCall{
						ID: event.ToolCall.ID, Name: event.ToolCall.Name, Arguments: string(event.ToolCall.Arguments),
					}}
				}
			case "done":
				usage := protocol.Usage{}
				if event.Message != nil && event.Message.Usage != nil {
					usage = protocol.Usage{
						InputTokens:      int(event.Message.Usage.Input),
						OutputTokens:     int(event.Message.Usage.Output),
						CacheReadTokens:  int(event.Message.Usage.CacheRead),
						CacheWriteTokens: int(event.Message.Usage.CacheWrite),
					}
				}
				out <- protocol.ChatDelta{Done: true, StopReason: event.Reason, Usage: usage}
				return
			case "error":
				message := "devin request failed"
				if event.Error != nil && event.Error.ErrorMessage != nil {
					message = *event.Error.ErrorMessage
				}
				err := capture.Failure()
				var providerError *protocol.ProviderError
				if !errors.As(err, &providerError) {
					providerError = &protocol.ProviderError{Provider: VendorDevin,
						Status: devinStatusFromMessage(message), Message: message}
				}
				out <- protocol.ChatDelta{Err: providerError}
				return
			}
		}
	}()
	return out, nil
}

// devinSystemPrompt joins the request's system messages; devinTranscriptMessages
// drops them because Cascade carries the prompt as a dedicated request field.
func devinSystemPrompt(messages []protocol.Message) string {
	parts := []string{}
	for _, m := range messages {
		if m.Role == protocol.RoleSystem && strings.TrimSpace(m.Content) != "" {
			parts = append(parts, m.Content)
		}
	}
	return joinPromptParts(parts)
}

// devinTranscriptTools renders the neutral tool schemas into transcript Tools.
// An enabled strict hint becomes a "prefer" JSON-schema sampling declaration so
// the wire builder resolves it through ResolveJSONSchemaStrictSampling rather
// than failing a request an unconstrained schema could still serve.
func devinTranscriptTools(schemas []protocol.ToolSchema) []Tool {
	out := make([]Tool, 0, len(schemas))
	for _, s := range schemas {
		parameters, err := json.Marshal(s.Parameters)
		if err != nil || len(s.Parameters) == 0 {
			parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		tool := Tool{Name: s.Name, Description: s.Description, Parameters: parameters}
		if s.Strict == protocol.ToolStrictEnabled {
			tool.ConstrainedSampling = json.RawMessage(`{"type":"json_schema","strict":"prefer"}`)
		}
		out = append(out, tool)
	}
	return out
}

// devinTranscriptMessages drops system messages: the system prompt is carried
// separately on the Cascade request (devinSystemPrompt above).
func devinTranscriptMessages(messages []protocol.Message) []Message {
	out := make([]Message, 0, len(messages))
	for _, m := range messages {
		if m.Role == protocol.RoleSystem {
			continue
		}
		converted := Message{Role: string(m.Role), Content: TextContent(m.Content)}
		switch m.Role {
		case protocol.RoleAssistant:
			blocks := []ContentBlock{}
			if m.Content != "" {
				blocks = append(blocks, ContentBlock{Type: "text", Text: m.Content})
			}
			for _, call := range m.ToolCalls {
				blocks = append(blocks, ContentBlock{Type: "toolCall", ID: call.ID, Name: call.Name,
					Arguments: json.RawMessage(call.Arguments)})
			}
			converted.Content = BlockContent(blocks...)
		case protocol.RoleTool:
			converted.Role = "toolResult"
			converted.ToolCallID = m.ToolCallID
			converted.ToolName = m.Name
			converted.IsError = m.Error != ""
		}
		out = append(out, converted)
	}
	return out
}

// Chat consumes the stream to completion (Cascade has no non-streaming mode).
func (p *DevinProvider) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	return chatViaStream(ctx, p, req)
}

// devinModelAndControls renders the neutral request into the model JSON the
// native stream expects plus its serializable controls. base is the configured
// endpoint (already defaulted); the model JSON must carry it — the stream
// prefers the model's baseUrl over its own fallback.
func devinModelAndControls(req protocol.ChatRequest, session, base string) (json.RawMessage, map[string]any, error) {
	model, err := json.Marshal(map[string]any{
		"id": req.Model, "name": req.Model, "api": VendorDevin, "provider": VendorDevin,
		"baseUrl": base, "reasoning": true, "input": []string{"text", "image"},
		"contextWindow": 262144, "maxTokens": devinDefaultMaxTokens,
	})
	if err != nil {
		return nil, nil, err
	}
	controls := map[string]any{"apiKey": session}
	if req.ReasoningEffort != "" {
		controls["reasoning"] = req.ReasoningEffort
	}
	if req.MaxTokens > 0 {
		controls["maxTokens"] = req.MaxTokens
	}
	if req.Temperature > 0 {
		controls["temperature"] = req.Temperature
	}
	if req.SessionID != "" {
		controls["sessionId"] = req.SessionID
	}
	if req.ParallelToolCalls != nil {
		controls["parallelToolCalls"] = *req.ParallelToolCalls
	}
	return model, controls, nil
}

// --- model listing and uid resolution ---

type devinCatalogEntry struct {
	uid        string
	label      string
	disabled   bool
	images     bool
	contextMax int64
	maxTokens  int64
	family     *devinFamilyMetadata
	isDefault  bool
}

// devinCatalogCache memoizes catalog entries per (base URL, session token).
// More than one slot is needed: the pooled wrapper rotates sessions, and a
// single slot would bounce the entry on every other turn.
var devinCatalogCache = struct {
	sync.Mutex
	entries map[string]devinCatalogCached
}{entries: map[string]devinCatalogCached{}}

type devinCatalogCached struct {
	expires time.Time
	entries []devinCatalogEntry
}

const (
	devinCatalogTTL      = 10 * time.Minute
	devinCatalogCacheMax = 8
)

// devinFetchCatalog calls GetCliModelConfigs and returns the decoded configs.
// The result is cached per (base URL, session token) for a short window so a
// run does not pay a discovery round-trip on every turn.
func devinFetchCatalog(ctx context.Context, client HTTPDoer, base, session string) ([]devinCatalogEntry, error) {
	if client == nil {
		client = defaultHTTP()
	}
	key := base + "\x00" + session
	devinCatalogCache.Lock()
	if cached, ok := devinCatalogCache.entries[key]; ok && time.Now().Before(cached.expires) && cached.entries != nil {
		entries := cached.entries
		devinCatalogCache.Unlock()
		return entries, nil
	}
	devinCatalogCache.Unlock()

	metadata := devinCLIIdentity()
	metadata.apiKey = devinNormalizeSessionToken(session)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, base+devinCLIModelsPath,
		bytes.NewReader(devinEncodeGetCliModelConfigsRequest(metadata)))
	if err != nil {
		return nil, err
	}
	request.Header.Set("content-type", "application/proto")
	request.Header.Set("connect-protocol-version", "1")
	request.Header.Set("accept", "*/*")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode/100 != 2 {
		return nil, &protocol.ProviderError{Provider: VendorDevin, Status: response.StatusCode,
			Message: strings.TrimSpace(string(raw))}
	}
	configs, err := devinDecodeCliModelConfigs(raw)
	if err != nil {
		return nil, err
	}
	entries := make([]devinCatalogEntry, 0, len(configs))
	for _, config := range configs {
		uid := strings.TrimSpace(config.modelUID)
		if uid == "" {
			continue
		}
		entries = append(entries, devinCatalogEntry{
			uid: uid, label: strings.TrimSpace(config.label), disabled: config.disabled,
			images: config.supportsImages, contextMax: config.maxTokens, maxTokens: config.maxTokens,
			family: config.family, isDefault: config.isDefaultModelInFamily,
		})
	}
	// An empty catalog means the endpoint is mid-deploy or the account is not
	// provisioned yet — cache only real entries so a later turn refetches rather
	// than spending the TTL on suffix-guessed uids.
	if len(entries) > 0 {
		devinCatalogCache.Lock()
		if len(devinCatalogCache.entries) >= devinCatalogCacheMax {
			devinCatalogCache.entries = map[string]devinCatalogCached{}
		}
		devinCatalogCache.entries[key] = devinCatalogCached{expires: time.Now().Add(devinCatalogTTL), entries: entries}
		devinCatalogCache.Unlock()
	}
	return entries, nil
}

// listDevinModels returns the wire uids the account can chat with.
func (p *DevinProvider) listDevinModels(ctx context.Context, client HTTPDoer, tok OAuthToken) ([]Model, error) {
	entries, err := devinFetchCatalog(ctx, client, p.baseURL(), tok.AccessToken)
	if err != nil {
		return nil, err
	}
	models := make([]Model, 0, len(entries))
	for _, entry := range entries {
		if entry.disabled {
			continue
		}
		window := int(entry.contextMax)
		if window <= 0 {
			window = 262144
		}
		models = append(models, Model{ID: entry.uid, API: VendorDevin, Provider: VendorDevin,
			ContextWindow: window, Input: devinModelInput(entry.images)})
	}
	return models, nil
}

func devinModelInput(images bool) []string {
	if images {
		return []string{"text", "image"}
	}
	return []string{"text"}
}

// devinResolveModelUID maps a configured model name (a logical family id or a
// raw wire uid) onto the wire uid for the requested effort. When the catalog
// cannot be reached the configured name is passed through unchanged.
func devinResolveModelUID(ctx context.Context, client HTTPDoer, base, session, modelID, effort string) string {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" || modelID == devinModelRouterID {
		return modelID
	}
	entries, err := devinFetchCatalog(ctx, client, base, session)
	if err != nil {
		return devinSuffixModelUID(modelID, effort)
	}
	for _, entry := range entries {
		if entry.uid == modelID {
			return modelID
		}
	}
	normalized := devinNormalizeLabel(modelID)
	var lane []devinCatalogEntry
	var defaultUID string
	for _, entry := range entries {
		if entry.disabled || entry.family == nil || devinNormalizeLabel(entry.family.label) != normalized {
			continue
		}
		lane = append(lane, entry)
		if defaultUID == "" && (entry.isDefault || entry.family.isDefaultModelInFamily) {
			defaultUID = entry.uid
		}
	}
	if len(lane) == 0 {
		return devinSuffixModelUID(modelID, effort)
	}
	if uid := devinEffortMember(lane, effort); uid != "" {
		return uid
	}
	if defaultUID != "" {
		return defaultUID
	}
	if uid := devinEffortMember(lane, "high"); uid != "" {
		return uid
	}
	return lane[0].uid
}

// devinEffortMember picks the family member whose declared effort matches.
func devinEffortMember(lane []devinCatalogEntry, effort string) string {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "" {
		return ""
	}
	for _, entry := range lane {
		if entry.family == nil {
			continue
		}
		for _, meta := range entry.family.entries {
			if meta.value == nil {
				continue
			}
			if !devinIsEffortKey(meta.key) {
				continue
			}
			if devinEffortName(meta.value.name) == effort {
				return entry.uid
			}
		}
	}
	return ""
}

func devinIsEffortKey(key string) bool {
	switch devinNormalizeLabel(key) {
	case "effort", "reasoning effort", "thinking":
		return true
	}
	return false
}

// devinEffortName normalizes an effort display name ("X High", "No Thinking").
func devinEffortName(name string) string {
	normalized := devinNormalizeLabel(name)
	switch normalized {
	case "none", "no thinking", "nothinking":
		return "off"
	}
	return normalized
}

// devinNormalizeLabel lowercases and collapses punctuation to single spaces.
func devinNormalizeLabel(value string) string {
	var b strings.Builder
	space := false
	for _, r := range strings.ToLower(strings.TrimSpace(value)) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			space = false
		default:
			if !space && b.Len() > 0 {
				b.WriteByte(' ')
				space = true
			}
		}
	}
	return strings.TrimSpace(b.String())
}

// devinSuffixModelUID is the offline fallback: Devin names most effort tiers
// `<base>-<effort>`.
func devinSuffixModelUID(modelID, effort string) string {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "" || effort == "off" || effort == "none" {
		return modelID
	}
	if strings.HasSuffix(modelID, "-"+effort) {
		return modelID
	}
	return modelID + "-" + effort
}
