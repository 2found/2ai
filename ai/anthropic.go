package ai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

const (
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	anthropicVersion        = "2023-06-01"
	anthropicDefaultTokens  = 4096
	// Reasoning replay material is provider-controlled opaque state that becomes
	// durable session data. Keep legitimate extended-thinking turns intact while
	// preventing a malformed stream from growing a laptop or server process
	// without bound. Oversized blocks are dropped whole because truncating a
	// signature or encrypted payload would make it invalid on replay.
	maxAnthropicReasoningBlocks     = 32
	maxAnthropicReasoningBlockBytes = 1 << 20
	maxAnthropicReasoningTotalBytes = 4 << 20
	// anthropicExtendedCacheBeta is the beta opt-in required for the 1-hour cache
	// window; the default 5-minute window needs no beta header.
	anthropicExtendedCacheBeta = "extended-cache-ttl-2025-04-11"
	// anthropicStructuredOutputBeta is the beta opt-in for output_format
	// (grammar-constrained JSON answers).
	anthropicStructuredOutputBeta = "structured-outputs-2025-11-13"

	// claudeCodeUserAgent is the User-Agent the Claude Code CLI sends; the
	// OAuth surface expects the subscription client's fingerprint, not a
	// generic SDK one.
	claudeCodeUserAgent = "claude-cli/2.1.257 (external, cli)"
	// claudeCodeSystemInstruction is the identity block Claude Code prepends
	// to every request's system prompt.
	claudeCodeSystemInstruction = "You are Claude Code, Anthropic's official CLI for Claude."
)

// claudeCodeBetas are the anthropic-beta values the Claude Code OAuth client
// always advertises, merged with whatever a request's own features need.
var claudeCodeBetas = []string{
	"claude-code-20250219",
	"oauth-2025-04-20",
	"interleaved-thinking-2025-05-14",
	"context-management-2025-06-27",
	"prompt-caching-scope-2026-01-05",
}

// antBetaHeader assembles the anthropic-beta header for a request: the
// comma-joined set of betas it actually uses, or "" when none apply. In OAuth
// (claude-code) mode the subscription client's fixed beta set always applies
// and the per-request betas merge on top.
func antBetaHeader(req protocol.ChatRequest, oauth bool) string {
	var betas []string
	if oauth {
		betas = append(betas, claudeCodeBetas...)
	}
	if usesExtendedCache(req) {
		betas = append(betas, anthropicExtendedCacheBeta)
	}
	if req.OutputSchema != nil {
		betas = append(betas, anthropicStructuredOutputBeta)
	}
	return strings.Join(betas, ",")
}

// antCacheTTL maps the neutral CacheRetention hint onto an Anthropic cache TTL:
// a long/24h hint asks for the extended 1-hour window, anything else uses the
// default 5-minute window (empty TTL).
func antCacheTTL(retention string) string {
	switch retention {
	case "long", "24h":
		return "1h"
	default:
		return ""
	}
}

// usesExtendedCache reports whether the request opts into the 1-hour cache
// window, which requires the extended-cache beta header.
func usesExtendedCache(req protocol.ChatRequest) bool {
	return req.CacheKey != "" && antCacheTTL(req.CacheRetention) == "1h"
}

// AnthropicProvider speaks the Anthropic Messages API. The wire format diverges
// from OpenAI completions (top-level system, tool_use/tool_result content
// blocks, x-api-key auth), so per the spec it gets its own implementation
// rather than a compat entry — and proves the protocol.LLMProvider seam (§12 AC: a second
// provider needs no edits to agent.go).
type AnthropicProvider struct {
	APIKey  string
	keyMu   sync.RWMutex
	BaseURL string
	// OAuth marks the claude-code subscription mode: the credential is an OAuth
	// access token sent as Authorization: Bearer (never x-api-key), the request
	// carries the Claude Code client fingerprint (User-Agent, x-app, the fixed
	// beta set), and the system prompt gains the Claude Code identity block.
	OAuth bool
	// HTTP serves the non-streamed Chat path (absolute cap, no header deadline).
	HTTP *http.Client
	// StreamHTTP serves the SSE path, where a header deadline is meaningful. Nil
	// falls back to HTTP, so a caller that overrides only HTTP still works.
	StreamHTTP *http.Client
}

// streamHTTP is the client the SSE path uses: StreamHTTP when set, otherwise
// whatever the caller put on HTTP.
func (p *AnthropicProvider) streamHTTP() *http.Client {
	if p.StreamHTTP != nil {
		return p.StreamHTTP
	}
	return p.HTTP
}

// NewAnthropicProvider builds a provider; an empty baseURL uses the vendor
// default.
func NewAnthropicProvider(apiKey, baseURL string) *AnthropicProvider {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = defaultAnthropicBaseURL
	}
	return &AnthropicProvider{
		APIKey:     apiKey,
		BaseURL:    strings.TrimRight(baseURL, "/"),
		HTTP:       NewChatHTTPClient(0),
		StreamHTTP: NewStreamHTTPClient(0),
	}
}

// UpdateAPIKey swaps the key used for subsequent requests (protocol.KeyUpdater), letting
// the loop refresh an expiring BYO token between turns.
func (p *AnthropicProvider) UpdateAPIKey(key string) {
	if key != "" {
		p.keyMu.Lock()
		p.APIKey = key
		p.keyMu.Unlock()
	}
}

func (p *AnthropicProvider) apiKey() string {
	p.keyMu.RLock()
	defer p.keyMu.RUnlock()
	return p.APIKey
}

// applyOAuthToken returns a per-call clone carrying the account credential
// (pooledProvider's oauthTokenApplier seam). The access token rides in APIKey;
// the OAuth flag makes it a Bearer credential instead of an x-api-key. The
// clone keeps concurrent calls from overwriting each other's token on the
// shared client.
func (p *AnthropicProvider) applyOAuthToken(tok OAuthToken) protocol.LLMProvider {
	return &AnthropicProvider{
		APIKey: tok.AccessToken, BaseURL: p.BaseURL, OAuth: p.OAuth,
		HTTP: p.HTTP, StreamHTTP: p.StreamHTTP,
	}
}

func (p *AnthropicProvider) Name() string {
	if p.OAuth {
		return VendorClaudeCode
	}
	return "anthropic"
}
func (p *AnthropicProvider) SupportsTools() bool { return true }

func (p *AnthropicProvider) ModelCapabilities(model string) protocol.ModelCapabilities {
	return CapabilitiesFor(p.Name(), model)
}

// setHeaders applies the request headers for the configured auth mode. OAuth
// mode sends the Claude Code client fingerprint — Bearer auth, the CLI
// User-Agent, x-app: cli, the fixed subscription beta set merged with the
// request's own betas, and Accept: application/json even when streaming (the
// OAuth surface answers SSE either way).
func (p *AnthropicProvider) setHeaders(httpReq *http.Request, req protocol.ChatRequest, stream bool) {
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	key := p.apiKey()
	if p.OAuth {
		httpReq.Header.Set("Authorization", "Bearer "+key)
		httpReq.Header.Set("User-Agent", claudeCodeUserAgent)
		httpReq.Header.Set("x-app", "cli")
		httpReq.Header.Set("Accept", "application/json")
	} else {
		httpReq.Header.Set("x-api-key", key)
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	}
	if betas := antBetaHeader(req, p.OAuth); betas != "" {
		httpReq.Header.Set("anthropic-beta", betas)
	}
}

// --- wire types (Anthropic Messages) ---

type antContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	// thinking / redacted_thinking replay payloads
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"`
	// image
	Source *antImageSource `json:"source,omitempty"`
	// CacheControl marks this block as a prompt-cache breakpoint. encode stamps
	// it on the final block of the final message (the "moving breakpoint"), so
	// an agent loop's whole transcript-so-far is read from cache next turn.
	CacheControl *antCacheControl `json:"cache_control,omitempty"`
}

type antImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type antMessage struct {
	Role    string            `json:"role"`
	Content []antContentBlock `json:"content"`
}

// antCacheControl marks a prefix block as cacheable (Anthropic prompt caching).
// Type is always "ephemeral"; TTL is "" (the default 5-minute window) or "1h"
// (the extended window, which also needs the extended-cache-ttl beta header).
type antCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// antSystemBlock is the structured form of the system prompt, used only when
// caching is requested so the stable system prefix can carry cache_control.
type antSystemBlock struct {
	Type         string           `json:"type"`
	Text         string           `json:"text"`
	CacheControl *antCacheControl `json:"cache_control,omitempty"`
}

type antTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type antRequest struct {
	Model     string `json:"model"`
	MaxTokens int    `json:"max_tokens"`
	// System is a plain string, or — when caching is requested — a []antSystemBlock
	// whose last block carries cache_control. any keeps the common (uncached) path
	// emitting the same bare string the API has always received.
	System   any          `json:"system,omitempty"`
	Messages []antMessage `json:"messages"`
	Tools    []antTool    `json:"tools,omitempty"`
	Stream   bool         `json:"stream,omitempty"`
	// OutputFormat constrains the text answer to a JSON Schema (structured
	// outputs beta). Sent only when the neutral request carries an protocol.OutputSchema.
	OutputFormat *antOutputFormat `json:"output_format,omitempty"`
	ToolChoice   *antToolChoice   `json:"tool_choice,omitempty"`
}

type antToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

// antOutputFormat is Anthropic's structured-output selector (json_schema form).
type antOutputFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema"`
}

// antUsage mirrors Anthropic's usage block. input_tokens already *excludes* the
// cached prefix, so it maps straight onto the neutral InputTokens; the two cache
// counters map onto CacheReadTokens (a cache hit) and CacheWriteTokens (cache
// creation).
type antUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

func (u antUsage) usage() protocol.Usage {
	return protocol.Usage{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens,
	}
}

type antResponse struct {
	Content    []antContentBlock `json:"content"`
	StopReason string            `json:"stop_reason"`
	Usage      antUsage          `json:"usage"`
	Error      *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func anthropicReasoningBlockBytes(block protocol.ReasoningBlock) int {
	return len(block.Text) + len(block.Signature) + len(block.Data)
}

func validAnthropicReasoningBlock(block protocol.ReasoningBlock) bool {
	if anthropicReasoningBlockBytes(block) > maxAnthropicReasoningBlockBytes {
		return false
	}
	switch block.Type {
	case protocol.ReasoningBlockThinking:
		return block.Signature != ""
	case protocol.ReasoningBlockRedacted:
		return block.Data != ""
	default:
		return false
	}
}

// replayScope binds opaque thinking signatures to the exact adapter endpoint
// and model that issued them. The digest avoids persisting deployment URLs in
// the neutral transcript while preventing cross-provider/model replay.
func (p *AnthropicProvider) replayScope(model string) string {
	sum := sha256.Sum256([]byte(p.Name() + "\x00" + strings.TrimRight(p.BaseURL, "/") + "\x00" + model))
	return fmt.Sprintf("anthropic:%x", sum[:12])
}

// Chat performs one non-streaming Messages call.
func (p *AnthropicProvider) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	body := p.encode(req)
	raw, err := json.Marshal(body)
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/v1/messages", bytes.NewReader(raw))
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	p.setHeaders(httpReq, req, false)

	resp, err := p.HTTP.Do(httpReq)
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	defer resp.Body.Close()
	var data []byte
	var readErr error
	if resp.StatusCode >= 400 {
		data = readProviderErrorBody(resp.Body)
	} else {
		data, readErr = io.ReadAll(resp.Body)
	}
	// Same trap the OpenAI path had: a body severed mid-flight decodes as a short
	// but well-formed answer, so a truncated turn looks like a successful one.
	// nil resp => Status 0 => IsRetryable, so the ladder retries instead.
	if readErr != nil && resp.StatusCode < 400 {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), nil,
			fmt.Sprintf("truncated response body (status %d): %s", resp.StatusCode, describeReadErr(ctx, p.HTTP, readErr)))
	}

	if resp.StatusCode >= 400 {
		message := strings.TrimSpace(string(data))
		var errorBody antResponse
		if json.Unmarshal(data, &errorBody) == nil && errorBody.Error != nil && errorBody.Error.Message != "" {
			message = errorBody.Error.Message
		}
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), resp, message)
	}
	if providerErr, ok := inBandProviderError(p.Name(), resp, data); ok {
		return protocol.ChatResponse{}, providerErr
	}

	var decoded antResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		return protocol.ChatResponse{}, fmt.Errorf("decode anthropic response (status %d): %w", resp.StatusCode, err)
	}

	msg := protocol.Message{Role: protocol.RoleAssistant}
	replayScope := p.replayScope(req.Model)
	reasoningBytes := 0
	for _, block := range decoded.Content {
		switch block.Type {
		case "text":
			msg.Content += block.Text
		case "thinking":
			candidate := protocol.ReasoningBlock{
				Type: protocol.ReasoningBlockThinking, Text: block.Thinking,
				Signature: block.Signature, ReplayScope: replayScope,
			}
			if validAnthropicReasoningBlock(candidate) &&
				len(msg.ReasoningBlocks) < maxAnthropicReasoningBlocks &&
				reasoningBytes+anthropicReasoningBlockBytes(candidate) <= maxAnthropicReasoningTotalBytes {
				msg.ReasoningBlocks = append(msg.ReasoningBlocks, candidate)
				reasoningBytes += anthropicReasoningBlockBytes(candidate)
			}
		case "redacted_thinking":
			candidate := protocol.ReasoningBlock{
				Type: protocol.ReasoningBlockRedacted, Data: block.Data, ReplayScope: replayScope,
			}
			if validAnthropicReasoningBlock(candidate) &&
				len(msg.ReasoningBlocks) < maxAnthropicReasoningBlocks &&
				reasoningBytes+anthropicReasoningBlockBytes(candidate) <= maxAnthropicReasoningTotalBytes {
				msg.ReasoningBlocks = append(msg.ReasoningBlocks, candidate)
				reasoningBytes += anthropicReasoningBlockBytes(candidate)
			}
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, protocol.ToolCall{
				ID: block.ID, Name: block.Name, Arguments: string(block.Input),
			})
		}
	}
	return protocol.ChatResponse{
		Message:    msg,
		StopReason: decoded.StopReason,
		Usage:      decoded.Usage.usage(),
	}, nil
}

// --- streaming wire types (Anthropic Messages, stream=true) ---

// antStreamEvent is the union of the Messages SSE event payloads we consume.
// Each data line carries a discriminating "type"; we dispatch on it rather than
// the parallel `event:` line.
type antStreamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	// content_block_start
	ContentBlock *antContentBlock `json:"content_block,omitempty"`
	// content_block_delta
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text,omitempty"`
		PartialJSON string `json:"partial_json,omitempty"`
		Thinking    string `json:"thinking,omitempty"`
		Signature   string `json:"signature,omitempty"`
		StopReason  string `json:"stop_reason,omitempty"`
	} `json:"delta,omitempty"`
	// message_start / message_delta
	Message *struct {
		Usage antUsage `json:"usage"`
	} `json:"message,omitempty"`
	Usage *struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type antReasoningCapture struct {
	block   protocol.ReasoningBlock
	invalid bool
}

// Stream performs a real token-streaming Messages call: text_delta blocks are
// forwarded as content deltas; tool_use blocks accumulate their input_json_delta
// fragments (keyed by content-block index) into whole ToolCalls flushed before
// the terminal Done delta. Proves the streaming seam holds for a second provider
// with no edits to the loop (§12 AC).
func (p *AnthropicProvider) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	body := p.encode(req)
	body.Stream = true
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/v1/messages", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq, req, true)

	resp, err := p.streamHTTP().Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		data := readProviderErrorBody(resp.Body)
		resp.Body.Close()
		return nil, protocol.NewProviderError(p.Name(), resp, strings.TrimSpace(string(data)))
	}

	ch := make(chan protocol.ChatDelta, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		// Each content-block index is either text or a tool_use call assembled
		// from JSON fragments; track the in-progress tool calls by index.
		toolAcc := map[int]*protocol.ToolCall{}
		var order []int
		reasoningAcc := map[int]*antReasoningCapture{}
		var reasoningOrder []int
		emittedReasoning := map[int]bool{}
		reasoningBytes := 0
		var stopReason string
		var usage protocol.Usage
		terminal := false
		replayScope := p.replayScope(req.Model)
		emitReasoning := func(index int) {
			if emittedReasoning[index] {
				return
			}
			capture := reasoningAcc[index]
			if capture == nil || capture.invalid {
				return
			}
			if !validAnthropicReasoningBlock(capture.block) {
				return
			}
			emittedReasoning[index] = true
			copy := capture.block
			ch <- protocol.ChatDelta{ReasoningBlock: &copy}
		}
		invalidateReasoning := func(capture *antReasoningCapture) {
			if capture == nil || capture.invalid {
				return
			}
			reasoningBytes -= anthropicReasoningBlockBytes(capture.block)
			capture.block = protocol.ReasoningBlock{}
			capture.invalid = true
		}
		appendReasoning := func(capture *antReasoningCapture, delta string, signature bool) {
			if capture == nil || capture.invalid || delta == "" {
				return
			}
			current := anthropicReasoningBlockBytes(capture.block)
			if current+len(delta) > maxAnthropicReasoningBlockBytes ||
				reasoningBytes+len(delta) > maxAnthropicReasoningTotalBytes {
				invalidateReasoning(capture)
				return
			}
			if signature {
				capture.block.Signature += delta
			} else {
				capture.block.Text += delta
			}
			reasoningBytes += len(delta)
		}

		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(line[len("data:"):])
			if payload == "" {
				continue
			}
			if providerErr, ok := inBandProviderError(p.Name(), resp, []byte(payload)); ok {
				ch <- protocol.ChatDelta{Done: true, Err: providerErr, Usage: usage}
				return
			}
			var ev antStreamEvent
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				continue
			}
			switch ev.Type {
			case "message_start":
				if ev.Message != nil {
					// message_start carries the full input accounting, including the
					// cache-read/cache-write split; output accrues via message_delta.
					u := ev.Message.Usage.usage()
					usage.InputTokens = u.InputTokens
					usage.CacheReadTokens = u.CacheReadTokens
					usage.CacheWriteTokens = u.CacheWriteTokens
				}
			case "content_block_start":
				if ev.ContentBlock != nil {
					switch ev.ContentBlock.Type {
					case "tool_use":
						toolAcc[ev.Index] = &protocol.ToolCall{ID: ev.ContentBlock.ID, Name: ev.ContentBlock.Name}
						order = append(order, ev.Index)
					case "thinking":
						if len(reasoningOrder) >= maxAnthropicReasoningBlocks {
							continue
						}
						capture := &antReasoningCapture{block: protocol.ReasoningBlock{
							Type: protocol.ReasoningBlockThinking, Text: ev.ContentBlock.Thinking,
							Signature: ev.ContentBlock.Signature, ReplayScope: replayScope,
						}}
						reasoningAcc[ev.Index] = capture
						reasoningOrder = append(reasoningOrder, ev.Index)
						size := anthropicReasoningBlockBytes(capture.block)
						if size > maxAnthropicReasoningBlockBytes || reasoningBytes+size > maxAnthropicReasoningTotalBytes {
							capture.invalid = true
							capture.block = protocol.ReasoningBlock{}
						} else {
							reasoningBytes += size
						}
					case "redacted_thinking":
						if len(reasoningOrder) >= maxAnthropicReasoningBlocks {
							continue
						}
						capture := &antReasoningCapture{block: protocol.ReasoningBlock{
							Type: protocol.ReasoningBlockRedacted, Data: ev.ContentBlock.Data, ReplayScope: replayScope,
						}}
						reasoningAcc[ev.Index] = capture
						reasoningOrder = append(reasoningOrder, ev.Index)
						size := anthropicReasoningBlockBytes(capture.block)
						if size > maxAnthropicReasoningBlockBytes || reasoningBytes+size > maxAnthropicReasoningTotalBytes {
							capture.invalid = true
							capture.block = protocol.ReasoningBlock{}
						} else {
							reasoningBytes += size
						}
					}
				}
			case "content_block_delta":
				if ev.Delta == nil {
					continue
				}
				switch ev.Delta.Type {
				case "text_delta":
					if ev.Delta.Text != "" {
						ch <- protocol.ChatDelta{ContentDelta: ev.Delta.Text}
					}
				case "input_json_delta":
					if acc, ok := toolAcc[ev.Index]; ok {
						acc.Arguments += ev.Delta.PartialJSON
					}
				case "thinking_delta":
					if acc := reasoningAcc[ev.Index]; acc != nil && acc.block.Type == protocol.ReasoningBlockThinking {
						appendReasoning(acc, ev.Delta.Thinking, false)
					}
				case "signature_delta":
					if acc := reasoningAcc[ev.Index]; acc != nil && acc.block.Type == protocol.ReasoningBlockThinking {
						appendReasoning(acc, ev.Delta.Signature, true)
					}
				}
			case "content_block_stop":
				emitReasoning(ev.Index)
			case "message_delta":
				if ev.Delta != nil && ev.Delta.StopReason != "" {
					stopReason = ev.Delta.StopReason
				}
				if ev.Usage != nil {
					usage.OutputTokens = ev.Usage.OutputTokens
				}
			case "error":
				if ev.Error != nil {
					ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), resp, ev.Error.Message), Usage: usage}
					return
				}
			case "message_stop":
				terminal = true
			}
		}
		if err := sc.Err(); err != nil {
			ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), nil, "stream read failed: "+err.Error()), Usage: usage}
			return
		}
		if !terminal {
			ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), nil, "stream ended before a terminal event"), Usage: usage}
			return
		}
		for _, idx := range reasoningOrder {
			emitReasoning(idx)
		}
		for _, idx := range order {
			tc := *toolAcc[idx]
			tc.Arguments = string(ParseStreamingJSON(tc.Arguments))
			ch <- protocol.ChatDelta{ToolCall: &tc}
		}
		ch <- protocol.ChatDelta{Done: true, StopReason: stopReason, Usage: usage}
	}()
	return ch, nil
}

// encode maps the neutral protocol.ChatRequest onto the Anthropic wire format: system
// messages collapse into the top-level system field; tool calls/results become
// tool_use/tool_result content blocks.
func (p *AnthropicProvider) encode(req protocol.ChatRequest) antRequest {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = anthropicDefaultTokens
	}
	out := antRequest{Model: req.Model, MaxTokens: maxTokens}
	if req.OutputSchema != nil {
		out.OutputFormat = &antOutputFormat{Type: "json_schema", Schema: req.OutputSchema.Schema}
	}

	var systemParts []string
	var anchored []int // out.Messages indices whose source message carried CacheAnchor
	// anchorRequested distinguishes "the caller placed no anchors" (standalone
	// use — fall back to the moving breakpoint) from "the caller placed anchors
	// that could not be mapped here" (an anchor on a system message, which is
	// hoisted out of Messages). Without the distinction the fallback silently
	// overrides a deliberate placement and re-anchors the final message — which
	// is the very thing markCacheAnchors moved the breakpoint away from.
	anchorRequested := false
	allowImages := imageInputAllowed(p.ModelCapabilities(req.Model))
	replayScope := p.replayScope(req.Model)
	for _, m := range req.Messages {
		switch m.Role {
		case protocol.RoleSystem:
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}
		case protocol.RoleTool:
			text := messageText(m)
			images := messageImages(m)
			var content any = text
			if allowImages && len(images) > 0 {
				blocks := make([]antContentBlock, 0, len(images)+1)
				if text != "" {
					blocks = append(blocks, antContentBlock{Type: "text", Text: text})
				}
				for _, image := range images {
					blocks = append(blocks, antContentBlock{Type: "image", Source: &antImageSource{
						Type: "base64", MediaType: image.MIMEType, Data: image.Data,
					}})
				}
				content = blocks
			} else if len(images) > 0 {
				content = textWithImageNotice(text, len(images), false)
			}
			out.Messages = append(out.Messages, antMessage{
				Role: "user",
				Content: []antContentBlock{{
					Type: "tool_result", ToolUseID: m.ToolCallID, Content: content,
				}},
			})
		case protocol.RoleAssistant:
			blocks := []antContentBlock{}
			reasoningBytes := 0
			reasoningCount := 0
			for _, reasoning := range m.ReasoningBlocks {
				size := anthropicReasoningBlockBytes(reasoning)
				if reasoning.ReplayScope != replayScope ||
					!validAnthropicReasoningBlock(reasoning) ||
					reasoningCount >= maxAnthropicReasoningBlocks ||
					reasoningBytes+size > maxAnthropicReasoningTotalBytes {
					continue
				}
				switch reasoning.Type {
				case protocol.ReasoningBlockThinking:
					blocks = append(blocks, antContentBlock{
						Type: "thinking", Thinking: reasoning.Text, Signature: reasoning.Signature,
					})
				case protocol.ReasoningBlockRedacted:
					blocks = append(blocks, antContentBlock{Type: "redacted_thinking", Data: reasoning.Data})
				}
				reasoningBytes += size
				reasoningCount++
			}
			if m.Content != "" {
				blocks = append(blocks, antContentBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(tc.Arguments)
				if len(input) == 0 {
					input = json.RawMessage("{}")
				}
				blocks = append(blocks, antContentBlock{Type: "tool_use", ID: tc.ID, Name: tc.Name, Input: input})
			}
			if len(blocks) == 0 {
				blocks = append(blocks, antContentBlock{Type: "text", Text: ""})
			}
			out.Messages = append(out.Messages, antMessage{Role: "assistant", Content: blocks})
		default: // user
			blocks := []antContentBlock{}
			text := messageText(m)
			images := messageImages(m)
			if !allowImages && len(images) > 0 {
				text = textWithImageNotice(text, len(images), false)
			}
			if text != "" || len(images) == 0 {
				blocks = append(blocks, antContentBlock{Type: "text", Text: text})
			}
			if allowImages {
				for _, image := range images {
					blocks = append(blocks, antContentBlock{Type: "image", Source: &antImageSource{
						Type: "base64", MediaType: image.MIMEType, Data: image.Data,
					}})
				}
			}
			out.Messages = append(out.Messages, antMessage{
				Role: "user", Content: blocks,
			})
		}
		// System messages don't reach out.Messages (hoisted into the system
		// block below), so an anchor on one has nothing to attach to.
		if m.CacheAnchor {
			anchorRequested = true
			if m.Role != protocol.RoleSystem && len(out.Messages) > 0 {
				anchored = append(anchored, len(out.Messages)-1)
			}
		}
	}
	// The system prompt is the largest stable prefix of a long run. When caching is
	// requested, send it as a structured block carrying cache_control so Anthropic
	// reuses it across turns; otherwise keep the bare-string form the API has always
	// received. An empty system stays omitted either way.
	//
	// OAuth (claude-code) mode always uses the structured form: the Claude Code
	// identity block leads, the caller's system text follows, and the cache
	// breakpoint — when requested — lands on the last block.
	if p.OAuth {
		blocks := []antSystemBlock{{Type: "text", Text: claudeCodeSystemInstruction}}
		if systemText := strings.Join(systemParts, "\n\n"); systemText != "" {
			blocks = append(blocks, antSystemBlock{Type: "text", Text: systemText})
		}
		if req.CacheKey != "" {
			blocks[len(blocks)-1].CacheControl = &antCacheControl{Type: "ephemeral", TTL: antCacheTTL(req.CacheRetention)}
		}
		out.System = blocks
	} else if systemText := strings.Join(systemParts, "\n\n"); systemText != "" {
		if req.CacheKey != "" {
			out.System = []antSystemBlock{{
				Type:         "text",
				Text:         systemText,
				CacheControl: &antCacheControl{Type: "ephemeral", TTL: antCacheTTL(req.CacheRetention)},
			}}
		} else {
			out.System = systemText
		}
	}

	// Anchor-driven breakpoints: the loop decides placement (protocol.Message.CacheAnchor,
	// see markCacheAnchors); this provider only translates each anchor into
	// cache_control on that message's last block. With no anchors REQUESTED at all
	// (the provider used standalone, outside the loop) fall back to the classic
	// moving breakpoint on the final message, so the whole transcript-so-far
	// becomes the cached prefix: next turn re-sends the same transcript plus one
	// exchange and Anthropic prefix-matches the previous breakpoint, billing
	// everything before it as a cache read.
	//
	// The fallback is deliberately NOT taken when anchors were requested but none
	// landed here (an anchor on a system message, hoisted into the system block
	// above). Falling back there would re-anchor the final message and undo the
	// loop's placement — and the loop moves the breakpoint off the final message
	// precisely because a ContextHook's regenerated trailer sits there.
	//
	// Anthropic allows at most 4 breakpoints per request; one is spent on the
	// system block, so only the last 3 anchors are honored.
	if req.CacheKey != "" && len(out.Messages) > 0 {
		if len(anchored) == 0 && !anchorRequested {
			anchored = []int{len(out.Messages) - 1}
		}
		if len(anchored) > 3 {
			anchored = anchored[len(anchored)-3:]
		}
		for _, idx := range anchored {
			msg := &out.Messages[idx]
			if n := len(msg.Content); n > 0 {
				msg.Content[n-1].CacheControl = &antCacheControl{Type: "ephemeral", TTL: antCacheTTL(req.CacheRetention)}
			}
		}
	}

	for _, s := range req.Tools {
		out.Tools = append(out.Tools, antTool{
			Name: s.Name, Description: s.Description,
			InputSchema: toolParameters(s.Parameters, toolSchemaGeneric),
		})
	}
	out.ToolChoice = anthropicToolChoice(req.ToolChoice, req.ParallelToolCalls, len(out.Tools) > 0)
	return out
}

// listClaudeCodeModels calls GET {base}/v1/models with the OAuth account's
// Bearer credential — the same endpoint as listAnthropicModels, but the
// subscription surface authenticates with the token, not x-api-key.
func listClaudeCodeModels(ctx context.Context, client HTTPDoer, baseURL string, tok OAuthToken) ([]Model, error) {
	listed, err := listModelsV1(ctx, client, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
		req.Header.Set("User-Agent", claudeCodeUserAgent)
		req.Header.Set("x-app", "cli")
	})
	if err != nil {
		return nil, err
	}
	out := make([]Model, 0, len(listed))
	for _, m := range listed {
		out = append(out, Model{ID: m.ID, ContextWindow: m.ContextWindow})
	}
	return out, nil
}
