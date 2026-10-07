package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

const (
	defaultOpenAIBaseURL    = "https://api.openai.com/v1"
	defaultOpenAIEmbedModel = "text-embedding-3-small"
)

// Compat is the per-vendor capability table. Most "providers" are the OpenAI
// completions API at a different base URL plus a few flags (pi-ai pattern), so
// an OpenAI-compatible vendor is config, not new code.
type Compat struct {
	// MaxTokensField is "max_tokens" or "max_completion_tokens".
	MaxTokensField string
	// SupportsTools indicates the vendor accepts the tools/tool_calls fields.
	SupportsTools bool
}

// DefaultCompat is the stock OpenAI behavior.
func DefaultCompat() Compat {
	return Compat{MaxTokensField: "max_tokens", SupportsTools: true}
}

// OpenAIProvider speaks the OpenAI chat-completions wire format. base_url
// resolution (per-config -> OPENAI_BASE_URL env -> vendor default) is performed
// by the caller; this struct receives the resolved BaseURL.
type OpenAIProvider struct {
	APIKey  string
	keyMu   sync.RWMutex
	BaseURL string
	Compat  Compat
	// HTTP serves the non-streamed Chat path (absolute cap, no header deadline —
	// a buffered completion's headers arrive only once generation is done).
	HTTP *http.Client
	// StreamHTTP serves the SSE path, where a header deadline is meaningful. Nil
	// falls back to HTTP, so a caller that overrides only HTTP still works.
	StreamHTTP *http.Client
	// Vendor, when set, overrides Name() for OpenAI-compatible vendors served
	// through this wire (the pi-ai pattern: most providers are this API at a
	// different base URL). Empty keeps the stock "openai" identity.
	Vendor string
}

// NewOpenAIProvider builds a provider. An empty baseURL falls back to the
// vendor default; an empty Compat falls back to DefaultCompat.
func NewOpenAIProvider(apiKey, baseURL string, compat Compat) *OpenAIProvider {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = defaultOpenAIBaseURL
	}
	if compat.MaxTokensField == "" {
		compat = DefaultCompat()
	}
	return &OpenAIProvider{
		APIKey:     apiKey,
		BaseURL:    strings.TrimRight(baseURL, "/"),
		Compat:     compat,
		HTTP:       NewChatHTTPClient(0),
		StreamHTTP: NewStreamHTTPClient(0),
	}
}

// streamHTTP is the client the SSE path uses: StreamHTTP when set, otherwise
// whatever the caller put on HTTP.
func (p *OpenAIProvider) streamHTTP() *http.Client {
	if p.StreamHTTP != nil {
		return p.StreamHTTP
	}
	return p.HTTP
}

func (p *OpenAIProvider) Name() string {
	if p.Vendor != "" {
		return p.Vendor
	}
	return "openai"
}
func (p *OpenAIProvider) SupportsTools() bool { return p.Compat.SupportsTools }

func (p *OpenAIProvider) ModelCapabilities(model string) protocol.ModelCapabilities {
	caps := CapabilitiesFor(p.Name(), model)
	// Compat is the authoritative wire-level tool flag for arbitrary
	// OpenAI-compatible endpoints, including an explicit false.
	caps.Tools = support(p.Compat.SupportsTools)
	return caps
}

// defaultGeminiBaseURL is Google's OpenAI-compatible surface for the Gemini
// API (chat completions + tools + streaming on the OpenAI wire).
const defaultGeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"

// NewGeminiProvider builds a Google Gemini provider on the OpenAI-compatible
// wire — provider breadth as config, not new code (Compat table pattern). Use
// Gemini model ids ("gemini-2.5-pro", "gemini-2.5-flash"); tools, streaming,
// and usage accounting ride the shared implementation, and Name() reports
// "google" so traces, tiers, and RefreshKey attribute to the right vendor.
func NewGeminiProvider(apiKey string) *OpenAIProvider {
	p := NewOpenAIProvider(apiKey, defaultGeminiBaseURL, DefaultCompat())
	p.Vendor = "google"
	return p
}

// UpdateAPIKey swaps the key used for subsequent requests (protocol.KeyUpdater), letting
// the loop refresh an expiring BYO token between turns.
func (p *OpenAIProvider) UpdateAPIKey(key string) {
	if key != "" {
		p.keyMu.Lock()
		p.APIKey = key
		p.keyMu.Unlock()
	}
}

func (p *OpenAIProvider) apiKey() string {
	p.keyMu.RLock()
	defer p.keyMu.RUnlock()
	return p.APIKey
}

// --- wire types (OpenAI chat-completions) ---

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content,omitempty"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

type oaiContentPart struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *oaiImageURL `json:"image_url,omitempty"`
}

type oaiImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type oaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
		Strict      *bool          `json:"strict,omitempty"`
	} `json:"function"`
}

type oaiRequest struct {
	Model               string            `json:"model"`
	Messages            []oaiMessage      `json:"messages"`
	Tools               []oaiTool         `json:"tools,omitempty"`
	MaxTokens           int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens int               `json:"max_completion_tokens,omitempty"`
	Stream              bool              `json:"stream,omitempty"`
	StreamOptions       *oaiStreamOptions `json:"stream_options,omitempty"`
	// PromptCacheKey routes this call to a shared cached prefix (OpenAI prompt
	// caching). Sent only when the neutral request opts in, so strict
	// OpenAI-compatible servers that reject unknown fields are never sent it.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
	// ReasoningEffort asks a reasoning model for that much thinking per turn
	// ("low" | "medium" | "high"). Sent only when the neutral request sets it.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// ResponseFormat constrains the answer to a JSON Schema (structured
	// outputs). Sent only when the neutral request carries an protocol.OutputSchema, so
	// strict compat servers are otherwise unaffected.
	ResponseFormat    *oaiResponseFormat `json:"response_format,omitempty"`
	ToolChoice        any                `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool              `json:"parallel_tool_calls,omitempty"`
}

// oaiResponseFormat is OpenAI's structured-output selector; the json_schema
// form is emitted, strict only when the caller opted in (see protocol.OutputSchema.Strict).
type oaiResponseFormat struct {
	Type       string        `json:"type"`
	JSONSchema oaiJSONSchema `json:"json_schema"`
}

type oaiJSONSchema struct {
	Name   string         `json:"name"`
	Strict bool           `json:"strict"`
	Schema map[string]any `json:"schema"`
}

type oaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaiResponse struct {
	Choices []struct {
		Message      oaiResponseMessage `json:"message"`
		FinishReason string             `json:"finish_reason"`
	} `json:"choices"`
	Usage oaiUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type oaiResponseMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	ToolCalls []oaiToolCall `json:"tool_calls,omitempty"`
}

// oaiUsage mirrors OpenAI's usage block, including the cached-token detail.
// prompt_tokens is the *total* input (cached + fresh); cached_tokens is the
// portion served from the prompt cache. usage maps these onto the neutral protocol.Usage
// so InputTokens stays full-price-only.
type oaiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// usage normalizes an OpenAI usage block: cached tokens are pulled out of the
// full prompt total so InputTokens counts only fresh, full-price input.
func (u oaiUsage) usage() protocol.Usage {
	cached := u.PromptTokensDetails.CachedTokens
	if cached > u.PromptTokens {
		cached = u.PromptTokens
	}
	return protocol.Usage{
		InputTokens:     u.PromptTokens - cached,
		OutputTokens:    u.CompletionTokens,
		CacheReadTokens: cached,
	}
}

// Chat performs one non-streaming completion. OpenAI-compatible endpoints are
// inconsistent about optional request hints; when one explicitly rejects a
// hint, the adapter removes only that hint and retains the endpoint lesson in
// the logical provider session.
func (p *OpenAIProvider) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	state, active, maxTokenField := p.adaptiveRequest(req)
	strictState, active := prepareStrictTools(active, p.Name(), p.BaseURL, toolSchemaGeneric)
	var learned protocol.ModelCapabilities
	switchedMaxTokenField := false
	retriedWithoutStrict := false
	for {
		resp, err := p.chatOnce(ctx, active, maxTokenField)
		if err == nil {
			if state != nil {
				state.remember(active.Model, learned)
				state.rememberMaxTokenField(active.Model, maxTokenField)
			}
			return resp, nil
		}
		if !switchedMaxTokenField {
			if alternate, ok := alternateMaxTokenField(maxTokenField, active, err); ok {
				maxTokenField = alternate
				switchedMaxTokenField = true
				continue
			}
		}
		if !retriedWithoutStrict && shouldRetryWithoutStrictTools(active, toolSchemaGeneric, err) {
			rememberStrictToolsRejected(strictState, active.Model)
			active = withoutStrictTools(active)
			retriedWithoutStrict = true
			continue
		}
		next, demotion, ok := withoutRejectedOpenAIHint(active, err)
		if !ok {
			return protocol.ChatResponse{}, err
		}
		active = next
		learned = learned.Overlay(demotion)
	}
}

func (p *OpenAIProvider) chatOnce(ctx context.Context, req protocol.ChatRequest, maxTokenField string) (protocol.ChatResponse, error) {
	body := p.encodeWithMaxTokenField(req, maxTokenField)
	raw, err := json.Marshal(body)
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return protocol.ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	setOptionalBearerAuth(httpReq, p.apiKey())

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

	// A body that stopped mid-flight is a transport failure wearing a 200: the
	// status line was written before the stream broke, so the status tells us
	// nothing. Discarding this error let a severed stream decode as if complete —
	// a truncated answer became a "successful" short one. Report it as transport
	// (nil resp => Status 0 => IsRetryable) so the ladder retries instead.
	// On a 4xx/5xx the status is the more useful signal, so fall through and let
	// the classification below run on whatever body arrived.
	if readErr != nil && resp.StatusCode < 400 {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), nil,
			fmt.Sprintf("truncated response body (status %d): %s", resp.StatusCode, describeReadErr(ctx, p.HTTP, readErr)))
	}

	// Some OpenAI-compatible gateways (e.g. 9router routing summarizer calls to a
	// cheap streamed model) reply with text/event-stream even when stream=false
	// was requested. Detect that and fold the SSE chunks into a single response
	// rather than failing the JSON decode and degrading the caller.
	if isSSEResponse(resp.Header.Get("Content-Type"), data) {
		return decodeSSEResponse(p, resp, data)
	}

	// Classify by HTTP status BEFORE decoding the body. A 5xx/429 from a gateway
	// often carries a non-JSON body (an HTML/text error page); decoding that first
	// returns a plain decode error that protocol.IsRetryable can't see as an protocol.ProviderError,
	// which turns a transient, retryable outage into a permanent failure that
	// drops the turn. The streaming path already checks status first — mirror it.
	// When the error body happens to be JSON with a structured message, surface it;
	// otherwise fall back to the raw body text.
	if resp.StatusCode >= 400 {
		message := strings.TrimSpace(string(data))
		var errBody oaiResponse
		if json.Unmarshal(data, &errBody) == nil && errBody.Error != nil && errBody.Error.Message != "" {
			message = errBody.Error.Message
		}
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), resp, message)
	}
	if providerErr, ok := inBandProviderError(p.Name(), resp, data); ok {
		return protocol.ChatResponse{}, providerErr
	}

	var decoded oaiResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		return protocol.ChatResponse{}, fmt.Errorf("decode openai response (status %d): %w", resp.StatusCode, err)
	}
	// A 2xx body can still carry a structured error (some compat gateways do this).
	if decoded.Error != nil {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), resp, decoded.Error.Message)
	}
	if len(decoded.Choices) == 0 {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), resp, "unexpected response")
	}

	choice := decoded.Choices[0]
	msg := protocol.Message{Role: protocol.RoleAssistant, Content: choice.Message.Content}
	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, protocol.ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Arguments: string(ParseStreamingJSON(tc.Function.Arguments)),
		})
	}
	return protocol.ChatResponse{
		Message:    msg,
		StopReason: choice.FinishReason,
		Usage:      decoded.Usage.usage(),
	}, nil
}

// isSSEResponse reports whether a (nominally non-streaming) completion reply is
// actually an SSE stream: either the content-type advertises it or the body
// begins with an SSE "data:" frame.
func isSSEResponse(contentType string, data []byte) bool {
	if strings.Contains(strings.ToLower(contentType), "text/event-stream") {
		return true
	}
	return strings.HasPrefix(strings.TrimSpace(string(data)), "data:")
}

// decodeSSEResponse folds an SSE chat-completions stream (already buffered into
// data) into a single ChatResponse: concatenating content deltas, reassembling
// tool-call fragments keyed by index, and keeping the last finish_reason/usage.
func decodeSSEResponse(p *OpenAIProvider, resp *http.Response, data []byte) (protocol.ChatResponse, error) {
	toolAcc := map[int]*protocol.ToolCall{}
	var order []int
	var content strings.Builder
	var stopReason string
	var usage protocol.Usage
	var gotChunk bool

	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "" || payload == "[DONE]" {
			continue
		}
		if providerErr, ok := inBandProviderError(p.Name(), resp, []byte(payload)); ok {
			return protocol.ChatResponse{}, providerErr
		}
		var chunk oaiStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // tolerate keep-alive/comment lines
		}
		gotChunk = true
		if chunk.Usage != nil {
			usage = chunk.Usage.usage()
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		content.WriteString(choice.Delta.Content)
		for _, tc := range choice.Delta.ToolCalls {
			acc, ok := toolAcc[tc.Index]
			if !ok {
				acc = &protocol.ToolCall{}
				toolAcc[tc.Index] = acc
				order = append(order, tc.Index)
			}
			if tc.ID != "" {
				acc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				acc.Name = tc.Function.Name
			}
			acc.Arguments += tc.Function.Arguments
		}
		if choice.FinishReason != "" {
			stopReason = choice.FinishReason
		}
	}
	// Both guards below pass a nil *http.Response on purpose. A gateway that
	// streams writes its 200 before generating anything, so the status describes
	// the handshake, not the outcome; stamping 200 onto the error would make
	// IsRetryable read a transport failure as a permanent one and drop the turn.
	if !gotChunk {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), nil, "empty SSE response")
	}

	// A stream that delivered no finish_reason AND nothing usable never ended — it
	// was severed after the opening role frame. Left alone this folds into a
	// perfectly well-formed empty turn, and the loop stops at turn 1 as if the
	// model had chosen to say nothing. gotChunk can't catch it: the role frame is
	// itself a valid chunk. Requiring all three to be absent keeps gateways that
	// simply omit finish_reason working, since those still return real output.
	if stopReason == "" && content.Len() == 0 && len(order) == 0 {
		return protocol.ChatResponse{}, protocol.NewProviderError(p.Name(), nil,
			"truncated SSE response: no content, tool calls, or finish_reason")
	}

	msg := protocol.Message{Role: protocol.RoleAssistant, Content: content.String()}
	for _, idx := range order {
		call := *toolAcc[idx]
		call.Arguments = string(ParseStreamingJSON(call.Arguments))
		msg.ToolCalls = append(msg.ToolCalls, call)
	}
	return protocol.ChatResponse{Message: msg, StopReason: stopReason, Usage: usage}, nil
}

// --- streaming wire types (OpenAI chat-completions, stream=true) ---

type oaiStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oaiUsage `json:"usage"`
}

// Stream performs a real token-streaming completion: it sets stream=true and
// emits one protocol.ChatDelta per server-sent content chunk, accumulating fragmented
// tool-call deltas (which arrive piecewise, keyed by index) into whole ToolCalls
// flushed before the terminal Done delta. The loop forwards content deltas to a
// live SSE sink; tool execution is unchanged.
func (p *OpenAIProvider) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	state, active, maxTokenField := p.adaptiveRequest(req)
	strictState, active := prepareStrictTools(active, p.Name(), p.BaseURL, toolSchemaGeneric)
	var learned protocol.ModelCapabilities
	switchedMaxTokenField := false
	retriedWithoutStrict := false
	for {
		ch, err := p.streamOnce(ctx, active, maxTokenField)
		if err == nil {
			var retryStrict bool
			ch, err, retryStrict = preflightStrictStream(ctx, active, toolSchemaGeneric, ch)
			if retryStrict {
				rememberStrictToolsRejected(strictState, active.Model)
				active = withoutStrictTools(active)
				retriedWithoutStrict = true
				continue
			}
			if err != nil {
				return nil, err
			}
			if state != nil {
				state.remember(active.Model, learned)
				state.rememberMaxTokenField(active.Model, maxTokenField)
			}
			return ch, nil
		}
		if !switchedMaxTokenField {
			if alternate, ok := alternateMaxTokenField(maxTokenField, active, err); ok {
				maxTokenField = alternate
				switchedMaxTokenField = true
				continue
			}
		}
		if !retriedWithoutStrict && shouldRetryWithoutStrictTools(active, toolSchemaGeneric, err) {
			rememberStrictToolsRejected(strictState, active.Model)
			active = withoutStrictTools(active)
			retriedWithoutStrict = true
			continue
		}
		next, demotion, ok := withoutRejectedOpenAIHint(active, err)
		if !ok {
			return nil, err
		}
		active = next
		learned = learned.Overlay(demotion)
	}
}

func (p *OpenAIProvider) streamOnce(ctx context.Context, req protocol.ChatRequest, maxTokenField string) (<-chan protocol.ChatDelta, error) {
	body := p.encodeWithMaxTokenField(req, maxTokenField)
	body.Stream = true
	body.StreamOptions = &oaiStreamOptions{IncludeUsage: true}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	setOptionalBearerAuth(httpReq, p.apiKey())

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

		// Tool-call fragments arrive across chunks keyed by index; accumulate
		// them in order and flush whole calls once the stream completes.
		toolAcc := map[int]*protocol.ToolCall{}
		var order []int
		var stopReason string
		var usage protocol.Usage
		terminal := false

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
			if payload == "[DONE]" {
				terminal = true
				break
			}
			if providerErr, ok := inBandProviderError(p.Name(), resp, []byte(payload)); ok {
				ch <- protocol.ChatDelta{Done: true, Err: providerErr, Usage: usage}
				return
			}
			var chunk oaiStreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue // tolerate keep-alive/comment lines
			}
			if chunk.Usage != nil {
				usage = chunk.Usage.usage()
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.Delta.Content != "" {
				ch <- protocol.ChatDelta{ContentDelta: choice.Delta.Content}
			}
			for _, tc := range choice.Delta.ToolCalls {
				acc, ok := toolAcc[tc.Index]
				if !ok {
					acc = &protocol.ToolCall{}
					toolAcc[tc.Index] = acc
					order = append(order, tc.Index)
				}
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				acc.Arguments += tc.Function.Arguments
			}
			if choice.FinishReason != "" {
				stopReason = choice.FinishReason
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
		for _, idx := range order {
			tc := *toolAcc[idx]
			tc.Arguments = string(ParseStreamingJSON(tc.Arguments))
			ch <- protocol.ChatDelta{ToolCall: &tc}
		}
		ch <- protocol.ChatDelta{Done: true, StopReason: stopReason, Usage: usage}
	}()
	return ch, nil
}

// --- embeddings (semantic memory recall, §14.7) ---

// OpenAIEmbedder calls the OpenAI /embeddings endpoint, reusing the project's
// BYO key and base_url. It backs the protocol.Embedder seam so semantic recall
// works for the OpenAI-wire provider family (OpenAI + compatible vendors) with
// no extra credentials.
type OpenAIEmbedder struct {
	APIKey  string
	BaseURL string
	Model   string
	HTTP    *http.Client
}

// NewOpenAIEmbedder builds an embedder. Empty baseURL/model fall back to the
// OpenAI defaults.
func NewOpenAIEmbedder(apiKey, baseURL, model string) *OpenAIEmbedder {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = defaultOpenAIBaseURL
	}
	if strings.TrimSpace(model) == "" {
		model = defaultOpenAIEmbedModel
	}
	return &OpenAIEmbedder{
		APIKey:  apiKey,
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		HTTP:    &http.Client{Timeout: 60 * time.Second},
	}
}

type oaiEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type oaiEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Embed returns one vector per input string, index-aligned.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	raw, err := json.Marshal(oaiEmbedRequest{Model: e.Model, Input: texts})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+"/embeddings", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	setOptionalBearerAuth(httpReq, e.APIKey)

	resp, err := e.HTTP.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var data []byte
	var readErr error
	if resp.StatusCode >= 400 {
		data = readProviderErrorBody(resp.Body)
	} else {
		data, readErr = io.ReadAll(resp.Body)
	}
	// Same trap as the chat path: a severed body would otherwise surface as a
	// plain JSON decode error, which IsRetryable can't see, turning a transient
	// truncation into a permanent embeddings failure.
	if readErr != nil && resp.StatusCode < 400 {
		return nil, protocol.NewProviderError("openai-embeddings", nil,
			fmt.Sprintf("truncated embeddings body (status %d): %s", resp.StatusCode, describeReadErr(ctx, e.HTTP, readErr)))
	}

	if resp.StatusCode >= 400 {
		message := strings.TrimSpace(string(data))
		var errorBody oaiEmbedResponse
		if json.Unmarshal(data, &errorBody) == nil && errorBody.Error != nil && errorBody.Error.Message != "" {
			message = errorBody.Error.Message
		}
		return nil, protocol.NewProviderError("openai-embeddings", resp, message)
	}
	if providerErr, ok := inBandProviderError("openai-embeddings", resp, data); ok {
		return nil, providerErr
	}

	var decoded oaiEmbedResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("decode embeddings response (status %d): %w", resp.StatusCode, err)
	}
	out := make([][]float32, len(texts))
	for _, d := range decoded.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	return out, nil
}

// encode maps the neutral protocol.ChatRequest onto the OpenAI wire format, honoring the
// compat table.
func (p *OpenAIProvider) encode(req protocol.ChatRequest) oaiRequest {
	return p.encodeWithMaxTokenField(req, p.maxTokensField())
}

func (p *OpenAIProvider) maxTokensField() string {
	if p.Compat.MaxTokensField == "max_completion_tokens" {
		return "max_completion_tokens"
	}
	return "max_tokens"
}

func (p *OpenAIProvider) encodeWithMaxTokenField(req protocol.ChatRequest, maxTokenField string) oaiRequest {
	out := oaiRequest{
		Model: req.Model, PromptCacheKey: req.CacheKey, ReasoningEffort: req.ReasoningEffort,
		ToolChoice:        openAIChatToolChoice(req.ToolChoice, len(req.Tools) > 0),
		ParallelToolCalls: cloneOptionalBool(req.ParallelToolCalls, len(req.Tools) > 0),
	}
	if maxTokenField == "max_completion_tokens" {
		out.MaxCompletionTokens = req.MaxTokens
	} else {
		out.MaxTokens = req.MaxTokens
	}
	if req.OutputSchema != nil {
		name := req.OutputSchema.Name
		if name == "" {
			name = "output"
		}
		// Strict is caller-opt-in: OpenAI's strict mode 400s on any schema
		// outside its subset (draft-07 features like oneOf/format, optional
		// properties), so hardcoding it would turn a valid schema into a fatal
		// error every turn instead of the documented soft-degrade.
		out.ResponseFormat = &oaiResponseFormat{
			Type:       "json_schema",
			JSONSchema: oaiJSONSchema{Name: name, Strict: req.OutputSchema.Strict, Schema: req.OutputSchema.Schema},
		}
	}
	allowImages := imageInputAllowed(p.ModelCapabilities(req.Model))
	for i := 0; i < len(req.Messages); i++ {
		m := req.Messages[i]
		// Chat Completions tool messages are text-only. Preserve tool-call/result
		// adjacency by batching every consecutive result first, then hoist their
		// images into one user message (the same shape OMP uses).
		if m.Role == protocol.RoleTool {
			var attached []oaiContentPart
			for ; i < len(req.Messages) && req.Messages[i].Role == protocol.RoleTool; i++ {
				tm := req.Messages[i]
				images := messageImages(tm)
				text := messageText(tm)
				if allowImages {
					text = textWithImageNotice(text, len(images), len(images) > 0)
					for _, image := range images {
						attached = append(attached, oaiContentPart{Type: "image_url", ImageURL: &oaiImageURL{
							URL: imageDataURL(image), Detail: normalizedImageDetail(image.Detail, false),
						}})
					}
				} else {
					text = textWithImageNotice(text, len(images), false)
				}
				out.Messages = append(out.Messages, oaiMessage{
					Role: string(tm.Role), Content: text, ToolCallID: tm.ToolCallID, Name: tm.Name,
				})
			}
			i--
			if len(attached) > 0 {
				attached = append([]oaiContentPart{{Type: "text", Text: "Images from the preceding tool results."}}, attached...)
				out.Messages = append(out.Messages, oaiMessage{Role: "user", Content: attached})
			}
			continue
		}
		content := any(messageText(m))
		if images := messageImages(m); len(images) > 0 {
			if allowImages && (m.Role == protocol.RoleUser) {
				parts := []oaiContentPart{}
				if text := messageText(m); text != "" {
					parts = append(parts, oaiContentPart{Type: "text", Text: text})
				}
				for _, image := range images {
					parts = append(parts, oaiContentPart{Type: "image_url", ImageURL: &oaiImageURL{
						URL: imageDataURL(image), Detail: normalizedImageDetail(image.Detail, false),
					}})
				}
				content = parts
			} else {
				content = textWithImageNotice(messageText(m), len(images), false)
			}
		}
		om := oaiMessage{Role: string(m.Role), Content: content, ToolCallID: m.ToolCallID, Name: m.Name}
		for _, tc := range m.ToolCalls {
			var otc oaiToolCall
			otc.ID = tc.ID
			otc.Type = "function"
			otc.Function.Name = tc.Name
			otc.Function.Arguments = tc.Arguments
			om.ToolCalls = append(om.ToolCalls, otc)
		}
		out.Messages = append(out.Messages, om)
	}
	if p.Compat.SupportsTools {
		for _, s := range req.Tools {
			var ot oaiTool
			ot.Type = "function"
			ot.Function.Name = s.Name
			ot.Function.Description = s.Description
			ot.Function.Parameters, ot.Function.Strict = projectedToolParameters(s, toolSchemaGeneric)
			out.Tools = append(out.Tools, ot)
		}
	}
	return out
}
