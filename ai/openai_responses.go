package ai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"sync"

	"github.com/2found/2ai/ai/protocol"
)

const (
	VendorOpenAIResponses = "openai-responses"
	responsesStatePrefix  = "openai-responses\x00"
	responsesStaleLimit   = 3
)

// OpenAIResponsesProvider speaks the public OpenAI Responses API. It is an
// explicit provider rather than a silent replacement for OpenAI Chat
// Completions: Responses can retain server-side conversation state, and that
// privacy/compatibility choice belongs to the workspace operator.
//
// When agentcore supplies a logical provider session, successful turns are
// chained with previous_response_id after an exact wire-prefix check. Without
// that state (or after an eviction) the adapter sends the complete transcript,
// so correctness never depends on process affinity.
type OpenAIResponsesProvider struct {
	APIKey     string
	keyMu      sync.RWMutex
	BaseURL    string
	HTTP       *http.Client
	StreamHTTP *http.Client
	// Vendor overrides Name() when this Responses wire is selected for a model
	// belonging to an ordinary OpenAI provider row. Empty preserves the explicit
	// openai-responses identity.
	Vendor string
}

// NewOpenAIResponsesProvider builds the public Responses wire client. baseURL
// is the API root (normally https://api.openai.com/v1); /responses is appended
// unless the caller already supplied the endpoint itself.
func NewOpenAIResponsesProvider(apiKey, baseURL string) *OpenAIResponsesProvider {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = defaultOpenAIBaseURL
	}
	return &OpenAIResponsesProvider{
		APIKey:     apiKey,
		BaseURL:    strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		HTTP:       NewChatHTTPClient(0),
		StreamHTTP: NewStreamHTTPClient(0),
	}
}

func (p *OpenAIResponsesProvider) Name() string {
	if p.Vendor != "" {
		return p.Vendor
	}
	return VendorOpenAIResponses
}
func (p *OpenAIResponsesProvider) SupportsTools() bool { return true }
func (p *OpenAIResponsesProvider) ModelCapabilities(model string) protocol.ModelCapabilities {
	// Capabilities describe the selected wire, not the credential/provider
	// identity returned by Name(). An identity-preserving OpenAI Responses
	// client must still advertise stateful chaining.
	return CapabilitiesFor(VendorOpenAIResponses, model)
}
func (p *OpenAIResponsesProvider) UpdateAPIKey(key string) {
	if key != "" {
		p.keyMu.Lock()
		p.APIKey = key
		p.keyMu.Unlock()
	}
}

func (p *OpenAIResponsesProvider) apiKey() string {
	p.keyMu.RLock()
	defer p.keyMu.RUnlock()
	return p.APIKey
}

func (p *OpenAIResponsesProvider) streamHTTP() *http.Client {
	if p.StreamHTTP != nil {
		return p.StreamHTTP
	}
	return p.HTTP
}

func (p *OpenAIResponsesProvider) responsesURL() string {
	base := strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
	if base == "" {
		base = defaultOpenAIBaseURL
	}
	if strings.HasSuffix(base, "/responses") {
		return base
	}
	return base + "/responses"
}

// --- Responses wire ---

type responsesInputItem struct {
	Type      string             `json:"type,omitempty"`
	Role      string             `json:"role,omitempty"`
	Content   []responsesContent `json:"content,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Output    any                `json:"output,omitempty"`
}

type responsesContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Refusal  string `json:"refusal,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

type responsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort"`
}

type responsesText struct {
	Format responsesTextFormat `json:"format"`
}

type responsesTextFormat struct {
	Type   string         `json:"type"`
	Name   string         `json:"name,omitempty"`
	Strict bool           `json:"strict,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
}

type responsesRequest struct {
	Model              string               `json:"model"`
	Instructions       string               `json:"instructions,omitempty"`
	Input              []responsesInputItem `json:"input"`
	Stream             bool                 `json:"stream"`
	Store              bool                 `json:"store"`
	Tools              []responsesTool      `json:"tools,omitempty"`
	MaxOutputTokens    int                  `json:"max_output_tokens,omitempty"`
	Temperature        float64              `json:"temperature,omitempty"`
	PromptCacheKey     string               `json:"prompt_cache_key,omitempty"`
	Reasoning          *responsesReasoning  `json:"reasoning,omitempty"`
	Text               *responsesText       `json:"text,omitempty"`
	PreviousResponseID string               `json:"previous_response_id,omitempty"`
	ToolChoice         any                  `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool                `json:"parallel_tool_calls,omitempty"`
}

// controls is the exact top-level request identity used by the chain check.
// Input and previous_response_id are deliberately absent; every other wire
// option must remain equal before a suffix can be appended.
type responsesControls struct {
	Model             string
	Instructions      string
	Stream            bool
	Store             bool
	Tools             []responsesTool
	MaxOutputTokens   int
	Temperature       float64
	PromptCacheKey    string
	Reasoning         *responsesReasoning
	Text              *responsesText
	ToolChoice        any
	ParallelToolCalls *bool
}

func (r responsesRequest) controls() responsesControls {
	return responsesControls{
		Model: r.Model, Instructions: r.Instructions, Stream: r.Stream,
		Store: r.Store, Tools: r.Tools, MaxOutputTokens: r.MaxOutputTokens,
		Temperature: r.Temperature, PromptCacheKey: r.PromptCacheKey,
		Reasoning: r.Reasoning, Text: r.Text,
		ToolChoice: r.ToolChoice, ParallelToolCalls: r.ParallelToolCalls,
	}
}

func (p *OpenAIResponsesProvider) encode(req protocol.ChatRequest) responsesRequest {
	out := responsesRequest{
		Model: req.Model, Stream: true, MaxOutputTokens: req.MaxTokens,
		Temperature: req.Temperature, PromptCacheKey: req.CacheKey,
		ToolChoice:        openAIResponsesToolChoice(req.ToolChoice, len(req.Tools) > 0),
		ParallelToolCalls: cloneOptionalBool(req.ParallelToolCalls, len(req.Tools) > 0),
	}
	var instructions []string
	allowImages := imageInputAllowed(p.ModelCapabilities(req.Model))
	for _, message := range req.Messages {
		switch message.Role {
		case protocol.RoleSystem:
			if message.Content != "" {
				instructions = append(instructions, message.Content)
			}
		case protocol.RoleTool:
			text := messageText(message)
			images := messageImages(message)
			var output any = text
			if allowImages && len(images) > 0 {
				parts := make([]responsesContent, 0, len(images)+1)
				if text != "" {
					parts = append(parts, responsesContent{Type: "input_text", Text: text})
				}
				for _, image := range images {
					parts = append(parts, responsesContent{
						Type: "input_image", ImageURL: imageDataURL(image),
						Detail: normalizedImageDetail(image.Detail, true),
					})
				}
				output = parts
			} else if len(images) > 0 {
				output = textWithImageNotice(text, len(images), false)
			}
			out.Input = append(out.Input, responsesInputItem{
				Type: "function_call_output", CallID: message.ToolCallID, Output: output,
			})
		case protocol.RoleAssistant:
			if message.Content != "" {
				out.Input = append(out.Input, responsesInputItem{
					Type: "message", Role: "assistant",
					Content: []responsesContent{{Type: "output_text", Text: message.Content}},
				})
			}
			for _, call := range message.ToolCalls {
				arguments := call.Arguments
				if strings.TrimSpace(arguments) == "" {
					arguments = "{}"
				}
				out.Input = append(out.Input, responsesInputItem{
					Type: "function_call", CallID: call.ID, Name: call.Name, Arguments: arguments,
				})
			}
		default:
			content := []responsesContent{{Type: "input_text", Text: messageText(message)}}
			images := messageImages(message)
			if allowImages {
				for _, image := range images {
					content = append(content, responsesContent{
						Type: "input_image", ImageURL: imageDataURL(image),
						Detail: normalizedImageDetail(image.Detail, true),
					})
				}
			} else if len(images) > 0 {
				content[0].Text = textWithImageNotice(content[0].Text, len(images), false)
			}
			out.Input = append(out.Input, responsesInputItem{
				Type: "message", Role: "user",
				Content: content,
			})
		}
	}
	out.Instructions = strings.Join(instructions, "\n\n")
	for _, schema := range req.Tools {
		parameters, strict := projectedToolParameters(schema, toolSchemaOpenAIResponses)
		out.Tools = append(out.Tools, responsesTool{
			Type: "function", Name: schema.Name, Description: schema.Description,
			Parameters: parameters, Strict: strict,
		})
	}
	if effort := strings.TrimSpace(req.ReasoningEffort); effort != "" {
		out.Reasoning = &responsesReasoning{Effort: effort}
	}
	if req.OutputSchema != nil {
		name := strings.TrimSpace(req.OutputSchema.Name)
		if name == "" {
			name = "output"
		}
		out.Text = &responsesText{Format: responsesTextFormat{
			Type: "json_schema", Name: name, Strict: req.OutputSchema.Strict, Schema: req.OutputSchema.Schema,
		}}
	}
	return out
}

// --- conversation state ---

type responsesChain struct {
	mu sync.Mutex

	lastControls responsesControls
	lastInput    []responsesInputItem
	lastOutput   []responsesInputItem
	lastID       string
	canAppend    bool
	stale        int
	disabled     bool
}

func (c *responsesChain) resetBaseline() {
	c.lastControls = responsesControls{}
	c.lastInput = nil
	c.lastOutput = nil
	c.lastID = ""
	c.canAppend = false
}

type openAIResponsesState struct {
	mu     sync.Mutex
	chains map[string]*responsesChain
	closed bool
}

func newOpenAIResponsesState() protocol.ProviderSessionState {
	return &openAIResponsesState{chains: make(map[string]*responsesChain)}
}

func (s *openAIResponsesState) chain(key string) *responsesChain {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if chain := s.chains[key]; chain != nil {
		return chain
	}
	chain := &responsesChain{}
	s.chains[key] = chain
	return chain
}

func (s *openAIResponsesState) Close() {
	s.mu.Lock()
	s.closed = true
	s.chains = nil
	s.mu.Unlock()
}

// ResetAccountScoped drops server-side response handles while preserving the
// endpoint-level circuit breaker. A deployment that categorically rejects
// previous_response_id should not be reprobed on every sibling credential.
func (s *openAIResponsesState) ResetAccountScoped() {
	s.mu.Lock()
	chains := make([]*responsesChain, 0, len(s.chains))
	if !s.closed {
		for _, chain := range s.chains {
			chains = append(chains, chain)
		}
	}
	s.mu.Unlock()
	for _, chain := range chains {
		chain.mu.Lock()
		chain.resetBaseline()
		chain.stale = 0
		chain.mu.Unlock()
	}
}

type responsesPlan struct {
	canonical responsesRequest
	wire      responsesRequest
	chain     *responsesChain
	chained   bool
	release   func()
}

func (p *OpenAIResponsesProvider) plan(req protocol.ChatRequest, apiKey string) responsesPlan {
	canonical := p.encode(req)
	plan := responsesPlan{canonical: canonical, wire: canonical, release: func() {}}
	if req.ProviderSession == nil || strings.TrimSpace(req.SessionID) == "" {
		// No retained conversation state: make the privacy/correctness behavior
		// explicit and replay the complete input.
		plan.wire.Store = false
		return plan
	}
	stateKey := responsesStatePrefix + p.Name() + "\x00" + strings.TrimRight(p.BaseURL, "/") + "\x00" + credentialFingerprint(apiKey)
	state, _ := req.ProviderSession.State(stateKey, newOpenAIResponsesState).(*openAIResponsesState)
	if state == nil {
		plan.wire.Store = false
		return plan
	}
	chain := state.chain(req.Model + "\x00" + req.SessionID)
	if chain == nil {
		plan.wire.Store = false
		return plan
	}
	chain.mu.Lock()
	plan.chain = chain
	var once sync.Once
	plan.release = func() { once.Do(chain.mu.Unlock) }
	if chain.disabled {
		plan.wire.Store = false
		return plan
	}
	plan.canonical.Store = true
	plan.wire.Store = true
	delta := responsesDelta(chain, plan.canonical)
	if len(delta) > 0 && chain.lastID != "" {
		plan.wire.Input = delta
		plan.wire.PreviousResponseID = chain.lastID
		plan.chained = true
		return plan
	}
	if chain.canAppend {
		// The caller rewrote/compacted history or changed a wire control. The
		// server's hidden prefix no longer describes this request.
		chain.resetBaseline()
	}
	return plan
}

func responsesDelta(chain *responsesChain, current responsesRequest) []responsesInputItem {
	if !chain.canAppend || !reflect.DeepEqual(chain.lastControls, current.controls()) {
		return nil
	}
	baseline := make([]responsesInputItem, 0, len(chain.lastInput)+len(chain.lastOutput))
	baseline = append(baseline, chain.lastInput...)
	baseline = append(baseline, chain.lastOutput...)
	if len(current.Input) <= len(baseline) {
		return nil
	}
	for i := range baseline {
		if !reflect.DeepEqual(baseline[i], current.Input[i]) {
			return nil
		}
	}
	return current.Input[len(baseline):]
}

func credentialFingerprint(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:8])
}

func cloneResponsesRequest(in responsesRequest) responsesRequest {
	out := in
	out.Input = append([]responsesInputItem(nil), in.Input...)
	for i := range out.Input {
		out.Input[i].Content = append([]responsesContent(nil), in.Input[i].Content...)
		if parts, ok := in.Input[i].Output.([]responsesContent); ok {
			out.Input[i].Output = append([]responsesContent(nil), parts...)
		}
	}
	out.Tools = append([]responsesTool(nil), in.Tools...)
	return out
}

func canonicalResponseOutput(text string, tools []protocol.ToolCall) []responsesInputItem {
	out := make([]responsesInputItem, 0, 1+len(tools))
	if text != "" {
		out = append(out, responsesInputItem{
			Type: "message", Role: "assistant",
			Content: []responsesContent{{Type: "output_text", Text: text}},
		})
	}
	for _, call := range tools {
		arguments := call.Arguments
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		out = append(out, responsesInputItem{
			Type: "function_call", CallID: call.ID, Name: call.Name, Arguments: arguments,
		})
	}
	return out
}

func (p responsesPlan) succeed(responseID, text string, tools []protocol.ToolCall) {
	defer p.release()
	if p.chain == nil {
		return
	}
	output := canonicalResponseOutput(text, tools)
	if responseID == "" || len(output) == 0 {
		p.chain.resetBaseline()
		return
	}
	canonical := cloneResponsesRequest(p.canonical)
	p.chain.lastControls = canonical.controls()
	p.chain.lastInput = canonical.Input
	p.chain.lastOutput = output
	p.chain.lastID = responseID
	p.chain.canAppend = true
	if p.chained {
		p.chain.stale = 0
	}
}

func (p responsesPlan) fail() {
	defer p.release()
	if p.chain != nil {
		p.chain.resetBaseline()
	}
}

func (p *responsesPlan) stale(zdr bool) {
	if p.chain == nil {
		return
	}
	p.chain.resetBaseline()
	if zdr {
		p.chain.stale = responsesStaleLimit
		p.chain.disabled = true
	} else {
		p.chain.stale++
		if p.chain.stale >= responsesStaleLimit {
			p.chain.disabled = true
		}
	}
	p.chained = false
	p.wire = p.canonical
	p.wire.Store = !p.chain.disabled
}

// --- streaming ---

type responsesStreamEvent struct {
	Type     string                 `json:"type"`
	Delta    string                 `json:"delta,omitempty"`
	Item     *responsesOutputItem   `json:"item,omitempty"`
	Response *responsesWireResponse `json:"response,omitempty"`
	Error    *responsesWireError    `json:"error,omitempty"`
	Code     string                 `json:"code,omitempty"`
	Message  string                 `json:"message,omitempty"`
}

type responsesOutputItem struct {
	Type      string             `json:"type"`
	Role      string             `json:"role,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Content   []responsesContent `json:"content,omitempty"`
}

type responsesWireError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type responsesWireUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details,omitempty"`
}

func (u *responsesWireUsage) usage() protocol.Usage {
	if u == nil {
		return protocol.Usage{}
	}
	cached := 0
	if u.InputTokensDetails != nil {
		cached = u.InputTokensDetails.CachedTokens
	}
	if cached > u.InputTokens {
		cached = u.InputTokens
	}
	return protocol.Usage{
		InputTokens: u.InputTokens - cached, OutputTokens: u.OutputTokens, CacheReadTokens: cached,
	}
}

type responsesWireResponse struct {
	ID                string                `json:"id"`
	Status            string                `json:"status"`
	Usage             *responsesWireUsage   `json:"usage,omitempty"`
	Output            []responsesOutputItem `json:"output,omitempty"`
	Error             *responsesWireError   `json:"error,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
}

// Chat drains the public API's streaming transport into one neutral response.
// Keeping one decoder prevents Chat and Stream from learning different chain
// baselines or tool-call semantics.
func (p *OpenAIResponsesProvider) Chat(ctx context.Context, req protocol.ChatRequest) (protocol.ChatResponse, error) {
	return chatViaStream(ctx, p, req)
}

func (p *OpenAIResponsesProvider) Stream(ctx context.Context, req protocol.ChatRequest) (<-chan protocol.ChatDelta, error) {
	// Snapshot once: provider-session state and Authorization must refer to the
	// same credential even if a sibling agent rotates the shared provider while
	// this request is being planned.
	apiKey := p.apiKey()
	strictState, active := prepareStrictTools(req, p.Name(), p.BaseURL, toolSchemaOpenAIResponses)
	retriedWithoutStrict := false
	for {
		plan := p.plan(active, apiKey)
		resp, err := p.open(ctx, plan.wire, apiKey)
		if err != nil && plan.chained && isStalePreviousResponse(err) {
			plan.stale(isZeroDataRetention(err))
			resp, err = p.open(ctx, plan.wire, apiKey)
		}
		if err != nil {
			plan.fail()
			if !retriedWithoutStrict && shouldRetryWithoutStrictTools(active, toolSchemaOpenAIResponses, err) {
				rememberStrictToolsRejected(strictState, active.Model)
				active = withoutStrictTools(active)
				retriedWithoutStrict = true
				continue
			}
			return nil, err
		}

		ch := make(chan protocol.ChatDelta, 16)
		go p.consume(resp, plan, ch)
		ready, preflightErr, retryStrict := preflightStrictStream(ctx, active, toolSchemaOpenAIResponses, ch)
		if retryStrict {
			rememberStrictToolsRejected(strictState, active.Model)
			active = withoutStrictTools(active)
			retriedWithoutStrict = true
			continue
		}
		if preflightErr != nil {
			return nil, preflightErr
		}
		return ready, nil
	}
}

func (p *OpenAIResponsesProvider) open(ctx context.Context, body responsesRequest, apiKey string) (*http.Response, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.responsesURL(), bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	setOptionalBearerAuth(req, apiKey)
	resp, err := p.streamHTTP().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 400 {
		return resp, nil
	}
	data := readProviderErrorBody(resp.Body)
	resp.Body.Close()
	message := strings.TrimSpace(string(data))
	var decoded struct {
		Error *responsesWireError `json:"error"`
	}
	if json.Unmarshal(data, &decoded) == nil && decoded.Error != nil && decoded.Error.Message != "" {
		message = decoded.Error.Message
		if decoded.Error.Code != "" {
			message += " (code=" + decoded.Error.Code + ")"
		}
	}
	return nil, protocol.NewProviderError(p.Name(), resp, message)
}

func (p *OpenAIResponsesProvider) consume(resp *http.Response, plan responsesPlan, ch chan<- protocol.ChatDelta) {
	defer close(ch)
	defer resp.Body.Close()

	var text strings.Builder
	var tools []protocol.ToolCall
	seenTools := make(map[string]bool)
	var responseID, stopReason string
	var usage protocol.Usage
	terminal := false
	failed := false
	defer func() {
		if failed {
			plan.fail()
		}
	}()

	emitItem := func(item responsesOutputItem, emitText bool) {
		switch item.Type {
		case "function_call":
			key := item.CallID
			if key == "" {
				key = item.Name + "\x00" + item.Arguments
			}
			if seenTools[key] {
				return
			}
			seenTools[key] = true
			arguments := string(ParseStreamingJSON(item.Arguments))
			call := protocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: arguments}
			tools = append(tools, call)
			ch <- protocol.ChatDelta{ToolCall: &call}
		case "message":
			if !emitText {
				return
			}
			for _, content := range item.Content {
				value := content.Text
				if value == "" {
					value = content.Refusal
				}
				if value != "" {
					text.WriteString(value)
					ch <- protocol.ChatDelta{ContentDelta: value}
				}
			}
		}
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			failed = true
			ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), nil, "decode responses stream event: "+err.Error())}
			return
		}
		switch event.Type {
		case "response.created":
			if event.Response != nil {
				responseID = event.Response.ID
			}
		case "response.output_text.delta", "response.refusal.delta":
			if event.Delta != "" {
				text.WriteString(event.Delta)
				ch <- protocol.ChatDelta{ContentDelta: event.Delta}
			}
		case "response.output_item.done":
			if event.Item != nil {
				emitItem(*event.Item, text.Len() == 0)
			}
		case "response.completed", "response.done", "response.incomplete":
			if event.Response == nil {
				continue
			}
			terminal = true
			if event.Response.ID != "" {
				responseID = event.Response.ID
			}
			usage = event.Response.Usage.usage()
			for _, item := range event.Response.Output {
				emitItem(item, text.Len() == 0)
			}
			switch event.Type {
			case "response.incomplete":
				stopReason = "incomplete"
				if event.Response.IncompleteDetails != nil && strings.Contains(event.Response.IncompleteDetails.Reason, "max") {
					stopReason = "length"
				}
			default:
				stopReason = "stop"
			}
		case "response.failed", "error":
			failed = true
			ch <- protocol.ChatDelta{Done: true, Err: p.responsesEventError(resp, &event)}
			return
		}
	}
	if err := scanner.Err(); err != nil {
		failed = true
		ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), nil,
			"responses stream ended with read error: "+err.Error())}
		return
	}
	if !terminal {
		failed = true
		ch <- protocol.ChatDelta{Done: true, Err: protocol.NewProviderError(p.Name(), nil,
			"responses stream ended before terminal completion event")}
		return
	}
	if len(tools) > 0 {
		stopReason = "tool_calls"
	}
	plan.succeed(responseID, text.String(), tools)
	ch <- protocol.ChatDelta{Done: true, StopReason: stopReason, Usage: usage}
}

func (p *OpenAIResponsesProvider) responsesEventError(resp *http.Response, event *responsesStreamEvent) error {
	message, code := event.Message, event.Code
	if event.Error != nil {
		message, code = event.Error.Message, event.Error.Code
	}
	if event.Response != nil && event.Response.Error != nil {
		message, code = event.Response.Error.Message, event.Response.Error.Code
	}
	if message == "" {
		message = "responses request failed"
	}
	if code != "" {
		message += " (code=" + code + ")"
	}
	status := providerStatusForCode(code)
	if status == 0 {
		if strictToolsRejectionMessage(message) {
			status = http.StatusUnprocessableEntity
		} else {
			status = http.StatusInternalServerError
		}
	}
	return providerErrorWithStatus(p.Name(), resp, status, message)
}

func isStalePreviousResponse(err error) bool {
	var providerErr *protocol.ProviderError
	if !errors.As(err, &providerErr) {
		return false
	}
	message := strings.ToLower(providerErr.Message)
	if strings.Contains(message, "previous_response_not_found") || strings.Contains(message, "previous response not found") {
		return true
	}
	mentionsPrevious := strings.Contains(message, "previous_response") || strings.Contains(message, "previous response")
	return mentionsPrevious && containsAny(message, "not found", "does not exist", "invalid", "expired", "stale", "unsupported", "not supported", "zero data retention")
}

func isZeroDataRetention(err error) bool {
	return strings.Contains(strings.ToLower(err.Error()), "zero data retention")
}

var (
	_ protocol.LLMProvider                = (*OpenAIResponsesProvider)(nil)
	_ protocol.ModelCapabilityProvider    = (*OpenAIResponsesProvider)(nil)
	_ protocol.KeyUpdater                 = (*OpenAIResponsesProvider)(nil)
	_ protocol.AccountScopedProviderState = (*openAIResponsesState)(nil)
)
