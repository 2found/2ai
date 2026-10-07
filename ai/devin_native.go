package ai

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// devin_native.go is the Connect-RPC producer for the Devin Cascade surface:
// gzip-framed protobuf over HTTP/1.1 with the 5-byte Connect envelope
// (1 flag byte + 4-byte big-endian length). One chat turn is three calls that
// must agree on the cascade id — GetUserJwt, optionally AssignModel, then the
// streaming GetChatMessage — exactly as oh-my-pi's provider does.
const (
	// devinConnectCompressedFlag marks a gzip payload.
	devinConnectCompressedFlag = 0x01
	// devinConnectEndStreamFlag marks the end-of-stream JSON trailer.
	devinConnectEndStreamFlag = 0x02
	// devinMaxFramePayload bounds one Connect frame so a corrupt length prefix
	// fails fast instead of buffering unbounded memory.
	devinMaxFramePayload = 16 << 20
	// devinStopReasonMaxTokens is StopReason.MAX_TOKENS.
	devinStopReasonMaxTokens = 3
)

// devinChatControls is the serializable per-turn control set, decoded from the
// options blob the engine passes (native) or from the protocol request.
type devinChatControls struct {
	sessionToken      string
	effort            string
	maxTokens         int
	temperature       float64
	sessionID         string
	hasParallel       bool
	parallelToolCalls bool
	toolChoice        string
}

func devinDecodeControls(controls map[string]any) devinChatControls {
	out := devinChatControls{temperature: devinDefaultTemperature}
	if value, ok := controls["apiKey"].(string); ok {
		out.sessionToken = value
	}
	if value, ok := controls["reasoning"].(string); ok {
		out.effort = value
	}
	switch value := controls["maxTokens"].(type) {
	case float64:
		out.maxTokens = int(value)
	case json.Number:
		if n, err := value.Int64(); err == nil {
			out.maxTokens = int(n)
		}
	case int:
		out.maxTokens = value
	}
	switch value := controls["temperature"].(type) {
	case float64:
		out.temperature = value
	case json.Number:
		if f, err := value.Float64(); err == nil {
			out.temperature = f
		}
	}
	if value, ok := controls["sessionId"].(string); ok {
		out.sessionID = value
	}
	if value, ok := controls["parallelToolCalls"].(bool); ok {
		out.hasParallel, out.parallelToolCalls = true, value
	}
	if value, ok := controls["toolChoice"].(string); ok && (value == "none" || value == "auto") {
		out.toolChoice = value
	}
	return out
}

// StreamDevinPooled binds an account source to the native Devin stream. It
// mirrors StreamCodexResponsesPooled: a typed 401/403 before visible content
// rotates the account and replays; after the first delta the failure is
// forwarded and Report feeds the outcome back to the pool.
func StreamDevinPooled(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, source TokenSource) (*AssistantMessageEventStream, error) {
	if source == nil {
		return nil, errors.New("Devin account pool is required")
	}
	return nativeOAuthPoolStream(ctx, rawModel, transcript, options, source, "Devin",
		func(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, _ OAuthToken) (*AssistantMessageEventStream, error) {
			inner := NewAssistantMessageEventStreamFor(ctx)
			go devinAgentStream(ctx, rawModel, transcript, options, inner)
			return inner, nil
		})
}

// devinFail settles the stream with a terminal error event and records the
// typed cause into the ctx failure capture so the OAuth pool and the protocol
// adapter see the real status (a message string cannot carry it).
func devinFail(ctx context.Context, out *AssistantMessageEventStream, cause error, aborted bool, now int64) {
	recordNativeFailure(ctx, VendorDevin, cause, false)
	devinFailStream(out, cause.Error(), aborted, now)
}

// devinStatusFromMessage extracts a leading HTTP status from a formatted
// provider error ("Devin auth error 401 Unauthorized: …").
func devinStatusFromMessage(message string) int {
	for _, field := range strings.Fields(message) {
		if n, err := strconv.Atoi(field); err == nil && n >= 400 && n < 600 {
			return n
		}
	}
	return 0
}

// devinFailStream settles the stream with a terminal error event.
func devinFailStream(out *AssistantMessageEventStream, message string, aborted bool, now int64) {
	status := "error"
	if aborted {
		status = "aborted"
	}
	text := message
	out.Synchronize(func() {
		out.Push(AssistantMessageEvent{Type: "error", Reason: status, Error: &Message{
			Role: "assistant", Content: BlockContent(), API: VendorDevin, Provider: VendorDevin,
			Usage: &Usage{}, StopReason: status, ErrorMessage: &text, Timestamp: now,
		}})
	})
}

// devinAgentStream runs one turn into out: auth, optional router assignment,
// then the streaming chat call. ctx carries the per-attempt failure capture.
func devinAgentStream(ctx context.Context, rawModel json.RawMessage, transcript TranscriptContext, options OpenAICompletionsStreamOptions, out *AssistantMessageEventStream) {
	defer out.End()
	defer func() {
		if recovered := recover(); recovered != nil {
			devinFail(ctx, out, fmt.Errorf("%v", recovered), false, time.Now().UnixMilli())
		}
	}()
	now := time.Now().UnixMilli()
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		devinFail(ctx, out, fmt.Errorf("invalid Devin model: %w", err), false, now)
		return
	}
	base := DevinDefaultBaseURL
	if strings.TrimSpace(model.BaseURL) != "" {
		base = strings.TrimRight(model.BaseURL, "/")
	}
	httpClient := options.Client
	if httpClient == nil {
		httpClient = NewStreamHTTPClient(0)
	}
	controls := map[string]any{}
	if len(options.Options) > 0 {
		_ = json.Unmarshal(options.Options, &controls)
	}
	control := devinDecodeControls(controls)
	session := devinNormalizeSessionToken(control.sessionToken)
	if session == "" {
		devinFail(ctx, out, &protocol.ProviderError{Provider: VendorDevin, Status: http.StatusUnauthorized,
			Message: "Devin auth error: no session token"}, false, now)
		return
	}
	cascadeID := control.sessionID
	if cascadeID == "" {
		cascadeID = newDevinCascadeID()
	}
	messages := transcript.Messages()
	system := ""
	list := make([]Message, 0, len(messages))
	for _, message := range messages {
		if message.Role == "system" {
			if text := GetSystemMessageText(message); text != "" {
				system = joinPromptParts([]string{system, text})
			}
			continue
		}
		list = append(list, message)
	}

	metadata := devinCLIIdentity()
	metadata.apiKey = session

	// GetUserJwt exchanges the long-lived session token for a per-turn JWT and
	// may redirect to a custom API server.
	userJWT, chatBase, err := devinFetchUserJWT(ctx, httpClient, base, metadata)
	if err != nil {
		devinFail(ctx, out, err, false, now)
		return
	}
	if chatBase == "" {
		chatBase = base
	}

	// Resolve the configured model name (a family id or a raw wire uid) onto
	// the wire uid for the requested effort.
	router := model.ID == devinModelRouterID
	chatModelUID := model.ID
	if !router {
		chatModelUID = devinResolveModelUID(ctx, httpClient, chatBase, session, model.ID, control.effort)
	}

	var assignment *devinModelAssignment
	if router {
		prompt := devinLastUserPrompt(list, cascadeID)
		resolved, err := devinAssignModel(ctx, httpClient, chatBase, metadata, model.ID, cascadeID, prompt)
		if err != nil {
			devinFail(ctx, out, err, false, now)
			return
		}
		assignment = &resolved
		chatModelUID = resolved.modelUID
	}

	maxTokens := control.maxTokens
	if maxTokens <= 0 {
		maxTokens = int(model.MaxTokens)
	}
	if maxTokens <= 0 {
		maxTokens = devinDefaultMaxTokens
	}
	parallel := !control.hasParallel || control.parallelToolCalls

	request := devinChatRequest{
		metadata:                 metadata,
		prompt:                   strings.TrimSpace(system),
		prompts:                  devinBuildPrompts(list, cascadeID, model),
		chatModelUID:             chatModelUID,
		requestType:              devinRequestTypeCascade,
		configuration:            devinCompletionConfiguration{maxTokens: uint64(maxTokens), temperature: control.temperature, topP: 1, stopPatterns: devinStopPatterns},
		tools:                    devinBuildTools(transcript),
		disableParallelToolCalls: !parallel,
		toolChoice:               control.toolChoice,
		systemPromptCacheOptions: devinPromptCacheOptions(devinCacheTypeEphemeral),
		cascadeID:                cascadeID,
		plannerMode:              devinPlannerDefault,
		executionID:              newDevinCascadeID(),
	}
	if assignment != nil {
		request.metadata.userJwt = userJWT
		request.modelAssignmentJWT = assignment.assignmentJWT
	} else {
		request.metadata.userJwt = userJWT
	}

	acc := newDevinAccumulator(model, control.sessionID, out, now)
	acc.start()
	if err := devinStreamChat(ctx, httpClient, chatBase, request, acc); err != nil {
		recordNativeFailure(ctx, VendorDevin, err, false)
		acc.fail(err.Error(), ctx.Err() != nil)
		return
	}
	acc.finish(ctx)
}

// devinStreamChat posts the gzip-framed Connect request and feeds decoded
// frames into the accumulator.
func devinStreamChat(ctx context.Context, httpClient *http.Client, base string, request devinChatRequest, acc *devinAccumulator) error {
	payload := request.encode()
	compressed, err := devinGzip(payload)
	if err != nil {
		return err
	}
	frame := make([]byte, 5+len(compressed))
	frame[0] = devinConnectCompressedFlag
	binary.BigEndian.PutUint32(frame[1:5], uint32(len(compressed)))
	copy(frame[5:], compressed)

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, base+devinChatPath, bytes.NewReader(frame))
	if err != nil {
		return err
	}
	httpRequest.Header.Set("content-type", "application/connect+proto")
	httpRequest.Header.Set("connect-protocol-version", "1")
	httpRequest.Header.Set("connect-content-encoding", "gzip")
	httpRequest.Header.Set("connect-accept-encoding", "gzip")
	httpRequest.Header.Set("accept-encoding", "identity")
	httpRequest.Header.Set("user-agent", "connect-go/1.18.1 (go1.26.3)")

	response, err := httpClient.Do(httpRequest)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<16))
		return devinHTTPError("API", response, body)
	}
	reader := &devinConnectReader{reader: response.Body}
	for {
		flag, payload, err := reader.next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if flag&devinConnectEndStreamFlag != 0 {
			if trailer := devinTrailerError(payload); trailer != nil {
				return trailer
			}
			continue
		}
		raw := payload
		if flag&devinConnectCompressedFlag != 0 {
			raw, err = devinGunzip(payload)
			if err != nil {
				return err
			}
		}
		delta, err := devinDecodeChatResponse(raw)
		if err != nil {
			return fmt.Errorf("Devin stream decode error: %w", err)
		}
		acc.chunk(delta)
	}
}

// devinConnectReader reads Connect envelope frames from a stream.
type devinConnectReader struct {
	reader io.Reader
	buf    []byte
}

func (r *devinConnectReader) next() (byte, []byte, error) {
	for {
		if len(r.buf) >= 5 {
			length := int(binary.BigEndian.Uint32(r.buf[1:5]))
			if length > devinMaxFramePayload {
				return 0, nil, fmt.Errorf("Devin Connect frame length %d exceeds %d-byte cap", length, devinMaxFramePayload)
			}
			if len(r.buf) >= 5+length {
				flag := r.buf[0]
				payload := append([]byte(nil), r.buf[5:5+length]...)
				r.buf = append(r.buf[:0], r.buf[5+length:]...)
				return flag, payload, nil
			}
		}
		chunk := make([]byte, 32*1024)
		n, err := r.reader.Read(chunk)
		if n > 0 {
			r.buf = append(r.buf, chunk[:n]...)
			continue
		}
		if err != nil {
			// A clean EOF arrives only with an empty buffer: leftover bytes are a
			// truncated frame, not the end of the stream.
			if err == io.EOF && len(r.buf) > 0 {
				return 0, nil, io.ErrUnexpectedEOF
			}
			return 0, nil, err
		}
	}
}

func devinGzip(payload []byte) ([]byte, error) {
	var buffer bytes.Buffer
	writer := gzip.NewWriter(&buffer)
	if _, err := writer.Write(payload); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

func devinGunzip(payload []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	return io.ReadAll(io.LimitReader(reader, devinMaxFramePayload))
}

// devinHTTPError formats a bounded, non-HTML error from a failed unary call.
func devinHTTPError(operation string, response *http.Response, payload []byte) error {
	message := strings.TrimSpace(string(payload))
	if strings.HasPrefix(strings.ToLower(message), "<") {
		message = ""
	}
	if len(message) > 4096 {
		message = message[:4096]
	}
	text := fmt.Sprintf("Devin %s error %d %s", operation, response.StatusCode, http.StatusText(response.StatusCode))
	if message != "" {
		text += ": " + message
	}
	return &protocol.ProviderError{Provider: VendorDevin, Status: response.StatusCode, Message: strings.TrimSpace(text)}
}

// devinTrailerError decodes a Connect end-of-stream trailer; nil means the
// trailer carried no error.
func devinTrailerError(payload []byte) error {
	text := strings.TrimSpace(string(payload))
	if text == "" {
		return nil
	}
	var trailer struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(text), &trailer); err != nil {
		return nil
	}
	if trailer.Error.Code == "" && trailer.Error.Message == "" {
		return nil
	}
	status := devinConnectStatus(trailer.Error.Code)
	if status == 0 {
		status = http.StatusInternalServerError
	}
	return &protocol.ProviderError{Provider: VendorDevin, Status: status,
		Message: fmt.Sprintf("Devin stream error %s: %s", trailer.Error.Code, trailer.Error.Message)}
}

// devinConnectStatus maps a Connect/gRPC status code onto an HTTP status. The
// distinction matters: an invalid_argument must classify as a client error so
// the retry/rotation budget is not spent replaying a request the server will
// always reject. Unknown codes fall through to the shared vendor table.
func devinConnectStatus(code string) int {
	switch strings.ToLower(strings.TrimSpace(code)) {
	case "invalid_argument", "failed_precondition", "out_of_range":
		return http.StatusBadRequest
	case "unauthenticated":
		return http.StatusUnauthorized
	case "permission_denied":
		return http.StatusForbidden
	case "not_found":
		return http.StatusNotFound
	case "already_exists", "aborted":
		return http.StatusConflict
	case "resource_exhausted":
		return http.StatusTooManyRequests
	case "unimplemented":
		return http.StatusNotImplemented
	case "unavailable":
		return http.StatusServiceUnavailable
	case "deadline_exceeded":
		return http.StatusGatewayTimeout
	case "internal", "unknown", "data_loss":
		return http.StatusInternalServerError
	}
	return providerStatusForCode(code)
}

// devinFetchUserJWT exchanges the session token for a per-turn JWT.
func devinFetchUserJWT(ctx context.Context, httpClient *http.Client, base string, metadata devinMetadata) (string, string, error) {
	response, err := devinUnary(ctx, httpClient, base+devinAuthPath, devinEncodeGetUserJwtRequest(metadata))
	if err != nil {
		return "", "", err
	}
	decoded, err := devinDecodeUserJwt(response)
	if err != nil {
		return "", "", fmt.Errorf("Devin auth error: %w", err)
	}
	if strings.TrimSpace(decoded.userJWT) == "" {
		return "", "", errors.New("Devin auth error: GetUserJwt returned an empty user JWT")
	}
	return decoded.userJWT, strings.TrimRight(strings.TrimSpace(decoded.customAPIServerURL), "/"), nil
}

// devinAssignModel resolves a router uid (e.g. "adaptive") into a concrete
// model uid plus its assignment JWT.
func devinAssignModel(ctx context.Context, httpClient *http.Client, base string, metadata devinMetadata, routerUID, cascadeID string, prompt *devinChatMessagePrompt) (devinModelAssignment, error) {
	body := devinEncodeAssignModelRequest(metadata, routerUID, cascadeID, prompt)
	response, err := devinUnary(ctx, httpClient, base+devinAssignModelPath, body)
	if err != nil {
		return devinModelAssignment{}, err
	}
	assignment, err := devinDecodeAssignModelResponse(response)
	if err != nil {
		return devinModelAssignment{}, fmt.Errorf("Devin AssignModel error: %w", err)
	}
	if assignment.assignmentJWT == "" || assignment.modelUID == "" {
		return devinModelAssignment{}, errors.New("Devin AssignModel error: response carried no assignment JWT and model uid")
	}
	return assignment, nil
}

// devinUnary posts a Connect unary request (content-type application/proto)
// and returns the raw response payload.
func devinUnary(ctx context.Context, httpClient *http.Client, url string, body []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("content-type", "application/proto")
	request.Header.Set("connect-protocol-version", "1")
	request.Header.Set("accept", "*/*")
	response, err := httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, devinMaxFramePayload))
	if err != nil {
		return nil, err
	}
	if response.StatusCode/100 != 2 {
		return nil, devinHTTPError("auth", response, payload)
	}
	return payload, nil
}

func newDevinCascadeID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("cascade-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", b[:])
}

// --- prompt building ---

// devinBuildTools renders the admitted tool schemas for the Cascade envelope.
func devinBuildTools(transcript TranscriptContext) []devinChatToolDefinition {
	tools := GetCurrentTools(transcript.Messages())
	out := make([]devinChatToolDefinition, 0, len(tools))
	for _, tool := range tools {
		schema := tool.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		strict, err := ResolveJSONSchemaStrictSampling(tool, true, nil)
		if err != nil {
			// "prefer" sampling degrades silently; this is unreachable unless a
			// tool declared "require" — then the raw schema is the safer send.
			strict = nil
		}
		out = append(out, devinChatToolDefinition{
			name: tool.Name, description: tool.Description,
			jsonSchema: string(schema), strict: strict != nil && *strict,
		})
	}
	return out
}

// devinBuildPrompts maps the transcript onto Cascade channels: user/developer →
// USER, assistant → SYSTEM (with thinking and tool calls), tool results → TOOL.
func devinBuildPrompts(messages []Message, cascadeID string, model completionsModel) []devinChatMessagePrompt {
	prompts := make([]devinChatMessagePrompt, 0, len(messages))
	for index, message := range messages {
		switch message.Role {
		case "user", "developer", "system":
			text, images := devinUserContent(message)
			prompts = append(prompts, devinChatMessagePrompt{
				messageID: devinStableID(cascadeID, index, message.Role), source: devinSourceUser, prompt: text, images: images,
			})
		case "assistant":
			prompt := &devinChatMessagePrompt{
				messageID: devinStableID(cascadeID, index, "assistant"), source: devinSourceSystem,
			}
			native := message.API == model.API && message.Provider == model.Provider && message.Model == model.ID
			for _, block := range message.Content.Blocks.Values() {
				if block == nil {
					continue
				}
				switch block.Type {
				case "text":
					prompt.prompt += block.Text + "\n"
				case "thinking":
					prompt.thinking += block.Thinking
					if native && prompt.signature == "" && block.ThinkingSignature != nil {
						prompt.signature = *block.ThinkingSignature
					}
				case "toolCall":
					prompt.toolCalls = append(prompt.toolCalls, devinChatToolCall{
						id: block.ID, name: block.Name, argumentsJSON: string(block.Arguments),
					})
				}
			}
			if prompt.prompt == "" && prompt.thinking == "" && prompt.signature == "" && len(prompt.toolCalls) == 0 {
				continue
			}
			if native && message.ResponseID != nil && *message.ResponseID != "" {
				prompt.messageID = *message.ResponseID
			} else {
				prompt.messageID = "bot-" + devinStableID(cascadeID, index, "assistant")
			}
			prompts = append(prompts, *prompt)
		default:
			text, images := devinUserContent(message)
			prompts = append(prompts, devinChatMessagePrompt{
				messageID: devinStableID(cascadeID, index, "tool"), source: devinSourceTool,
				toolCallID: message.ToolCallID, toolResultIsError: message.IsError, prompt: text, images: images,
			})
		}
	}
	return prompts
}

func devinUserContent(message Message) (string, []devinImageData) {
	if message.Content.Text != nil {
		return *message.Content.Text, nil
	}
	var text strings.Builder
	images := []devinImageData{}
	for _, block := range message.Content.Blocks.Values() {
		if block == nil {
			continue
		}
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "image":
			mime := block.MIMEType
			if mime == "" {
				mime = "image/png"
			}
			images = append(images, devinImageData{base64Data: block.Data, mimeType: mime})
		}
	}
	return text.String(), images
}

// devinLastUserPrompt is the router scoring prompt: the current user turn.
func devinLastUserPrompt(messages []Message, cascadeID string) *devinChatMessagePrompt {
	for index := len(messages) - 1; index >= 0; index-- {
		message := messages[index]
		if message.Role == "user" || message.Role == "developer" {
			text, images := devinUserContent(message)
			return &devinChatMessagePrompt{messageID: devinStableID(cascadeID, index, message.Role), source: devinSourceUser, prompt: text, images: images}
		}
	}
	return nil
}

// devinStableID derives a deterministic prompt id from the cascade id, message
// index and role, so ids stay stable across history rebuilds.
func devinStableID(cascadeID string, index int, role string) string {
	return fmt.Sprintf("%s-%d-%s", cascadeID, index, role)
}

// --- accumulator ---

// devinAccumulator turns decoded Cascade deltas into Pi assistant events.
type devinAccumulator struct {
	model   completionsModel
	cost    completionsCost
	session string
	stream  *AssistantMessageEventStream
	output  *Message
	ended   bool

	textBlock     *ContentBlock
	thinkingBlock *ContentBlock
	toolBlocks    map[string]*ContentBlock
	toolJSON      map[string]string
	activeTool    string
	stopReason    int
}

func newDevinAccumulator(model completionsModel, session string, stream *AssistantMessageEventStream, now int64) *devinAccumulator {
	return &devinAccumulator{
		model: model, cost: model.Cost, session: session, stream: stream,
		output:     &Message{Role: "assistant", Content: BlockContent(), API: model.API, Provider: model.Provider, Model: model.ID, Usage: &Usage{}, StopReason: "pending", Timestamp: now},
		toolBlocks: map[string]*ContentBlock{}, toolJSON: map[string]string{},
	}
}

func (a *devinAccumulator) publish(event AssistantMessageEvent) {
	if event.Type != "done" && event.Type != "error" {
		event.Partial = a.output
	}
	a.stream.Push(event)
}

func (a *devinAccumulator) start() {
	a.stream.Synchronize(func() { a.publish(AssistantMessageEvent{Type: "start"}) })
}

func (a *devinAccumulator) indexOf(block *ContentBlock) int {
	for i, candidate := range a.output.Content.Blocks.Values() {
		if candidate == block {
			return i
		}
	}
	return -1
}

func (a *devinAccumulator) endText() {
	block := a.textBlock
	if block == nil {
		return
	}
	a.textBlock = nil
	index := a.indexOf(block)
	a.publish(AssistantMessageEvent{Type: "text_end", ContentIndex: index, Content: block.Text})
}

func (a *devinAccumulator) endThinking() {
	block := a.thinkingBlock
	if block == nil {
		return
	}
	a.thinkingBlock = nil
	a.publish(AssistantMessageEvent{Type: "thinking_end", ContentIndex: a.indexOf(block), Content: block.Thinking})
}

// chunk consumes one decoded stream frame.
func (a *devinAccumulator) chunk(delta devinChatDelta) {
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		if delta.messageID != "" && a.output.ResponseID == nil {
			id := delta.messageID
			a.output.ResponseID = &id
		}
		if delta.actualModelUID != "" {
			upstream := delta.actualModelUID
			a.output.ResponseModel = &upstream
		}
		if delta.usage != nil {
			a.output.Usage.Input = float64(delta.usage.inputTokens)
			a.output.Usage.Output = float64(delta.usage.outputTokens)
			a.output.Usage.CacheRead = float64(delta.usage.cacheReadTokens)
			a.output.Usage.CacheWrite = float64(delta.usage.cacheWriteTokens)
		}
		if delta.deltaThinking != "" {
			if a.thinkingBlock == nil {
				block := &ContentBlock{Type: "thinking"}
				a.thinkingBlock = block
				a.output.Content.Blocks.Append(block)
				a.publish(AssistantMessageEvent{Type: "thinking_start", ContentIndex: a.indexOf(block)})
			}
			a.thinkingBlock.Thinking += delta.deltaThinking
			if delta.deltaSignature != "" {
				signature := delta.deltaSignature
				a.thinkingBlock.ThinkingSignature = &signature
			}
			a.publish(AssistantMessageEvent{Type: "thinking_delta", ContentIndex: a.indexOf(a.thinkingBlock), Delta: delta.deltaThinking})
		}
		if delta.deltaText != "" {
			a.endThinking()
			if a.textBlock == nil {
				block := &ContentBlock{Type: "text"}
				a.textBlock = block
				a.output.Content.Blocks.Append(block)
				a.publish(AssistantMessageEvent{Type: "text_start", ContentIndex: a.indexOf(block)})
			}
			a.textBlock.Text += delta.deltaText
			a.publish(AssistantMessageEvent{Type: "text_delta", ContentIndex: a.indexOf(a.textBlock), Delta: delta.deltaText})
		}
		for _, call := range delta.toolCalls {
			a.endText()
			a.endThinking()
			id := call.id
			if id == "" {
				id = a.activeTool
			}
			if id == "" {
				continue
			}
			block := a.toolBlocks[id]
			if block == nil {
				block = &ContentBlock{Type: "toolCall", ID: id, Name: call.name}
				a.toolBlocks[id] = block
				a.toolJSON[id] = ""
				a.output.Content.Blocks.Append(block)
				a.publish(AssistantMessageEvent{Type: "toolcall_start", ContentIndex: a.indexOf(block)})
			}
			if call.name != "" {
				block.Name = call.name
			}
			a.activeTool = id
			if call.argumentsJSON == "" {
				continue
			}
			previous := a.toolJSON[id]
			accumulated := call.argumentsJSON
			if !strings.HasPrefix(call.argumentsJSON, previous) {
				accumulated = previous + call.argumentsJSON
			}
			a.toolJSON[id] = accumulated
			delta := accumulated[len(previous):]
			if delta != "" {
				a.publish(AssistantMessageEvent{Type: "toolcall_delta", ContentIndex: a.indexOf(block), Delta: delta})
			}
		}
		if delta.stopReason != 0 {
			a.stopReason = delta.stopReason
		}
	})
}

func (a *devinAccumulator) finish(ctx context.Context) {
	a.stream.Synchronize(func() {
		if a.ended {
			return
		}
		if ctx.Err() != nil {
			a.failLocked("Request was aborted", true)
			return
		}
		a.endText()
		a.endThinking()
		for id, block := range a.toolBlocks {
			if arguments := strings.TrimSpace(a.toolJSON[id]); arguments != "" {
				block.Arguments = json.RawMessage(arguments)
			} else {
				block.Arguments = json.RawMessage("{}")
			}
			a.publish(AssistantMessageEvent{Type: "toolcall_end", ContentIndex: a.indexOf(block), ToolCall: block})
		}
		reason := "stop"
		if len(a.toolBlocks) > 0 {
			reason = "toolUse"
		} else if a.stopReason == devinStopReasonMaxTokens {
			reason = "length"
		}
		a.output.StopReason = reason
		a.output.Usage.TotalTokens = a.output.Usage.Input + a.output.Usage.Output + a.output.Usage.CacheRead + a.output.Usage.CacheWrite
		calculateNativeUsageCost(a.cost, a.output.Usage)
		a.ended = true
		a.publish(AssistantMessageEvent{Type: "done", Reason: reason, Message: a.output})
		a.stream.End()
	})
}

func (a *devinAccumulator) fail(message string, aborted bool) {
	a.stream.Synchronize(func() {
		if !a.ended {
			a.failLocked(message, aborted)
		}
	})
}

func (a *devinAccumulator) failLocked(message string, aborted bool) {
	a.output.StopReason = "error"
	if aborted {
		a.output.StopReason = "aborted"
	}
	a.output.ErrorMessage = &message
	a.ended = true
	a.publish(AssistantMessageEvent{Type: "error", Reason: a.output.StopReason, Error: a.output})
	a.stream.End()
}
