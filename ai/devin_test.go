package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// --- proto codec -------------------------------------------------------------

func TestDevinSummaryDisablesToolChoiceOnWire(t *testing.T) {
	control := devinDecodeControls(map[string]any{"toolChoice": "none"})
	r := devinChatRequest{toolChoice: control.toolChoice}
	fields, err := pbScan(r.encode())
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fields {
		if f.Number == 12 {
			inner, err := pbScan(f.Bytes)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range inner {
				if v.Number == 1 && string(v.Bytes) == "none" {
					return
				}
			}
		}
	}
	t.Fatal("none tool choice did not reach Devin request")
}

func TestDevinProtoRequestFields(t *testing.T) {
	metadata := devinCLIIdentity()
	metadata.apiKey = devinSessionPrefix + "session-token"
	metadata.userJwt = "user-jwt"
	request := devinChatRequest{
		metadata:     metadata,
		prompt:       "system prompt",
		prompts:      []devinChatMessagePrompt{{messageID: "m1", source: devinSourceUser, prompt: "hello"}},
		chatModelUID: "swe-2-high",
		requestType:  devinRequestTypeCascade,
		configuration: devinCompletionConfiguration{
			maxTokens: 64000, temperature: 0.4, topP: 1, stopPatterns: devinStopPatterns,
		},
		tools:                    []devinChatToolDefinition{{name: "reply", description: "d", jsonSchema: `{"type":"object"}`}},
		disableParallelToolCalls: false,
		systemPromptCacheOptions: devinPromptCacheOptions(devinCacheTypeEphemeral),
		cascadeID:                "cascade-1",
		plannerMode:              devinPlannerDefault,
		executionID:              "exec-1",
		modelAssignmentJWT:       "assignment-jwt",
	}
	fields, err := pbScan(request.encode())
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int][]byte{}
	for _, field := range fields {
		seen[field.Number] = field.Bytes
	}
	if got := string(seen[21]); got != "swe-2-high" {
		t.Fatalf("chatModelUid field 21 = %q", got)
	}
	if got := string(seen[16]); got != "cascade-1" {
		t.Fatalf("cascadeId field 16 = %q", got)
	}
	if got := string(seen[26]); got != "assignment-jwt" {
		t.Fatalf("modelAssignmentJwt field 26 = %q", got)
	}
	if seen[10] == nil {
		t.Fatal("tools field 10 missing")
	}
	inner, err := pbScan(seen[1])
	if err != nil {
		t.Fatal(err)
	}
	var apiKey, userJWT, ideType string
	for _, field := range inner {
		switch field.Number {
		case 3:
			apiKey = string(field.Bytes)
		case 21:
			userJWT = string(field.Bytes)
		case 28:
			ideType = string(field.Bytes)
		}
	}
	if apiKey != devinSessionPrefix+"session-token" || userJWT != "user-jwt" || ideType != "chisel" {
		t.Fatalf("metadata apiKey=%q userJwt=%q ideType=%q", apiKey, userJWT, ideType)
	}
}

func TestDevinProtoChatDeltaDecode(t *testing.T) {
	// deltaText (3), stopReason (5), deltaThinking (9), deltaSignature (10),
	// usage (7), actualModelUid (23), messageId (1).
	usage := pbWriter{}
	usage.varint(2, 120)
	usage.varint(3, 34)
	usage.varint(5, 7)
	usage.str(9, "swe-2-high")

	writer := pbWriter{}
	writer.str(1, "msg-1")
	writer.str(3, "hello")
	writer.enum(5, devinStopReasonMaxTokens)
	writer.msg(7, usage)
	writer.str(9, "think")
	writer.str(10, "sig")
	writer.str(23, "swe-2-max")

	delta, err := devinDecodeChatResponse(writer)
	if err != nil {
		t.Fatal(err)
	}
	if delta.messageID != "msg-1" || delta.deltaText != "hello" || delta.deltaThinking != "think" || delta.deltaSignature != "sig" {
		t.Fatalf("delta=%+v", delta)
	}
	if delta.stopReason != devinStopReasonMaxTokens || delta.actualModelUID != "swe-2-max" {
		t.Fatalf("stop=%d actual=%q", delta.stopReason, delta.actualModelUID)
	}
	if delta.usage == nil || delta.usage.inputTokens != 120 || delta.usage.outputTokens != 34 || delta.usage.cacheReadTokens != 7 {
		t.Fatalf("usage=%+v", delta.usage)
	}
}

func TestDevinConnectFrameReader(t *testing.T) {
	payload := []byte("frame-payload")
	frame := make([]byte, 5+len(payload))
	frame[0] = devinConnectEndStreamFlag
	frame[1], frame[2], frame[3], frame[4] = 0, 0, 0, byte(len(payload))
	copy(frame[5:], payload)

	reader := &devinConnectReader{reader: bytes.NewReader(frame)}
	flag, decoded, err := reader.next()
	if err != nil {
		t.Fatal(err)
	}
	if flag != devinConnectEndStreamFlag || string(decoded) != string(payload) {
		t.Fatalf("flag=%d payload=%q", flag, decoded)
	}
	if _, _, err := reader.next(); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF, got %v", err)
	}
}

func TestDevinConnectTrailerError(t *testing.T) {
	trailer := []byte(`{"error":{"code":"invalid_argument","message":"bad request"}}`)
	err := devinTrailerError(trailer)
	if err == nil {
		t.Fatal("expected trailer error")
	}
	var providerError *protocol.ProviderError
	if !errors.As(err, &providerError) {
		t.Fatalf("error type %T", err)
	}
	if providerError.Status != http.StatusBadRequest {
		t.Fatalf("status=%d", providerError.Status)
	}
	if devinTrailerError([]byte(`{"metadata":{}}`)) != nil {
		t.Fatal("non-error trailer must be ignored")
	}
}

// --- registration ------------------------------------------------------------

func TestDevinVendorRegistration(t *testing.T) {
	if got := NormalizeOAuthVendor("devin"); got != VendorDevin {
		t.Fatalf("NormalizeOAuthVendor(devin) = %q", got)
	}
	if !IsOAuthVendor("devin-agent") {
		t.Fatal("devin-agent alias must be an OAuth vendor")
	}
	capabilities := CapabilitiesFor(VendorDevin, "swe-2")
	if capabilities.Tools != protocol.CapabilitySupported || capabilities.ToolChoice != protocol.CapabilitySupported {
		t.Fatalf("capabilities=%+v", capabilities)
	}
	if err := ValidateNativeModel(json.RawMessage(`{"api":"devin-agent"}`)); err != nil {
		t.Fatalf("devin-agent must validate: %v", err)
	}
	if err := ValidateNativeModel(json.RawMessage(`{"api":"devin-unknown"}`)); err == nil {
		t.Fatal("unknown API must not validate")
	}

	client, err := NewNativeClient(ClientSpec{Name: "devin", TokenSource: &fakeTokenSource{tokens: []OAuthToken{{AccessToken: "token"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if client.api != "devin-agent" || client.provider != VendorDevin || client.endpoint != DevinDefaultBaseURL {
		t.Fatalf("client api=%q provider=%q endpoint=%q", client.api, client.provider, client.endpoint)
	}
	if _, err := NewNativeClient(ClientSpec{Name: "devin"}); err == nil {
		t.Fatal("an OAuth vendor without a pool must be refused")
	}
}

// --- uid resolution ----------------------------------------------------------

// writeDevinCatalog encodes a GetCliModelConfigsResponse for tests.
func writeDevinCatalog(t *testing.T, configs ...[]byte) []byte {
	t.Helper()
	writer := pbWriter{}
	for _, config := range configs {
		writer.msg(1, config)
	}
	return writer
}

func devinTestConfig(uid, label string, defaults bool, familyLabel string, efforts map[string]string) []byte {
	writer := pbWriter{}
	writer.str(1, label)
	writer.str(22, uid)
	writer.boolean(31, defaults)
	if familyLabel != "" || len(efforts) > 0 {
		family := pbWriter{}
		family.str(1, familyLabel)
		for name, value := range efforts {
			entry := pbWriter{}
			entry.str(1, "Reasoning Effort")
			meta := pbWriter{}
			meta.str(2, name)
			entry.msg(2, meta)
			family.msg(2, entry)
			_ = value
		}
		writer.msg(30, family)
	}
	return writer
}

func TestDevinResolveModelUIDFromCatalog(t *testing.T) {
	catalog := writeDevinCatalog(t,
		devinTestConfig("swe-2-medium", "SWE-2", false, "SWE-2", map[string]string{"Medium": "swe-2-medium"}),
		devinTestConfig("swe-2-high", "SWE-2", true, "SWE-2", map[string]string{"High": "swe-2-high"}),
		devinTestConfig("swe-2-max", "SWE-2", false, "SWE-2", map[string]string{"Max": "swe-2-max"}),
	)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(catalog)), Request: request}, nil
	})}
	devinCatalogCache.Lock()
	devinCatalogCache.entries = map[string]devinCatalogCached{}
	devinCatalogCache.Unlock()

	base := "https://catalog.test"
	if got := devinResolveModelUID(context.Background(), client, base, "tok", "swe-2", "max"); got != "swe-2-max" {
		t.Fatalf("effort max → %q", got)
	}
	if got := devinResolveModelUID(context.Background(), client, base, "tok", "swe-2", ""); got != "swe-2-high" {
		t.Fatalf("default member → %q", got)
	}
	if got := devinResolveModelUID(context.Background(), client, base, "tok", "swe-2-high", "max"); got != "swe-2-high" {
		t.Fatalf("raw wire uid must win: %q", got)
	}
	if got := devinResolveModelUID(context.Background(), client, base, "tok", "adaptive", "high"); got != "adaptive" {
		t.Fatalf("router uid must pass through: %q", got)
	}
}

func TestDevinResolveModelUIDOfflineFallback(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return nil, errors.New("offline")
	})}
	devinCatalogCache.Lock()
	devinCatalogCache.entries = map[string]devinCatalogCached{}
	devinCatalogCache.Unlock()

	if got := devinResolveModelUID(context.Background(), client, "https://catalog.test", "tok", "swe-2", "high"); got != "swe-2-high" {
		t.Fatalf("suffix fallback → %q", got)
	}
	if got := devinResolveModelUID(context.Background(), client, "https://catalog.test", "tok", "swe-2-high", "high"); got != "swe-2-high" {
		t.Fatalf("already suffixed → %q", got)
	}
	if got := devinResolveModelUID(context.Background(), client, "https://catalog.test", "tok", "swe-2", "none"); got != "swe-2" {
		t.Fatalf("no effort → %q", got)
	}
}

// --- end-to-end transport ----------------------------------------------------

// devinTestServer serves GetUserJwt and a Connect-framed GetChatMessage stream.
func devinTestServer(t *testing.T, frames [][]byte, captured *[]byte) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch {
		case strings.HasSuffix(request.URL.Path, devinAuthPath):
			body := pbWriter{}
			body.str(1, "user-jwt")
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(body)), Request: request}, nil
		case strings.HasSuffix(request.URL.Path, devinCLIModelsPath):
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(nil)), Request: request}, nil
		case strings.HasSuffix(request.URL.Path, devinChatPath):
			raw, err := io.ReadAll(request.Body)
			if err != nil {
				return nil, err
			}
			if captured != nil {
				*captured = raw
			}
			var stream bytes.Buffer
			for _, payload := range frames {
				stream.WriteByte(0) // identity frame
				var length [4]byte
				length[0] = byte(len(payload) >> 24)
				length[1] = byte(len(payload) >> 16)
				length[2] = byte(len(payload) >> 8)
				length[3] = byte(len(payload))
				stream.Write(length[:])
				stream.Write(payload)
			}
			trailer := []byte(`{}`)
			stream.WriteByte(devinConnectEndStreamFlag)
			var length [4]byte
			length[3] = byte(len(trailer))
			stream.Write(length[:])
			stream.Write(trailer)
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(&stream), Request: request}, nil
		}
		return &http.Response{StatusCode: 404, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("")), Request: request}, nil
	})}
}

func TestDevinPooledStreamTextThinkingAndToolCall(t *testing.T) {
	thinking := pbWriter{}
	thinking.str(9, "let me think")
	text := pbWriter{}
	text.str(3, "hello")
	toolCall := pbWriter{}
	toolCall.str(1, "call-1")
	toolCall.str(2, "reply")
	toolCall.str(3, `{"text":"hi"}`)
	tool := pbWriter{}
	tool.str(1, "msg-9")
	tool.msg(6, toolCall)
	tool.enum(5, 0)
	usage := pbWriter{}
	usage.varint(2, 10)
	usage.varint(3, 20)
	usage.varint(5, 3)
	finish := pbWriter{}
	finish.msg(7, usage)

	var captured []byte
	client := devinTestServer(t, [][]byte{thinking, text, tool, finish}, &captured)

	model := json.RawMessage(`{"id":"swe-2","api":"devin-agent","provider":"devin","baseUrl":"https://cascade.test","maxTokens":1000,"input":["text"],"cost":{}}`)
	transcript := NormalizeContext(Context{
		SystemPrompt: "be useful",
		Messages:     []Message{{Role: "user", Content: TextContent("say hi")}},
		Tools:        []Tool{{Name: "reply", Description: "d", Parameters: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`)}},
	})
	options, err := json.Marshal(map[string]any{"reasoning": "high", "sessionId": "cascade-42"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := StreamDevinPooled(ctx, model, transcript, OpenAICompletionsStreamOptions{Client: client, Options: options},
		&fakeTokenSource{tokens: []OAuthToken{{AccountID: "acct", AccessToken: "session-token"}}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "toolUse" {
		t.Fatalf("stop reason %q (error %v)", result.StopReason, result.ErrorMessage)
	}
	texts, thinkingTexts, calls := "", "", []string{}
	for _, block := range result.Content.Blocks.Values() {
		switch block.Type {
		case "text":
			texts += block.Text
		case "thinking":
			thinkingTexts += block.Thinking
		case "toolCall":
			calls = append(calls, block.Name+string(block.Arguments))
		}
	}
	if texts != "hello" || thinkingTexts != "let me think" {
		t.Fatalf("text=%q thinking=%q", texts, thinkingTexts)
	}
	if len(calls) != 1 || calls[0] != `reply{"text":"hi"}` {
		t.Fatalf("tool calls=%v", calls)
	}
	if result.Usage.Input != 10 || result.Usage.Output != 20 || result.Usage.CacheRead != 3 {
		t.Fatalf("usage=%+v", result.Usage)
	}
	if !result.Usage.Observation.UsageObserved || result.Usage.Observation.UsageSource != "wire" {
		t.Fatalf("Devin usage provenance missing: %+v", result.Usage)
	}

	// The captured frame is gzip-compressed Connect framing; decode and assert
	// the turn identity and resolved model uid.
	if len(captured) < 5 || captured[0] != devinConnectCompressedFlag {
		t.Fatalf("frame flag=%v", captured[:min(5, len(captured))])
	}
	payload, err := devinGunzip(captured[5:])
	if err != nil {
		t.Fatal(err)
	}
	fields, err := pbScan(payload)
	if err != nil {
		t.Fatal(err)
	}
	values := map[int]string{}
	for _, field := range fields {
		values[field.Number] = string(field.Bytes)
	}
	if values[16] != "cascade-42" {
		t.Fatalf("cascadeId = %q", values[16])
	}
	if values[21] != "swe-2-high" {
		t.Fatalf("chatModelUid = %q", values[21])
	}
	if values[2] != "be useful" {
		t.Fatalf("system prompt = %q", values[2])
	}
	if values[10] == "" {
		t.Fatal("tool definitions missing")
	}
}

func TestDevinPooledStreamRotatesOnAuthFailure(t *testing.T) {
	attempts := 0
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch {
		case strings.HasSuffix(request.URL.Path, devinAuthPath):
			attempts++
			if attempts == 1 {
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("")), Request: request}, nil
			}
			body := pbWriter{}
			body.str(1, "user-jwt")
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(body)), Request: request}, nil
		case strings.HasSuffix(request.URL.Path, devinCLIModelsPath):
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(nil)), Request: request}, nil
		case strings.HasSuffix(request.URL.Path, devinChatPath):
			delta := pbWriter{}
			delta.str(3, "ok")
			stream := bytes.Buffer{}
			stream.WriteByte(0)
			var length [4]byte
			length[3] = byte(len(delta))
			stream.Write(length[:])
			stream.Write(delta)
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(&stream), Request: request}, nil
		}
		return &http.Response{StatusCode: 404, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("")), Request: request}, nil
	})}

	source := &fakeTokenSource{tokens: []OAuthToken{
		{AccountID: "dead", AccessToken: "dead-token"},
		{AccountID: "live", AccessToken: "live-token"},
	}}
	model := json.RawMessage(`{"id":"swe-2","api":"devin-agent","provider":"devin","baseUrl":"https://cascade.test","cost":{}}`)
	transcript := NormalizeContext(Context{Messages: []Message{{Role: "user", Content: TextContent("hi")}}})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := StreamDevinPooled(ctx, model, transcript, OpenAICompletionsStreamOptions{Client: client}, source)
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.WaitForEnd(ctx); err != nil {
		t.Fatal(err)
	}
	if result.StopReason == "error" {
		t.Fatalf("expected rotation to succeed, got %v", result.ErrorMessage)
	}
	failures := 0
	for _, report := range source.reports {
		if report.err != nil {
			failures++
		}
	}
	if failures != 1 {
		t.Fatalf("expected one reported failure, got %d", failures)
	}
}

func TestDevinChatDeltaAdapter(t *testing.T) {
	provider := NewDevinProvider()
	provider.tok = OAuthToken{AccessToken: "session"}
	if provider.Name() != VendorDevin || !provider.SupportsTools() {
		t.Fatal("provider identity")
	}
	if provider.baseURL() != DevinDefaultBaseURL {
		t.Fatalf("base url %q", provider.baseURL())
	}
}
