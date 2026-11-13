package ai

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
)

// devin_proto.go is the protobuf wire codec for the Devin (Codeium Cascade)
// Connect-RPC surface. It is a field-number port of oh-my-pi's
// packages/catalog/src/discovery/devin-proto.ts — the repo vendors only the
// messages the client speaks, so this file encodes/decodes the same subset by
// hand rather than pulling a protobuf runtime for one vendor.

// --- encoder ---

type pbWriter []byte

func (w *pbWriter) tag(field, wireType int) {
	*w = binary.AppendUvarint(*w, uint64(field<<3|wireType))
}
func (w *pbWriter) varint(field int, v uint64) {
	if v == 0 {
		return
	}
	w.tag(field, 0)
	*w = binary.AppendUvarint(*w, v)
}
func (w *pbWriter) str(field int, s string) {
	if s == "" {
		return
	}
	w.tag(field, 2)
	*w = binary.AppendUvarint(*w, uint64(len(s)))
	*w = append(*w, s...)
}
func (w *pbWriter) msg(field int, m []byte) {
	if m == nil {
		return
	}
	w.tag(field, 2)
	*w = binary.AppendUvarint(*w, uint64(len(m)))
	*w = append(*w, m...)
}
func (w *pbWriter) boolean(field int, v bool) {
	if v {
		w.varint(field, 1)
	}
}
func (w *pbWriter) enum(field int, v int) { w.varint(field, uint64(v)) }
func (w *pbWriter) fixed64(field int, v uint64) {
	w.tag(field, 1)
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	*w = append(*w, b[:]...)
}
func (w *pbWriter) double(field int, v float64) {
	w.fixed64(field, math.Float64bits(v))
}

// --- decoder ---

// pbField is one decoded wire field; Bytes is set for length-delimited kinds,
// Varint for varint/enum/bool kinds.
type pbField struct {
	Number int
	Wire   int
	Varint uint64
	Bytes  []byte
}

// pbScan decodes a protobuf message into its fields. Unknown fields are
// skipped; the caller reads only the numbers it knows.
func pbScan(b []byte) ([]pbField, error) {
	var out []pbField
	for len(b) > 0 {
		tag, n := binary.Uvarint(b)
		if n <= 0 {
			return nil, fmt.Errorf("devin proto: bad tag")
		}
		b = b[n:]
		f := pbField{Number: int(tag >> 3), Wire: int(tag & 7)}
		switch f.Wire {
		case 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return nil, fmt.Errorf("devin proto: bad varint field %d", f.Number)
			}
			f.Varint = v
			b = b[n:]
		case 1:
			if len(b) < 8 {
				return nil, fmt.Errorf("devin proto: truncated fixed64 field %d", f.Number)
			}
			f.Varint = binary.LittleEndian.Uint64(b)
			b = b[8:]
		case 5:
			if len(b) < 4 {
				return nil, fmt.Errorf("devin proto: truncated fixed32 field %d", f.Number)
			}
			f.Varint = uint64(binary.LittleEndian.Uint32(b))
			b = b[4:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || l > uint64(len(b)-n) {
				return nil, fmt.Errorf("devin proto: bad length field %d", f.Number)
			}
			f.Bytes = b[n : n+int(l)]
			b = b[n+int(l):]
		default:
			return nil, fmt.Errorf("devin proto: unsupported wire type %d", f.Wire)
		}
		out = append(out, f)
	}
	return out, nil
}

// --- messages (field numbers mirror devin-proto.ts) ---

// devinMetadata is exa.codeium_common_pb.Metadata. The backend gates the CLI
// model surface on the chisel identity tuple; these fields are the full set
// the reference client sends.
type devinMetadata struct {
	ideName          string
	ideType          string
	ideVersion       string
	extensionName    string
	extensionVersion string
	locale           string
	os               string
	apiKey           string
	userJwt          string
}

func devinCLIIdentity() devinMetadata {
	os := "linux"
	switch runtime.GOOS {
	case "darwin":
		os = "darwin"
	case "windows":
		os = "windows"
	}
	return devinMetadata{
		ideName:          "devin-cli",
		ideType:          "chisel",
		ideVersion:       "3000.11.3",
		extensionName:    "chisel",
		extensionVersion: "3000.11.3",
		locale:           "en",
		os:               os,
	}
}

func (m devinMetadata) encode() []byte {
	w := pbWriter{}
	w.str(1, m.ideName)
	w.str(2, m.extensionVersion)
	w.str(3, m.apiKey)
	w.str(4, m.locale)
	w.str(5, m.os)
	w.str(7, m.ideVersion)
	w.str(12, m.extensionName)
	w.str(21, m.userJwt)
	w.str(28, m.ideType)
	return w
}

// devinImageData is exa.codeium_common_pb.ImageData.
type devinImageData struct {
	base64Data string
	mimeType   string
}

func (m devinImageData) encode() []byte {
	w := pbWriter{}
	w.str(1, m.base64Data)
	w.str(2, m.mimeType)
	return w
}

// devinChatToolCall is exa.codeium_common_pb.ChatToolCall.
type devinChatToolCall struct {
	id            string
	name          string
	argumentsJSON string
}

func (m devinChatToolCall) encode() []byte {
	w := pbWriter{}
	w.str(1, m.id)
	w.str(2, m.name)
	w.str(3, m.argumentsJSON)
	return w
}

// devinPromptCacheOptions is exa.chat_pb.PromptCacheOptions.
func devinPromptCacheOptions(cacheType int) []byte {
	w := pbWriter{}
	w.enum(1, cacheType)
	return w
}

// devinChatToolDefinition is exa.chat_pb.ChatToolDefinition.
type devinChatToolDefinition struct {
	name        string
	description string
	jsonSchema  string
	strict      bool
}

func (m devinChatToolDefinition) encode() []byte {
	w := pbWriter{}
	w.str(1, m.name)
	w.str(2, m.description)
	w.str(3, m.jsonSchema)
	w.boolean(12, m.strict)
	return w
}

// devinChatMessagePrompt is exa.chat_pb.ChatMessagePrompt.
type devinChatMessagePrompt struct {
	messageID         string
	source            int
	prompt            string
	toolCalls         []devinChatToolCall
	toolCallID        string
	toolResultIsError bool
	images            []devinImageData
	thinking          string
	signature         string
	signatureType     string
}

func (m devinChatMessagePrompt) encode() []byte {
	w := pbWriter{}
	w.str(1, m.messageID)
	w.enum(2, m.source)
	w.str(3, m.prompt)
	for _, tc := range m.toolCalls {
		w.msg(6, tc.encode())
	}
	w.str(7, m.toolCallID)
	w.boolean(9, m.toolResultIsError)
	for _, img := range m.images {
		w.msg(10, img.encode())
	}
	w.str(11, m.thinking)
	w.str(12, m.signature)
	w.str(18, m.signatureType)
	return w
}

// devinCompletionConfiguration is exa.codeium_common_pb.CompletionConfiguration.
// The fixed defaults mirror buildDevinChatRequest in omp's provider.
type devinCompletionConfiguration struct {
	maxTokens    uint64
	temperature  float64
	topP         float64
	stopPatterns []string
}

func (m devinCompletionConfiguration) encode() []byte {
	w := pbWriter{}
	w.varint(1, 1) // numCompletions
	w.varint(2, m.maxTokens)
	w.varint(3, 200) // maxNewlines
	w.double(5, m.temperature)
	w.double(6, m.temperature) // firstTemperature
	w.varint(7, 50)            // topK
	w.double(8, m.topP)
	for _, p := range m.stopPatterns {
		w.str(9, p)
	}
	w.double(11, 1) // fimEotProbThreshold
	return w
}

// devinChatToolChoice encodes exa.chat_pb.ChatToolChoice with the "optionName"
// oneof arm ("auto"/"none"/a tool name goes in toolName for forced calls).
func devinChatToolChoice(optionName string) []byte {
	w := pbWriter{}
	w.str(1, optionName)
	return w
}

// devinChatRequest is exa.api_server_pb.GetChatMessageRequest.
type devinChatRequest struct {
	metadata                 devinMetadata
	prompt                   string
	prompts                  []devinChatMessagePrompt
	chatModelUID             string
	requestType              int
	configuration            devinCompletionConfiguration
	tools                    []devinChatToolDefinition
	disableParallelToolCalls bool
	toolChoice               string
	systemPromptCacheOptions []byte
	cascadeID                string
	plannerMode              int
	executionID              string
	modelAssignmentJWT       string
}

func (m devinChatRequest) encode() []byte {
	w := pbWriter{}
	w.msg(1, m.metadata.encode())
	w.str(2, m.prompt)
	for _, p := range m.prompts {
		w.msg(3, p.encode())
	}
	w.enum(7, m.requestType)
	w.msg(8, m.configuration.encode())
	for _, t := range m.tools {
		w.msg(10, t.encode())
	}
	w.boolean(11, m.disableParallelToolCalls)
	choice := m.toolChoice
	if choice == "" {
		choice = "auto"
	}
	if tc := devinChatToolChoice(choice); tc != nil {
		w.msg(12, tc)
	}
	w.msg(13, m.systemPromptCacheOptions)
	w.str(16, m.cascadeID)
	w.enum(20, m.plannerMode)
	w.str(21, m.chatModelUID)
	w.str(22, m.executionID)
	w.str(26, m.modelAssignmentJWT)
	return w
}

// devinAssignModelRequest is exa.api_server_pb.AssignModelRequest.
func devinEncodeAssignModelRequest(metadata devinMetadata, routerUID, cascadeID string, prompt *devinChatMessagePrompt) []byte {
	w := pbWriter{}
	w.msg(1, metadata.encode())
	w.str(2, routerUID)
	w.str(3, cascadeID)
	if prompt != nil {
		w.msg(5, prompt.encode())
	}
	return w
}

func devinEncodeGetUserJwtRequest(metadata devinMetadata) []byte {
	w := pbWriter{}
	w.msg(1, metadata.encode())
	return w
}

func devinEncodeGetCliModelConfigsRequest(metadata devinMetadata) []byte {
	w := pbWriter{}
	w.msg(1, metadata.encode())
	return w
}

// --- response decoders ---

type devinUserJwt struct {
	userJWT            string
	customAPIServerURL string
}

func devinDecodeUserJwt(b []byte) (devinUserJwt, error) {
	fields, err := pbScan(b)
	if err != nil {
		return devinUserJwt{}, err
	}
	var out devinUserJwt
	for _, f := range fields {
		switch f.Number {
		case 1:
			out.userJWT = string(f.Bytes)
		case 2:
			out.customAPIServerURL = string(f.Bytes)
		}
	}
	return out, nil
}

type devinModelAssignment struct {
	assignmentJWT string
	modelUID      string
}

func devinDecodeAssignModelResponse(b []byte) (devinModelAssignment, error) {
	fields, err := pbScan(b)
	if err != nil {
		return devinModelAssignment{}, err
	}
	var out devinModelAssignment
	for _, f := range fields {
		if f.Number != 1 {
			continue
		}
		inner, err := pbScan(f.Bytes)
		if err != nil {
			return devinModelAssignment{}, err
		}
		for _, sub := range inner {
			switch sub.Number {
			case 1:
				out.assignmentJWT = string(sub.Bytes)
			case 2:
				out.modelUID = string(sub.Bytes)
			}
		}
	}
	return out, nil
}

type devinModelUsageStats struct {
	invalid          bool
	modelUid         string
	inputTokens      uint64
	outputTokens     uint64
	cacheWriteTokens uint64
	cacheReadTokens  uint64
}

func devinDecodeUsageStats(b []byte) devinModelUsageStats {
	fields, err := pbScan(b)
	if err != nil {
		return devinModelUsageStats{invalid: true}
	}
	var out devinModelUsageStats
	for _, f := range fields {
		if f.Number >= 2 && f.Number <= 5 && f.Wire != 0 {
			out.invalid = true
		}
		switch f.Number {
		case 9:
			out.modelUid = string(f.Bytes)
		case 2:
			out.inputTokens = f.Varint
		case 3:
			out.outputTokens = f.Varint
		case 4:
			out.cacheWriteTokens = f.Varint
		case 5:
			out.cacheReadTokens = f.Varint
		}
	}
	return out
}

// devinChatDelta is one decoded GetChatMessageResponse stream frame.
type devinChatDelta struct {
	messageID           string
	deltaText           string
	stopReason          int
	toolCalls           []devinChatToolCall
	usage               *devinModelUsageStats
	creditCost          int64
	deltaThinking       string
	deltaSignature      string
	committedCreditCost int64
	committedAcuCost    float64
	actualModelUID      string
}

func devinDecodeChatResponse(b []byte) (devinChatDelta, error) {
	fields, err := pbScan(b)
	if err != nil {
		return devinChatDelta{}, err
	}
	var out devinChatDelta
	for _, f := range fields {
		switch f.Number {
		case 1:
			out.messageID = string(f.Bytes)
		case 3:
			out.deltaText = string(f.Bytes)
		case 5:
			out.stopReason = int(f.Varint)
		case 6:
			inner, err := pbScan(f.Bytes)
			if err != nil {
				return out, err
			}
			var call devinChatToolCall
			for _, sub := range inner {
				switch sub.Number {
				case 1:
					call.id = string(sub.Bytes)
				case 2:
					call.name = string(sub.Bytes)
				case 3:
					call.argumentsJSON = string(sub.Bytes)
				}
			}
			out.toolCalls = append(out.toolCalls, call)
		case 7:
			usage := devinDecodeUsageStats(f.Bytes)
			if f.Wire != 2 {
				usage.invalid = true
			}
			out.usage = &usage
		case 14:
			out.creditCost = int64(f.Varint)
		case 9:
			out.deltaThinking = string(f.Bytes)
		case 10:
			out.deltaSignature = string(f.Bytes)
		case 18:
			out.committedCreditCost = int64(f.Varint)
		case 22:
			out.committedAcuCost = math.Float64frombits(f.Varint)
		case 23:
			out.actualModelUID = string(f.Bytes)
		}
	}
	return out, nil
}

// --- discovery (GetCliModelConfigs) decode ---

type devinFamilyMetadataValue struct {
	order int64
	name  string
}

type devinFamilyMetadataEntry struct {
	key   string
	value *devinFamilyMetadataValue
}

type devinFamilyMetadata struct {
	label                  string
	entries                []devinFamilyMetadataEntry
	isDefaultModelInFamily bool
}

type devinClientModelConfig struct {
	label                  string
	modelUID               string
	disabled               bool
	supportsImages         bool
	maxTokens              int64
	family                 *devinFamilyMetadata
	isDefaultModelInFamily bool
}

func devinDecodeFamilyValue(b []byte) *devinFamilyMetadataValue {
	fields, err := pbScan(b)
	if err != nil {
		return nil
	}
	out := &devinFamilyMetadataValue{}
	for _, f := range fields {
		switch f.Number {
		case 1:
			out.order = int64(f.Varint)
		case 2:
			out.name = string(f.Bytes)
		}
	}
	return out
}

func devinDecodeFamilyMetadata(b []byte) *devinFamilyMetadata {
	fields, err := pbScan(b)
	if err != nil {
		return nil
	}
	out := &devinFamilyMetadata{}
	for _, f := range fields {
		switch f.Number {
		case 1:
			out.label = string(f.Bytes)
		case 2:
			inner, err := pbScan(f.Bytes)
			if err != nil {
				continue
			}
			entry := devinFamilyMetadataEntry{}
			for _, sub := range inner {
				switch sub.Number {
				case 1:
					entry.key = string(sub.Bytes)
				case 2:
					entry.value = devinDecodeFamilyValue(sub.Bytes)
				}
			}
			out.entries = append(out.entries, entry)
		case 3:
			out.isDefaultModelInFamily = f.Varint != 0
		}
	}
	return out
}

func devinDecodeClientModelConfig(b []byte) devinClientModelConfig {
	var out devinClientModelConfig
	fields, err := pbScan(b)
	if err != nil {
		return out
	}
	for _, f := range fields {
		switch f.Number {
		case 1:
			out.label = string(f.Bytes)
		case 22:
			out.modelUID = string(f.Bytes)
		case 4:
			out.disabled = f.Varint != 0
		case 5:
			out.supportsImages = f.Varint != 0
		case 18:
			out.maxTokens = int64(f.Varint)
		case 30:
			out.family = devinDecodeFamilyMetadata(f.Bytes)
		case 31:
			out.isDefaultModelInFamily = f.Varint != 0
		}
	}
	return out
}

func devinDecodeCliModelConfigs(b []byte) ([]devinClientModelConfig, error) {
	fields, err := pbScan(b)
	if err != nil {
		return nil, err
	}
	var out []devinClientModelConfig
	for _, f := range fields {
		if f.Number == 1 {
			out = append(out, devinDecodeClientModelConfig(f.Bytes))
		}
	}
	return out, nil
}
