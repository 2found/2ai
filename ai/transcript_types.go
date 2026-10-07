package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/2found/2ai/jsonjs"
)

// Message is Pi's transcript union. Role selects the applicable fields. Unlike
// agentcore.Message, content order, signatures, tool results, and system deltas
// remain in their original representation. Extra retains application metadata.
type Message struct {
	encoding              *transcriptEncoding
	Role                  string          `json:"role"`
	Content               MessageContent  `json:"content"`
	Timestamp             int64           `json:"timestamp"`
	Sections              SystemSections  `json:"sections,omitempty"`
	ToolsAdded            []Tool          `json:"toolsAdded,omitempty"`
	ToolsRemoved          []ToolReference `json:"toolsRemoved,omitempty"`
	API                   string          `json:"api,omitempty"`
	Provider              string          `json:"provider,omitempty"`
	Model                 string          `json:"model,omitempty"`
	ResponseModel         *string         `json:"responseModel,omitempty"`
	ResponseID            *string         `json:"responseId,omitempty"`
	ProviderThinkingLevel *string         `json:"providerThinkingLevel,omitempty"`
	ThinkingLevel         *string         `json:"thinkingLevel,omitempty"`
	Diagnostics           json.RawMessage `json:"diagnostics,omitempty"`
	Usage                 *Usage          `json:"usage,omitempty"`
	StopReason            string          `json:"stopReason,omitempty"`
	Deferred              json.RawMessage `json:"deferred,omitempty"`
	ErrorMessage          *string         `json:"errorMessage,omitempty"`
	RawStopReason         *string         `json:"rawStopReason,omitempty"`
	EndTurn               *bool           `json:"endTurn,omitempty"`
	ToolCallID            string          `json:"toolCallId,omitempty"`
	ToolName              string          `json:"toolName,omitempty"`
	// Details is a live JSON-shaped value. nil/Undefined mean absent; Null is explicit null.
	Details     any                        `json:"-"`
	NestedCalls json.RawMessage            `json:"nestedCalls,omitempty"`
	IsError     bool                       `json:"isError,omitempty"`
	Extra       map[string]json.RawMessage `json:"-"`
}

// ContentBlock is Pi's text/thinking/image/toolCall union. Optional signatures
// use pointers so an explicit empty signature survives JSON round trips.
// A nil slot represents an array hole; decoded null entries retain a distinct
// sentinel. Both serialize as null within an array, but filtering distinguishes them.
type ContentBlock struct {
	encoding          *transcriptEncoding
	null              bool
	Type              string                     `json:"type"`
	Text              string                     `json:"text,omitempty"`
	TextSignature     *string                    `json:"textSignature,omitempty"`
	Thinking          string                     `json:"thinking,omitempty"`
	ThinkingSignature *string                    `json:"thinkingSignature,omitempty"`
	Redacted          *bool                      `json:"redacted,omitempty"`
	Data              string                     `json:"data,omitempty"`
	MIMEType          string                     `json:"mimeType,omitempty"`
	ID                string                     `json:"id,omitempty"`
	Name              string                     `json:"name,omitempty"`
	Arguments         json.RawMessage            `json:"arguments,omitempty"`
	ThoughtSignature  *string                    `json:"thoughtSignature,omitempty"`
	Namespace         *string                    `json:"namespace,omitempty"`
	Extra             map[string]json.RawMessage `json:"-"`
}

// IsNull reports an explicit decoded null. A nil block pointer is a sparse
// content-array hole, which Array.filter skips without reading its properties.
func (b *ContentBlock) IsNull() bool { return b != nil && b.null }

// NullContentBlock constructs an explicitly present null entry, distinct from
// a nil slot (hole) in MessageContent.Blocks.
func NullContentBlock() *ContentBlock { return &ContentBlock{null: true} }

// HasContent reports own-field presence without serializing the message. Native
// zero content is explicit null; decoded messages can retain an absent field.
// Assigning text or a non-nil block list makes the field present again.
func (m Message) HasContent() bool {
	return !m.Content.undefined || m.Content.Text != nil || m.Content.Blocks != nil
}

// MessageContent preserves the distinction between string and array content.
// Its zero value is JSON null, as accepted by transformMessages for old logs.
type MessageContent struct {
	undefined bool
	Text      *string
	Blocks    *BlockList
}

func TextContent(text string) MessageContent { return MessageContent{Text: &text} }
func BlockContent(blocks ...ContentBlock) MessageContent {
	refs := make([]*ContentBlock, len(blocks))
	for i, block := range blocks {
		refs[i] = &block
	}
	return BlockReferences(refs...)
}

// BlockReferences constructs a new list retaining the supplied block objects.
// Copy MessageContent to share the list itself. BlockContent also constructs
// new block objects when the caller does not need to retain references.
func BlockReferences(blocks ...*ContentBlock) MessageContent {
	return MessageContent{Blocks: NewBlockList(blocks...)}
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	if c.Text != nil {
		return jsonjs.QuoteString(*c.Text), nil
	}
	return json.Marshal(c.Blocks)
}
func (c *MessageContent) UnmarshalJSON(data []byte) error {
	*c = MessageContent{}
	if len(bytes.TrimSpace(data)) > 0 && bytes.TrimSpace(data)[0] == '"' {
		value, err := jsonjs.DecodeJSON(data)
		if err != nil {
			return err
		}
		text := value.(string)
		c.Text = &text
		return nil
	}
	return json.Unmarshal(data, &c.Blocks)
}

type Usage struct {
	encoding     *transcriptEncoding
	Input        float64   `json:"input"`
	Output       float64   `json:"output"`
	CacheRead    float64   `json:"cacheRead"`
	CacheWrite   float64   `json:"cacheWrite"`
	CacheWrite1h *float64  `json:"cacheWrite1h,omitempty"`
	Reasoning    *float64  `json:"reasoning,omitempty"`
	TotalTokens  float64   `json:"totalTokens"`
	Cost         UsageCost `json:"cost"`
}
type UsageCost struct {
	encoding   *transcriptEncoding
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

type Tool struct {
	encoding            *transcriptEncoding
	Name                string                     `json:"name"`
	Description         string                     `json:"description"`
	Parameters          json.RawMessage            `json:"parameters"`
	ConstrainedSampling json.RawMessage            `json:"constrainedSampling,omitempty"`
	Extra               map[string]json.RawMessage `json:"-"`
}
type ToolReference struct {
	encoding *transcriptEncoding
	Name     string                     `json:"name"`
	Extra    map[string]json.RawMessage `json:"-"`
}

type Context struct {
	SystemPrompt string    `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages"`
	Tools        []Tool    `json:"tools,omitempty"`
}

// TranscriptContext can only be constructed through NormalizeContext. Provider
// implementations receive the prompt and tools through system messages.
type TranscriptContext struct{ messages []Message }

func (c TranscriptContext) Messages() []Message { return c.messages }
func (c TranscriptContext) MarshalJSON() ([]byte, error) {
	messages := c.messages
	if messages == nil {
		messages = []Message{}
	}
	return json.Marshal(struct {
		Messages []Message `json:"messages"`
	}{messages})
}

// SystemSections is an ordered JSON object, not a Go map: section order changes
// the actual model prompt. A nil Value removes a section. As in JavaScript,
// array-index names enumerate before other names, in numeric order.
type SystemSection struct {
	Name  string
	Value *string
}
type SystemSections []SystemSection

func (s SystemSections) ordered() SystemSections {
	result := make(SystemSections, 0, len(s))
	indices := map[string]int{}
	for _, section := range s {
		if i, ok := indices[section.Name]; ok {
			result[i] = section
		} else {
			indices[section.Name] = len(result)
			result = append(result, section)
		}
	}
	sort.SliceStable(result, func(i, j int) bool {
		a, ai := jsonjs.ArrayIndex(result[i].Name)
		b, bi := jsonjs.ArrayIndex(result[j].Name)
		if ai && bi {
			return a < b
		}
		return ai && !bi
	})
	return result
}
func (s SystemSections) MarshalJSON() ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('{')
	for i, section := range s.ordered() {
		if i > 0 {
			out.WriteByte(',')
		}
		out.Write(jsonjs.QuoteString(section.Name))
		out.WriteByte(':')
		if section.Value == nil {
			out.WriteString("null")
		} else {
			out.Write(jsonjs.QuoteString(*section.Value))
		}
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}
func (s *SystemSections) UnmarshalJSON(data []byte) error {
	*s = nil
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil
	}
	properties, err := jsonjs.DecodeObjectProperties(data)
	if err != nil {
		return err
	}
	if bytes.TrimSpace(data)[0] != '{' {
		return fmt.Errorf("ai: sections must be an object")
	}
	for _, property := range properties {
		var value *string
		if err := json.Unmarshal(property.Value, &value); err != nil {
			return err
		}
		if value != nil {
			decoded, err := jsonjs.DecodeJSON(property.Value)
			if err != nil {
				return err
			}
			*value = decoded.(string)
		}
		*s = append(*s, SystemSection{Name: property.Name, Value: value})
	}
	*s = s.ordered()
	return nil
}

// JSON union encoding retains required empty fields and unknown metadata. The
// latter is necessary for Pi's spread-based transcript transformations.
func marshalTranscriptObject(value any, extra map[string]json.RawMessage, required map[string]any) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var fields jsonjs.RawObject
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	// encoding/json replaces lone UTF-16 units carried as WTF-8. Restore all
	// emitted string fields, including optional signatures and model metadata.
	v := reflect.ValueOf(value)
	for i := 0; i < v.NumField(); i++ {
		name := strings.SplitN(v.Type().Field(i).Tag.Get("json"), ",", 2)[0]
		if _, emitted := fields[name]; !emitted {
			continue
		}
		field := v.Field(i)
		if field.Kind() == reflect.Pointer && !field.IsNil() {
			field = field.Elem()
		}
		if field.Kind() == reflect.String {
			fields[name] = jsonjs.QuoteString(field.String())
		}
	}
	for key, value := range extra {
		if _, exists := fields[key]; !exists {
			fields[key] = value
		}
	}
	for key, value := range required {
		if jsonjs.IsUndefined(value) {
			delete(fields, key)
			continue
		}
		// Native validation/provider strings may contain decoded lone UTF-16
		// surrogates. encoding/json replaces their WTF-8 bytes; retain their
		// original code units when exporting required transcript text fields.
		if text, ok := value.(string); ok {
			fields[key] = jsonjs.QuoteString(text)
			continue
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		fields[key] = encoded
	}
	return json.Marshal(fields)
}

func (m Message) MarshalJSON() ([]byte, error) {
	type plain Message
	required := map[string]any{}
	if m.ErrorMessage != nil {
		required["errorMessage"] = *m.ErrorMessage
	}
	if !m.HasContent() {
		required["content"] = Undefined
	}
	if m.Role == "assistant" {
		required["api"], required["provider"], required["model"] = m.API, m.Provider, m.Model
		required["stopReason"] = m.StopReason
		if m.Usage == nil {
			required["usage"] = Usage{}
		}
	}
	if m.Role == "toolResult" {
		required["isError"], required["toolCallId"], required["toolName"] = m.IsError, m.ToolCallID, m.ToolName
	}
	if m.ToolsAdded != nil {
		required["toolsAdded"] = m.ToolsAdded
	}
	if m.ToolsRemoved != nil {
		required["toolsRemoved"] = m.ToolsRemoved
	}
	if m.Sections != nil {
		required["sections"] = m.Sections
	}
	details, err := jsonjs.MarshalOptional(m.Details, "details")
	if err != nil {
		return nil, err
	}
	if details != nil {
		required["details"] = json.RawMessage(details)
	}
	raw, err := marshalTranscriptObject(plain(m), m.Extra, required)
	return restoreTranscriptEncoding(raw, err, m.encoding)
}
func (b ContentBlock) MarshalJSON() ([]byte, error) {
	if b.null {
		return []byte("null"), nil
	}
	type plain ContentBlock
	required := map[string]any{}
	switch b.Type {
	case "text":
		required["text"] = b.Text
	case "thinking":
		required["thinking"] = b.Thinking
	case "image":
		required["data"], required["mimeType"] = b.Data, b.MIMEType
	case "toolCall":
		required["id"], required["name"], required["arguments"] = b.ID, b.Name, b.Arguments
	}
	raw, err := marshalTranscriptObject(plain(b), b.Extra, required)
	return restoreTranscriptEncoding(raw, err, b.encoding)
}
func (t Tool) MarshalJSON() ([]byte, error) {
	type plain Tool
	raw, err := marshalTranscriptObject(plain(t), t.Extra, nil)
	return restoreTranscriptEncoding(raw, err, t.encoding)
}

func (t ToolReference) MarshalJSON() ([]byte, error) {
	type plain ToolReference
	raw, err := marshalTranscriptObject(plain(t), t.Extra, nil)
	return restoreTranscriptEncoding(raw, err, t.encoding)
}

func (t *ToolReference) UnmarshalJSON(data []byte) error {
	type plain ToolReference
	*t = ToolReference{}
	extra, err := decodeTranscriptObject(data, (*plain)(t), "name")
	t.Extra = extra
	if err == nil {
		t.encoding, err = captureTranscriptEncoding(data, t)
	}
	return err
}

func decodeTranscriptObject(data []byte, target any, known string) (map[string]json.RawMessage, error) {
	if err := json.Unmarshal(data, target); err != nil {
		return nil, err
	}
	var extra jsonjs.RawObject
	if err := json.Unmarshal(data, &extra); err != nil {
		return nil, err
	}
	// Recover the original code units before callers append deltas or copy
	// decoded fields. Retaining raw encoding metadata alone cannot do that.
	v := reflect.ValueOf(target).Elem()
	for i := 0; i < v.NumField(); i++ {
		name := strings.SplitN(v.Type().Field(i).Tag.Get("json"), ",", 2)[0]
		raw := bytes.TrimSpace(extra[name])
		if len(raw) == 0 || raw[0] != '"' {
			continue
		}
		field := v.Field(i)
		if field.Kind() == reflect.Pointer && !field.IsNil() {
			field = field.Elem()
		}
		if field.CanSet() && field.Kind() == reflect.String {
			decoded, err := jsonjs.DecodeJSON(raw)
			if err != nil {
				return nil, err
			}
			field.SetString(decoded.(string))
		}
	}
	for _, key := range bytes.Fields([]byte(known)) {
		delete(extra, string(key))
	}
	if len(extra) == 0 {
		extra = nil
	}
	return extra, nil
}
func (m *Message) UnmarshalJSON(data []byte) error {
	type plain Message
	*m = Message{}
	var fields jsonjs.RawObject
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	// Pi copies model identity properties without string validation, including
	// into failure messages. Keep non-string JSON values in encoding metadata;
	// native string fields remain empty until explicitly replaced by callers.
	identity := map[string]json.RawMessage{}
	for _, key := range []string{"api", "provider", "model"} {
		value := bytes.TrimSpace(fields[key])
		if len(value) > 0 && value[0] != '"' && !bytes.Equal(value, []byte("null")) {
			identity[key] = fields[key]
			delete(fields, key)
		}
	}
	decoded := data
	if len(identity) > 0 {
		var err error
		decoded, err = json.Marshal(fields)
		if err != nil {
			return err
		}
	}
	extra, err := decodeTranscriptObject(decoded, (*plain)(m), "role content timestamp sections toolsAdded toolsRemoved api provider model responseModel responseId providerThinkingLevel thinkingLevel diagnostics usage stopReason deferred errorMessage rawStopReason endTurn toolCallId toolName details nestedCalls isError")
	m.Extra = extra
	if err == nil {
		m.Content.undefined = fields["content"] == nil
		m.Details, err = jsonjs.DecodeOptional(fields["details"])
	}
	if err == nil {
		m.encoding, err = captureTranscriptEncoding(data, m)
	}
	if err == nil && len(identity) > 0 {
		if m.encoding == nil {
			m.encoding = &transcriptEncoding{original: map[string]json.RawMessage{}, normalized: map[string]json.RawMessage{}}
		}
		for key, value := range identity {
			m.encoding.original[key] = value
			m.encoding.normalized[key] = nil
			if m.Role == "assistant" {
				m.encoding.normalized[key] = json.RawMessage(`""`)
			}
		}
	}
	return err
}
func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	type plain ContentBlock
	*b = ContentBlock{}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		b.null = true
		return nil
	}
	extra, err := decodeTranscriptObject(data, (*plain)(b), "type text thinking data mimeType id name arguments")
	// Optional pointer fields distinguish strings/bools from null, but cannot
	// distinguish null from omission alone. Keep explicit nulls in Extra so
	// snapshots and proxy terminal updates retain the original wire shape.
	for name, present := range map[string]bool{
		"textSignature": b.TextSignature != nil, "thinkingSignature": b.ThinkingSignature != nil,
		"thoughtSignature": b.ThoughtSignature != nil, "namespace": b.Namespace != nil, "redacted": b.Redacted != nil,
	} {
		if present {
			delete(extra, name)
		}
	}
	b.Extra = extra
	return err
}
func (t *Tool) UnmarshalJSON(data []byte) error {
	type plain Tool
	*t = Tool{}
	extra, err := decodeTranscriptObject(data, (*plain)(t), "name description parameters constrainedSampling")
	t.Extra = extra
	if err == nil {
		t.encoding, err = captureTranscriptEncoding(data, t)
	}
	return err
}

// Retain the wire shape of decoded records. Pi accepts history objects as-is;
// serializing a Go zero value must not add absent fields or replace explicit
// nulls in those objects. Changed fields still serialize from their Go values.
type transcriptEncoding struct {
	original   map[string]json.RawMessage
	normalized map[string]json.RawMessage
}

func captureTranscriptEncoding(raw []byte, value any) (*transcriptEncoding, error) {
	var original, normalized jsonjs.RawObject
	if err := json.Unmarshal(raw, &original); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		return nil, err
	}
	encoding := &transcriptEncoding{original: map[string]json.RawMessage{}, normalized: map[string]json.RawMessage{}}
	// Only presence/null differences need metadata. Do not retain another copy
	// of large text, image, or tool-result bodies for already lossless fields.
	for key, baseline := range normalized {
		source, exists := original[key]
		if !exists || (bytes.Equal(bytes.TrimSpace(source), []byte("null")) && !bytes.Equal(source, baseline)) {
			encoding.normalized[key] = baseline
			if exists {
				encoding.original[key] = source
			}
		}
	}
	for key, source := range original {
		if _, exists := normalized[key]; !exists {
			encoding.original[key] = source
		}
	}
	if len(encoding.original) == 0 && len(encoding.normalized) == 0 {
		return nil, nil
	}
	return encoding, nil
}

func restoreTranscriptEncoding(raw []byte, err error, encoding *transcriptEncoding) ([]byte, error) {
	if err != nil || encoding == nil {
		return raw, err
	}
	var current jsonjs.RawObject
	if err := json.Unmarshal(raw, &current); err != nil {
		return nil, err
	}
	for key, baseline := range encoding.normalized {
		if bytes.Equal(current[key], baseline) {
			if original, exists := encoding.original[key]; exists {
				current[key] = original
			} else {
				delete(current, key)
			}
		}
	}
	for key, original := range encoding.original {
		if _, known := encoding.normalized[key]; !known {
			if _, changed := current[key]; !changed {
				current[key] = original
			}
		}
	}
	return json.Marshal(current)
}
