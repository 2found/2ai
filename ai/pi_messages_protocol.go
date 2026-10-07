package ai

import (
	"errors"
	"io"
	"strings"
	"time"

	"github.com/2found/2ai/jsonjs"
	"golang.org/x/text/encoding/unicode"
)

// The gateway protocol uses the first data line, not SSE's joined data lines.
// A false yield result releases the reader without consuming the remaining
// events. The HTTP owner is responsible for closing the response body.
func readPiMessagesEvents(reader io.Reader, yield func(any) (bool, error)) error {
	decoder := unicode.UTF8BOM.NewDecoder().Reader(reader)
	buffer := ""
	chunk := make([]byte, 4096)
	parse := func(raw string) (bool, error) {
		data := ""
		for _, line := range strings.Split(raw, "\n") {
			if strings.HasPrefix(line, "data:") {
				data = strings.TrimFunc(line[5:], jsWhitespace)
				break
			}
		}
		if data == "" || data == "[DONE]" {
			return true, nil
		}
		if err := jsonjs.ValidateJSON([]byte(data)); err != nil {
			return false, err
		}
		event, err := jsonjs.DecodeValue([]byte(data))
		if err != nil || !catalogEntryTruthy(event) {
			return err == nil, err
		}
		return yield(event)
	}
	for {
		n, readErr := decoder.Read(chunk)
		buffer = strings.ReplaceAll(buffer+string(chunk[:n]), "\r\n", "\n")
		for {
			index := strings.Index(buffer, "\n\n")
			if index < 0 {
				break
			}
			keepReading, err := parse(buffer[:index])
			if err != nil || !keepReading {
				return err
			}
			buffer = buffer[index+2:]
		}
		if readErr != nil {
			if readErr != io.EOF {
				return readErr
			}
			if strings.TrimFunc(buffer, jsWhitespace) != "" {
				_, err := parse(buffer)
				return err
			}
			return nil
		}
	}
}

type piMessagesToolJSON struct {
	index any
	text  string
}

// Conversion retains the partial message and content objects. Callers serialize
// access when forwarding these live values across goroutines.
type piMessagesEventConverter struct {
	partial *Object
	content *Array
	tools   []piMessagesToolJSON
	now     func() float64
}

func newPiMessagesEventConverter(model *Object, now func() float64) *piMessagesEventConverter {
	if now == nil {
		now = func() float64 { return float64(time.Now().UnixMilli()) }
	}
	content := NewArray()
	usage := NewObject(Property{Name: "input", Value: 0}, Property{Name: "output", Value: 0}, Property{Name: "cacheRead", Value: 0}, Property{Name: "cacheWrite", Value: 0}, Property{Name: "totalTokens", Value: 0}, Property{Name: "cost", Value: NewObject(Property{Name: "input", Value: 0}, Property{Name: "output", Value: 0}, Property{Name: "cacheRead", Value: 0}, Property{Name: "cacheWrite", Value: 0}, Property{Name: "total", Value: 0})})
	return &piMessagesEventConverter{now: now, content: content, partial: NewObject(Property{Name: "role", Value: "assistant"}, Property{Name: "content", Value: content}, Property{Name: "api", Value: catalogProperty(model, "api")}, Property{Name: "provider", Value: catalogProperty(model, "provider")}, Property{Name: "model", Value: catalogProperty(model, "id")}, Property{Name: "usage", Value: usage}, Property{Name: "stopReason", Value: "pending"}, Property{Name: "timestamp", Value: now()})}
}

func (c *piMessagesEventConverter) toolIndex(index any) int {
	for i, entry := range c.tools {
		if catalogSameValueZero(entry.index, index) {
			return i
		}
	}
	return -1
}

func piMessagesAdd(left, right any) (any, error) {
	primitive := func(value any) (any, error) {
		switch value.(type) {
		case *Object, *Array:
			return catalogKey(value, make(map[*Array]bool))
		}
		return value, nil
	}
	a, err := primitive(left)
	if err != nil {
		return nil, err
	}
	b, err := primitive(right)
	if err != nil {
		return nil, err
	}
	_, aString := a.(string)
	_, bString := b.(string)
	if aString || bString {
		x, err := catalogKey(a, make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		y, err := catalogKey(b, make(map[*Array]bool))
		return x + y, err
	}
	x, err := authNumber(a)
	if err != nil {
		return nil, err
	}
	y, err := authNumber(b)
	return x + y, err
}

func (c *piMessagesEventConverter) convert(event any) (*Object, error) {
	if jsonjs.IsNullish(event) {
		return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(event), "event.type")
	}
	kind, _ := catalogProperty(event, "type").(string)
	index := catalogProperty(event, "contentIndex")
	key := ""
	switch kind {
	case "text_start", "text_delta", "text_end", "thinking_start", "thinking_delta", "thinking_end", "toolcall_start", "toolcall_delta", "toolcall_end":
		var err error
		key, err = catalogKey(index, make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
	}
	blockValue := func() any { value, _ := c.content.GetProperty(key); return value }
	block := func() (*Object, error) {
		value := blockValue()
		if jsonjs.IsNullish(value) {
			return nil, errors.New("Object.assign requires that input parameter not be null or undefined")
		}
		valueObject, ok := value.(*Object)
		if !ok {
			return nil, errors.New("pi-messages content is not an object")
		}
		return valueObject, nil
	}
	switch kind {
	case "done", "error":
		c.partial.Set("stopReason", catalogProperty(event, "reason"))
		c.partial.Set("usage", catalogProperty(event, "usage"))
		c.partial.Set("responseId", catalogProperty(event, "responseId"))
		field := "message"
		if kind == "error" {
			field = "error"
			c.partial.Set("errorMessage", catalogProperty(event, "errorMessage"))
		}
		if level := catalogProperty(event, "providerThinkingLevel"); !jsonjs.IsUndefined(level) {
			c.partial.Set("providerThinkingLevel", level)
		}
		if rewrite := catalogProperty(event, "rewrite"); catalogEntryTruthy(rewrite) {
			diagnostics := NewArray()
			if existing, ok := c.partial.Get("diagnostics").(*Array); ok {
				diagnostics = NewArray(existing.Values()...)
			}
			diagnostics.Append(NewObject(Property{Name: "type", Value: "pi_messages_rewrite"}, Property{Name: "timestamp", Value: c.now()}, Property{Name: "details", Value: authSpread(rewrite)}))
			c.partial.Set("diagnostics", diagnostics)
		}
		return NewObject(Property{Name: "type", Value: kind}, Property{Name: "reason", Value: catalogProperty(event, "reason")}, Property{Name: field, Value: c.partial}), nil
	case "text_start", "thinking_start":
		field := "text"
		if kind == "thinking_start" {
			field = "thinking"
		}
		c.content.SetProperty(key, NewObject(Property{Name: "type", Value: field}, Property{Name: field, Value: ""}))
	case "text_delta", "thinking_delta":
		field := "text"
		if kind == "thinking_delta" {
			field = "thinking"
		}
		value := blockValue()
		if jsonjs.IsNullish(value) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(value), "partial.content[event.contentIndex]."+field)
		}
		target, err := block()
		if err != nil {
			return nil, err
		}
		next, err := piMessagesAdd(catalogProperty(target, field), catalogProperty(event, "delta"))
		if err != nil {
			return nil, err
		}
		target.Set(field, next)
	case "text_end", "thinking_end":
		target, err := block()
		if err != nil {
			return nil, err
		}
		field := "text"
		if kind == "thinking_end" {
			field = "thinking"
			target.Set("redacted", catalogProperty(event, "redacted"))
		}
		target.Set(field, catalogProperty(event, "content"))
		target.Set(field+"Signature", catalogProperty(event, "contentSignature"))
	case "toolcall_start":
		c.content.SetProperty(key, NewObject(Property{Name: "type", Value: "toolCall"}, Property{Name: "id", Value: catalogProperty(event, "id")}, Property{Name: "name", Value: catalogProperty(event, "toolName")}, Property{Name: "arguments", Value: NewObject()}))
		i := c.toolIndex(index)
		if i < 0 {
			c.tools = append(c.tools, piMessagesToolJSON{index: index})
		} else {
			c.tools[i].text = ""
		}
	case "toolcall_delta":
		i := c.toolIndex(index)
		if i < 0 {
			c.tools = append(c.tools, piMessagesToolJSON{index: index})
			i = len(c.tools) - 1
		}
		delta, err := catalogKey(catalogProperty(event, "delta"), make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		c.tools[i].text += delta
		if value := blockValue(); jsonjs.IsNullish(value) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(value), "partial.content[event.contentIndex].arguments = parseStreamingJson(json)")
		}
		target, err := block()
		if err != nil {
			return nil, err
		}
		arguments, err := jsonjs.DecodeValue(ParseStreamingJSON(c.tools[i].text))
		if err != nil {
			return nil, err
		}
		target.Set("arguments", arguments)
	case "toolcall_end":
		target, err := block()
		if err != nil {
			return nil, err
		}
		authSpreadInto(target, catalogProperty(event, "toolCall"))
		if i := c.toolIndex(index); i >= 0 {
			c.tools = append(c.tools[:i], c.tools[i+1:]...)
		}
		return NewObject(Property{Name: "type", Value: kind}, Property{Name: "contentIndex", Value: index}, Property{Name: "toolCall", Value: target}, Property{Name: "partial", Value: c.partial}), nil
	}
	result := authSpread(event)
	result.Set("partial", c.partial)
	return result, nil
}
