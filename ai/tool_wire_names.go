package ai

import (
	"bytes"
	"encoding/json"

	"github.com/2found/2ai/ai/protocol"
)

// maxWireToolName is the shortest function-name limit among the
// OpenAI-compatible backends a gateway may route to.
const maxWireToolName = 64

// toolWireNames maps the tool names a caller uses onto names every
// OpenAI-compatible backend accepts, and maps the model's calls back.
//
// Callers name tools freely ("2found-hub_cms_save"), but Gemini-class
// backends require a function name to start with a letter or underscore, and a
// gateway routing to one rewrites the name on the way in without restoring it
// on the way out: the model then calls "_2found-hub_cms_save", which no tool
// answers. Sending an already-portable name leaves a gateway nothing to
// rewrite, and translating the reply here keeps the rename invisible to the
// caller. A portable name passes through unchanged.
type toolWireNames struct {
	wire      map[string]string
	canonical map[string]string
}

func newToolWireNames(req protocol.ChatRequest) toolWireNames {
	var names []string
	for _, tool := range req.Tools {
		names = append(names, tool.Name)
	}
	for _, message := range req.Messages {
		for _, call := range message.ToolCalls {
			names = append(names, call.Name)
		}
		if message.Role == protocol.RoleTool {
			names = append(names, message.Name)
		}
	}
	if req.ToolChoice.Mode == protocol.ToolChoiceNamed {
		names = append(names, req.ToolChoice.Name)
	}
	return toolWireNamesFor(names)
}

// toolWireNamesFor gives every name that needs one a distinct portable name.
func toolWireNamesFor(names []string) toolWireNames {
	taken := map[string]bool{}
	for _, name := range names {
		taken[name] = true
	}
	out := toolWireNames{}
	for _, name := range names {
		if name == "" || out.wire[name] != "" {
			continue
		}
		wire := portableToolName(name)
		if wire == name {
			continue
		}
		if taken[wire] {
			wire = portableToolName(name + "_" + completionsShortHash(name))
		}
		if out.wire == nil {
			out.wire, out.canonical = map[string]string{}, map[string]string{}
		}
		taken[wire] = true
		out.wire[name], out.canonical[wire] = wire, name
	}
	return out
}

// toWire is the name sent for a caller's tool.
func (n toolWireNames) toWire(name string) string {
	if wire, ok := n.wire[name]; ok {
		return wire
	}
	return name
}

// fromWire is the caller's name for a tool the model called.
func (n toolWireNames) fromWire(name string) string {
	if canonical, ok := n.canonical[name]; ok {
		return canonical
	}
	return name
}

// portableToolName keeps [A-Za-z0-9_-], starts with a letter or underscore and
// fits maxWireToolName; anything longer keeps a hash of the original so two
// long names cannot meet.
func portableToolName(name string) string {
	out := make([]byte, 0, len(name)+1)
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-' {
			out = append(out, c)
		} else {
			out = append(out, '_')
		}
	}
	if len(out) == 0 || !(out[0] >= 'a' && out[0] <= 'z' || out[0] >= 'A' && out[0] <= 'Z' || out[0] == '_') {
		out = append([]byte{'_'}, out...)
	}
	if len(out) > maxWireToolName {
		hash := completionsShortHash(name)
		if len(hash) > 8 {
			hash = hash[:8]
		}
		out = append(out[:maxWireToolName-len(hash)-1], '_')
		out = append(out, hash...)
	}
	return string(out)
}

// portableCompletionsToolNames applies toolWireNames to a built Chat
// Completions body: declared tools (top level and those a system message
// anchors), earlier calls, tool results and a named tool_choice. A body whose
// names are already portable is returned byte for byte.
func portableCompletionsToolNames(params json.RawMessage) (json.RawMessage, toolWireNames, error) {
	var names []string
	if _, err := walkCompletionsToolNames(params, func(name string) string { names = append(names, name); return name }); err != nil {
		return nil, toolWireNames{}, err
	}
	codec := toolWireNamesFor(names)
	if len(codec.wire) == 0 {
		return params, codec, nil
	}
	out, err := walkCompletionsToolNames(params, codec.toWire)
	return out, codec, err
}

func walkCompletionsToolNames(params json.RawMessage, rename func(string) string) (json.RawMessage, error) {
	return editObject(params, func(body map[string]json.RawMessage) error {
		var err error
		if body["tools"], err = editArray(body["tools"], func(tool json.RawMessage) (json.RawMessage, error) {
			return renameCallee(tool, rename)
		}); err != nil {
			return err
		}
		if body["tool_choice"], err = renameCallee(body["tool_choice"], rename); err != nil {
			return err
		}
		body["messages"], err = editArray(body["messages"], func(message json.RawMessage) (json.RawMessage, error) {
			return editObject(message, func(m map[string]json.RawMessage) error {
				var err error
				if m["tools"], err = editArray(m["tools"], func(tool json.RawMessage) (json.RawMessage, error) {
					return renameCallee(tool, rename)
				}); err != nil {
					return err
				}
				if m["tool_calls"], err = editArray(m["tool_calls"], func(call json.RawMessage) (json.RawMessage, error) {
					return renameCallee(call, rename)
				}); err != nil {
					return err
				}
				if samplingString(m["role"]) == "tool" {
					m["name"], err = renameString(m["name"], rename)
				}
				return err
			})
		})
		return err
	})
}

// renameCallee renames {"function":{"name"}} and {"custom":{"name"}}.
func renameCallee(raw json.RawMessage, rename func(string) string) (json.RawMessage, error) {
	return editObject(raw, func(holder map[string]json.RawMessage) error {
		for _, key := range []string{"function", "custom"} {
			var err error
			if holder[key], err = editObject(holder[key], func(callee map[string]json.RawMessage) error {
				var err error
				callee["name"], err = renameString(callee["name"], rename)
				return err
			}); err != nil {
				return err
			}
		}
		return nil
	})
}

func renameString(raw json.RawMessage, rename func(string) string) (json.RawMessage, error) {
	var name string
	if len(raw) == 0 || json.Unmarshal(raw, &name) != nil {
		return raw, nil
	}
	if next := rename(name); next != name {
		return json.Marshal(next)
	}
	return raw, nil
}

// editObject and editArray leave anything that is not an object or array, and
// every value edit leaves alone, exactly as it was.
func editObject(raw json.RawMessage, edit func(map[string]json.RawMessage) error) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return raw, nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &object); err != nil {
		return nil, err
	}
	before := make(map[string]string, len(object))
	for key, value := range object {
		before[key] = string(value)
	}
	if err := edit(object); err != nil {
		return nil, err
	}
	changed := false
	for key, value := range object {
		if value == nil {
			if _, ok := before[key]; ok {
				object[key] = json.RawMessage(before[key])
			} else {
				delete(object, key)
			}
			continue
		}
		if string(value) != before[key] {
			changed = true
		}
	}
	if !changed {
		return raw, nil
	}
	return json.Marshal(object)
}

func editArray(raw json.RawMessage, edit func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return raw, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return nil, err
	}
	changed := false
	for i, item := range items {
		next, err := edit(item)
		if err != nil {
			return nil, err
		}
		if string(next) != string(item) {
			items[i], changed = next, true
		}
	}
	if !changed {
		return raw, nil
	}
	return json.Marshal(items)
}
