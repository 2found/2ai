package ai

import "github.com/2found/2ai/ai/protocol"

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
