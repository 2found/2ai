package ai

import "github.com/2found/2ai/ai/protocol"

func cloneOptionalBool(value *bool, toolsPresent bool) *bool {
	if value == nil || !toolsPresent {
		return nil
	}
	cloned := *value
	return &cloned
}

func openAIChatToolChoice(choice protocol.ToolChoice, toolsPresent bool) any {
	if !toolsPresent {
		return nil
	}
	switch choice.Mode {
	case protocol.ToolChoiceAuto, protocol.ToolChoiceNone, protocol.ToolChoiceRequired:
		return string(choice.Mode)
	case protocol.ToolChoiceNamed:
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": choice.Name},
		}
	default:
		return nil
	}
}

func openAIResponsesToolChoice(choice protocol.ToolChoice, toolsPresent bool) any {
	if !toolsPresent {
		return nil
	}
	switch choice.Mode {
	case protocol.ToolChoiceAuto, protocol.ToolChoiceNone, protocol.ToolChoiceRequired:
		return string(choice.Mode)
	case protocol.ToolChoiceNamed:
		return map[string]any{"type": "function", "name": choice.Name}
	default:
		return nil
	}
}

func anthropicToolChoice(choice protocol.ToolChoice, parallel *bool, toolsPresent bool) *antToolChoice {
	if !toolsPresent {
		return nil
	}
	var out *antToolChoice
	switch choice.Mode {
	case protocol.ToolChoiceAuto:
		out = &antToolChoice{Type: "auto"}
	case protocol.ToolChoiceNone:
		out = &antToolChoice{Type: "none"}
	case protocol.ToolChoiceRequired:
		out = &antToolChoice{Type: "any"}
	case protocol.ToolChoiceNamed:
		out = &antToolChoice{Type: "tool", Name: choice.Name}
	}
	if parallel != nil {
		if out == nil {
			out = &antToolChoice{Type: "auto"}
		}
		disabled := !*parallel
		out.DisableParallelToolUse = &disabled
	}
	return out
}

func googleToolChoice(choice protocol.ToolChoice, toolsPresent bool) *agFunctionCallingConfig {
	if !toolsPresent {
		return nil
	}
	switch choice.Mode {
	case protocol.ToolChoiceAuto:
		return &agFunctionCallingConfig{Mode: "AUTO"}
	case protocol.ToolChoiceNone:
		return &agFunctionCallingConfig{Mode: "NONE"}
	case protocol.ToolChoiceRequired:
		return &agFunctionCallingConfig{Mode: "ANY"}
	case protocol.ToolChoiceNamed:
		return &agFunctionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{choice.Name}}
	default:
		return nil
	}
}
