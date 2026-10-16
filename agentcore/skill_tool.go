package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// readSkillToolName is the stable name of the built-in progressive-disclosure
// tool. The system prompt advertises skill headers; the model calls this tool to
// pull one skill's full body into context only when the task matches (pi's
// progressive skill disclosure).
const readSkillToolName = "read_skill"

// readSkillTool is a host-injected built-in that materializes a single skill
// body on demand. It is registered automatically by runLoop when the definition
// carries skills, so a consumer never wires it by hand. It resolves an
// advertised identifier (a skill's ID, or its Name when no ID is set) to the
// body — returning a preloaded Body verbatim, or invoking the SkillLoader for
// exactly that one skill.
type readSkillTool struct {
	skills []Skill     // enabled skills, with any preloaded bodies
	loader SkillLoader // optional deferred loader; nil means bodies must be preloaded
}

// withReadSkill returns a copy of base with the built-in read_skill tool added,
// leaving base untouched so the shared Agent toolset is never mutated per run.
func withReadSkill(base *ToolSet, def AgentDefinition) *ToolSet {
	clone := &ToolSet{
		order: append([]string{}, base.order...),
		byKey: make(map[string]Tool, len(base.byKey)+1),
	}
	for k, v := range base.byKey {
		clone.byKey[k] = v
	}
	clone.Add(newReadSkillTool(def))
	return clone
}

// newReadSkillTool builds the tool over the enabled skills of a definition.
func newReadSkillTool(def AgentDefinition) readSkillTool {
	enabled := make([]Skill, 0, len(def.Skills))
	for _, s := range def.Skills {
		if s.Enabled {
			enabled = append(enabled, s)
		}
	}
	return readSkillTool{skills: enabled, loader: def.SkillLoader}
}

func (t readSkillTool) Name() string { return readSkillToolName }

// Parallel marks the tool read-only so a batch of read_skill calls may run
// concurrently with other read-only tools.
func (t readSkillTool) Parallel() bool { return true }

func (t readSkillTool) Schema() ToolSchema {
	return ToolSchema{
		Name:   readSkillToolName,
		Strict: ToolStrictEnabled,
		Description: "Load the full instructions for one of the available skills listed in the system prompt. " +
			"Call this with the skill's id when the current task matches that skill's description, then follow the returned steps.",
		Parameters: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties": map[string]any{
				"id": map[string]any{
					"type":        "string",
					"description": "The id of the skill to load, exactly as advertised in the available skills list.",
				},
			},
			"required": []any{"id"},
		},
	}
}

// readSkillArgs is the decoded argument shape.
type readSkillArgs struct {
	ID string `json:"id"`
}

func (t readSkillTool) Run(ctx context.Context, args string) (string, error) {
	var a readSkillArgs
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	want := strings.TrimSpace(a.ID)
	if want == "" {
		return "", fmt.Errorf("id is required")
	}
	found := t.find(want)
	if found == nil {
		return "", fmt.Errorf("unknown skill %q", want)
	}

	// A preloaded body wins; otherwise pull exactly this one via the loader.
	if body := strings.TrimSpace(found.Body); body != "" {
		return body, nil
	}
	if t.loader == nil {
		return "", fmt.Errorf("skill %q has no body and no loader is configured", want)
	}
	bodies, err := t.loader(ctx, []string{found.ID})
	if err != nil {
		return "", fmt.Errorf("loading skill %q: %w", want, err)
	}
	body := strings.TrimSpace(bodies[found.ID])
	if body == "" {
		return "", fmt.Errorf("skill %q loaded empty", want)
	}
	return body, nil
}

// find resolves an advertised identifier to an enabled skill: on ID first,
// then Name.
func (t readSkillTool) find(want string) *Skill {
	for i := range t.skills {
		if s := &t.skills[i]; skillIdentifier(*s) == want || strings.EqualFold(s.Name, want) {
			return s
		}
	}
	return nil
}

// reads names the enabled skill a read_skill call's arguments would load, so
// the run's tool budget can tell a first read from a repeat (reserveCall).
func (t readSkillTool) reads(args string) (string, bool) {
	var a readSkillArgs
	if json.Unmarshal([]byte(args), &a) != nil {
		return "", false
	}
	if found := t.find(strings.TrimSpace(a.ID)); found != nil {
		return skillIdentifier(*found), true
	}
	return "", false
}

// skillIdentifier is the stable token advertised to the model and accepted by
// read_skill: the skill's ID when set, else its Name. Keeping the two in sync
// here means the prompt and the tool never disagree on what to call a skill.
func skillIdentifier(s Skill) string {
	if id := strings.TrimSpace(s.ID); id != "" {
		return id
	}
	return strings.TrimSpace(s.Name)
}
