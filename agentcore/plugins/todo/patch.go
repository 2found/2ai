package todo

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/2found/2ai/agentcore"
)

// PatchToolName updates stable item IDs without replacing unrelated phases.
const PatchToolName = "patch_plan"

type Operation struct {
	Action string `json:"action"`
	ID     string `json:"id"`
	Item   *Item  `json:"item,omitempty"`
}

type patchTool struct{ store *Store }

func (*patchTool) Name() string      { return PatchToolName }
func (*patchTool) Bookkeeping() bool { return true }
func (*patchTool) Schema() agentcore.ToolSchema {
	item := NewTool(NewStore()).Schema().Parameters["properties"].(map[string]any)["items"].(map[string]any)["items"]
	return agentcore.ToolSchema{Name: PatchToolName, Description: "Atomically add, replace or remove identified plan steps. Supply the complete item for add/replace, with its phase and status. Unknown IDs and invalid plans reject the whole batch.", Parameters: map[string]any{
		"type": "object", "required": []string{"operations"}, "properties": map[string]any{"operations": map[string]any{"type": "array", "minItems": 1, "maxItems": 128, "items": map[string]any{"type": "object", "required": []string{"action", "id"}, "properties": map[string]any{"action": map[string]any{"type": "string", "enum": []string{"add", "replace", "remove"}}, "id": map[string]any{"type": "string", "minLength": 1, "maxLength": 64}, "item": item}}}},
	}}
}
func (t *patchTool) Run(_ context.Context, args string) (string, error) {
	t.store.mutation.Lock()
	defer t.store.mutation.Unlock()
	items, err := patchItems(t.store.List(), args)
	if err != nil {
		return "", err
	}
	t.store.set(items)
	return "Plan updated.\n" + t.store.Render(), nil
}
func patchItems(items []Item, args string) ([]Item, error) {
	var in struct {
		Operations []Operation `json:"operations"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return nil, err
	}
	if len(in.Operations) == 0 || len(in.Operations) > 128 {
		return nil, fmt.Errorf("patch_plan: require 1..128 operations")
	}
	items = append([]Item(nil), items...)
	for _, op := range in.Operations {
		if op.ID == "" || len(op.ID) > 64 {
			return nil, fmt.Errorf("patch_plan: invalid id")
		}
		index := -1
		for i, it := range items {
			if it.ID == op.ID {
				index = i
				break
			}
		}
		switch op.Action {
		case "add", "replace":
			if op.Item == nil || (op.Item.ID != "" && op.Item.ID != op.ID) {
				return nil, fmt.Errorf("patch_plan: matching item required")
			}
			item := *op.Item
			item.ID = op.ID
			if op.Action == "add" {
				if index >= 0 {
					return nil, fmt.Errorf("patch_plan: duplicate id %q", op.ID)
				}
				items = append(items, item)
			} else {
				if index < 0 {
					return nil, fmt.Errorf("patch_plan: unknown id %q", op.ID)
				}
				items[index] = item
			}
		case "remove":
			if index < 0 {
				return nil, fmt.Errorf("patch_plan: unknown id %q", op.ID)
			}
			items = append(items[:index], items[index+1:]...)
		default:
			return nil, fmt.Errorf("patch_plan: unknown action %q", op.Action)
		}
	}
	raw, _ := json.Marshal(struct {
		Items []Item `json:"items"`
	}{items})
	return parseItems(string(raw))
}
