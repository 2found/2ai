package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/2found/2ai/agentcore"
)

const ToolEditFile = "edit_file"

// EditFileTool performs surgical in-place edits: it replaces an exact substring
// in a workspace file rather than rewriting the whole thing (Claude Code's Edit /
// pi's edit). This keeps large files cheap to change and makes intent reviewable —
// the model states the precise text it is swapping. It shares Workspace and the
// same substrate seam as read_file/write_file, so the read-modify-write happens
// wherever those happen — on the host under the symlink/escape guards by
// default, inside the sandbox when one is provided.
type EditFileTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewEditFileTool builds edit_file over the given sandbox. sb is optional: nil
// edits the file on the host filesystem under the Workspace guards (the
// default), non-nil reads and rewrites it from inside the sandbox.
func NewEditFileTool(sb agentcore.Sandbox, workspace *Workspace) *EditFileTool {
	return &EditFileTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *EditFileTool) Name() string { return ToolEditFile }

func (t *EditFileTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolEditFile,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Replace an exact string in a UTF-8 text file inside the agent workspace. " +
			"Pass expected_hash from the latest read_file, write_file, edit_file, or edit_lines result; the edit " +
			"is refused if the file changed since that snapshot. " +
			"old_string must appear exactly once unless replace_all is true; otherwise the edit is " +
			"refused as ambiguous. If no exact match exists, a fuzzy pass retries tolerating smart " +
			"quotes, unicode dashes/spaces, trailing whitespace, and CRLF differences. Use this for " +
			"surgical changes instead of rewriting the whole file with write_file. Paths must be " +
			"relative and cannot escape the workspace.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":          map[string]any{"type": "string", "description": "Workspace-relative file path."},
				"expected_hash": map[string]any{"type": "string", "description": "content_hash from the latest read_file, write_file, edit_file, or edit_lines result for this path."},
				"old_string":    map[string]any{"type": "string", "description": "Exact text to replace. Include enough surrounding context to be unique."},
				"new_string":    map[string]any{"type": "string", "description": "Replacement text. Must differ from old_string."},
				"replace_all": map[string]any{
					"type":        "boolean",
					"description": "Replace every occurrence instead of requiring a unique match. Defaults to false.",
				},
			},
			"required": []string{"path", "expected_hash", "old_string", "new_string"},
		},
	}
}

func (t *EditFileTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Path         string `json:"path"`
		ExpectedHash string `json:"expected_hash"`
		OldString    string `json:"old_string"`
		NewString    string `json:"new_string"`
		ReplaceAll   bool   `json:"replace_all"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("edit_file: invalid arguments: %w", err)
	}
	if in.OldString == in.NewString {
		return "", fmt.Errorf("edit_file: old_string and new_string are identical")
	}
	if in.OldString == "" {
		return "", fmt.Errorf("edit_file: old_string is empty")
	}

	_, rel, err := t.workspace.Resolve(in.Path)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	data, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	expectedHash, err := normalizeFileHash(in.ExpectedHash)
	if err != nil {
		return "", fmt.Errorf("edit_file: expected_hash %s; read %s again and copy its content_hash", err, rel)
	}
	actualHash := fileContentHash(data)
	if expectedHash != actualHash {
		return "", fmt.Errorf("edit_file: stale snapshot for %s: expected %s, current content_hash is %s; re-read the file before editing", rel, expectedHash, actualHash)
	}
	// Match in a canonical view — BOM stripped, LF line endings — and restore
	// both on write, so models never have to reproduce a BOM or CRLF exactly.
	content, hadBOM := stripBOM(string(data))
	crlf := detectCRLF(content)
	content = normalizeToLF(content)
	oldS := normalizeToLF(in.OldString)
	newS := normalizeToLF(in.NewString)

	var updated string
	matched := "exact"
	count := strings.Count(content, oldS)
	switch {
	case count > 1 && !in.ReplaceAll:
		return "", fmt.Errorf("edit_file: old_string appears %d times in %s; add context to make it unique or set replace_all", count, rel)
	case count > 0 && in.ReplaceAll:
		updated = strings.ReplaceAll(content, oldS, newS)
	case count > 0:
		updated = strings.Replace(content, oldS, newS, 1)
	default:
		// No exact match: retry in the fuzzy-normalized view (smart quotes,
		// unicode dashes/spaces, trailing whitespace).
		updated, count, err = fuzzyReplace(content, oldS, newS, rel, in.ReplaceAll)
		if err != nil {
			return "", err
		}
		matched = "fuzzy (normalized quotes/dashes/spaces/trailing whitespace)"
	}

	if crlf {
		updated = strings.ReplaceAll(updated, "\n", "\r\n")
	}
	if hadBOM {
		updated = "\uFEFF" + updated
	}
	updatedBytes := []byte(updated)
	// Matching can be non-trivial for a large file. Revalidate immediately
	// before the write so a concurrent user/tool change made while we prepared
	// the replacement is not silently overwritten. This cannot make arbitrary
	// external filesystem writers transactional, but it reduces the remaining
	// race to the final read/write pair rather than the whole model/tool turn.
	latest, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("edit_file: revalidate %s: %w", rel, err)
	}
	latestHash := fileContentHash(latest)
	if latestHash != actualHash {
		return "", fmt.Errorf("edit_file: file changed while preparing edit for %s: expected %s, current content_hash is %s; re-read the file before editing", rel, actualHash, latestHash)
	}
	if err := t.fs.WriteFile(ctx, rel, updatedBytes); err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	return fmt.Sprintf("path: %s\ncontent_hash: %s\nreplacements: %d\nbytes: %d\nmatch: %s", rel, fileContentHash(updatedBytes), count, len(updatedBytes), matched), nil
}
