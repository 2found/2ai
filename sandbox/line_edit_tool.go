package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

const (
	ToolEditLines = "edit_lines"
	maxLineEdits  = 100
)

// EditLinesTool applies several line-addressed changes to one immutable file
// snapshot. Every coordinate refers to the original snapshot, so an agent can
// change distant regions in one call without recalculating line numbers after
// each hunk. The whole-file hash remains the authority: line numbers make calls
// compact, while the hash prevents them from landing on a different revision.
type EditLinesTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewEditLinesTool builds edit_lines over the same host-or-sandbox filesystem
// seam as read_file, write_file, and edit_file.
func NewEditLinesTool(sb agentcore.Sandbox, workspace *Workspace) *EditLinesTool {
	return &EditLinesTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *EditLinesTool) Name() string { return ToolEditLines }

func (t *EditLinesTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolEditLines,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Apply up to 100 non-overlapping line edits to one UTF-8 workspace file as one guarded update. " +
			"Pass expected_hash from the latest read_file, write_file, edit_file, or edit_lines result. " +
			"All line numbers refer to that original snapshot, not to earlier edits in this call. " +
			"Use replace/delete with inclusive start_line and end_line; insert_before/insert_after " +
			"with line; or append at EOF. Text may contain multiple lines. The tool rejects stale " +
			"snapshots, invalid ranges, overlapping edits, and duplicate inserts at one boundary. " +
			"It preserves the file's BOM and line-ending convention.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":          map[string]any{"type": "string", "description": "Workspace-relative file path."},
				"expected_hash": map[string]any{"type": "string", "description": "content_hash from the latest result for this path."},
				"edits": map[string]any{
					"type":        "array",
					"minItems":    1,
					"maxItems":    maxLineEdits,
					"description": "Changes addressed against the original snapshot. Operations must not overlap.",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"op": map[string]any{
								"type": "string",
								"enum": []string{"replace", "delete", "insert_before", "insert_after", "append"},
							},
							"start_line": map[string]any{"type": "integer", "description": "First original line for replace/delete (1-based, inclusive)."},
							"end_line":   map[string]any{"type": "integer", "description": "Last original line for replace/delete (1-based, inclusive)."},
							"line":       map[string]any{"type": "integer", "description": "Original line for insert_before/insert_after (1-based)."},
							"text":       map[string]any{"type": "string", "description": "New line content for replace/insert/append. A final newline is treated as a delimiter, not an extra blank row."},
						},
						"required": []string{"op"},
					},
				},
			},
			"required": []string{"path", "expected_hash", "edits"},
		},
	}
}

type lineEditRequest struct {
	Path         string         `json:"path"`
	ExpectedHash string         `json:"expected_hash"`
	Edits        []lineEditSpec `json:"edits"`
}

type lineEditSpec struct {
	Op        string `json:"op"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Line      int    `json:"line"`
	Text      string `json:"text"`
}

type compiledLineEdit struct {
	requestIndex int
	start        int // zero-based boundary in the original line slice
	end          int // exclusive; start == end is an insertion boundary
	replacement  []string
}

func (t *EditLinesTool) Run(ctx context.Context, args string) (string, error) {
	var in lineEditRequest
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("edit_lines: invalid arguments: %w", err)
	}
	if len(in.Edits) == 0 {
		return "", fmt.Errorf("edit_lines: edits must contain at least one operation")
	}
	if len(in.Edits) > maxLineEdits {
		return "", fmt.Errorf("edit_lines: too many operations: got %d, maximum is %d", len(in.Edits), maxLineEdits)
	}

	_, rel, err := t.workspace.Resolve(in.Path)
	if err != nil {
		return "", fmt.Errorf("edit_lines: %w", err)
	}
	data, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("edit_lines: %w", err)
	}
	expectedHash, err := normalizeFileHash(in.ExpectedHash)
	if err != nil {
		return "", fmt.Errorf("edit_lines: expected_hash %s; read %s again and copy its content_hash", err, rel)
	}
	actualHash := fileContentHash(data)
	if expectedHash != actualHash {
		return "", fmt.Errorf("edit_lines: stale snapshot for %s: expected %s, current content_hash is %s; re-read the file before editing", rel, expectedHash, actualHash)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("edit_lines: %s is not valid UTF-8", rel)
	}

	content, hadBOM := stripBOM(string(data))
	crlf := detectCRLF(content)
	content = normalizeToLF(content)
	lines, hadFinalNewline := splitEditableLines(content)

	compiled := make([]compiledLineEdit, 0, len(in.Edits))
	for i, edit := range in.Edits {
		ce, err := compileLineEdit(edit, i+1, len(lines))
		if err != nil {
			return "", fmt.Errorf("edit_lines: %w", err)
		}
		compiled = append(compiled, ce)
	}
	if err := rejectConflictingLineEdits(compiled); err != nil {
		return "", fmt.Errorf("edit_lines: %w", err)
	}

	sort.Slice(compiled, func(i, j int) bool {
		if compiled[i].start != compiled[j].start {
			return compiled[i].start < compiled[j].start
		}
		return compiled[i].end < compiled[j].end
	})
	updatedLines := make([]string, 0, len(lines))
	cursor := 0
	for _, edit := range compiled {
		updatedLines = append(updatedLines, lines[cursor:edit.start]...)
		updatedLines = append(updatedLines, edit.replacement...)
		cursor = edit.end
	}
	updatedLines = append(updatedLines, lines[cursor:]...)

	updated := strings.Join(updatedLines, "\n")
	if hadFinalNewline && len(updatedLines) > 0 {
		updated += "\n"
	}
	if updated == content {
		return "", fmt.Errorf("edit_lines: operations make no changes to %s", rel)
	}
	if crlf {
		updated = strings.ReplaceAll(updated, "\n", "\r\n")
	}
	if hadBOM {
		updated = "\uFEFF" + updated
	}
	updatedBytes := []byte(updated)

	// Keep the same lost-update defense as edit_file: expensive preparation is
	// done against one snapshot, then the raw bytes are checked again directly
	// before the write on either filesystem substrate.
	latest, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("edit_lines: revalidate %s: %w", rel, err)
	}
	latestHash := fileContentHash(latest)
	if latestHash != actualHash {
		return "", fmt.Errorf("edit_lines: file changed while preparing edits for %s: expected %s, current content_hash is %s; re-read the file before editing", rel, actualHash, latestHash)
	}
	if err := t.fs.WriteFile(ctx, rel, updatedBytes); err != nil {
		return "", fmt.Errorf("edit_lines: %w", err)
	}
	return fmt.Sprintf("path: %s\ncontent_hash: %s\nedits_applied: %d\nbytes: %d\nlines: %d", rel, fileContentHash(updatedBytes), len(compiled), len(updatedBytes), len(updatedLines)), nil
}

func splitEditableLines(content string) ([]string, bool) {
	if content == "" {
		return nil, false
	}
	hadFinalNewline := strings.HasSuffix(content, "\n")
	lines := strings.Split(content, "\n")
	if hadFinalNewline {
		lines = lines[:len(lines)-1]
	}
	return lines, hadFinalNewline
}

func splitLineEditText(text string) []string {
	text = normalizeToLF(text)
	if strings.HasSuffix(text, "\n") {
		text = strings.TrimSuffix(text, "\n")
	}
	return strings.Split(text, "\n")
}

func compileLineEdit(edit lineEditSpec, requestIndex, lineCount int) (compiledLineEdit, error) {
	op := strings.TrimSpace(edit.Op)
	ce := compiledLineEdit{requestIndex: requestIndex}
	switch op {
	case "replace", "delete":
		if edit.StartLine < 1 || edit.EndLine < edit.StartLine || edit.EndLine > lineCount {
			return ce, fmt.Errorf("operation %d (%s) has invalid range %d-%d for a %d-line file", requestIndex, op, edit.StartLine, edit.EndLine, lineCount)
		}
		if edit.Line != 0 {
			return ce, fmt.Errorf("operation %d (%s) must use start_line/end_line, not line", requestIndex, op)
		}
		ce.start, ce.end = edit.StartLine-1, edit.EndLine
		if op == "replace" {
			ce.replacement = splitLineEditText(edit.Text)
		} else if edit.Text != "" {
			return ce, fmt.Errorf("operation %d (delete) must not include text", requestIndex)
		}
	case "insert_before", "insert_after":
		if edit.Line < 1 || edit.Line > lineCount {
			return ce, fmt.Errorf("operation %d (%s) has invalid line %d for a %d-line file", requestIndex, op, edit.Line, lineCount)
		}
		if edit.StartLine != 0 || edit.EndLine != 0 {
			return ce, fmt.Errorf("operation %d (%s) must use line, not start_line/end_line", requestIndex, op)
		}
		ce.start = edit.Line - 1
		if op == "insert_after" {
			ce.start = edit.Line
		}
		ce.end = ce.start
		ce.replacement = splitLineEditText(edit.Text)
	case "append":
		if edit.Line != 0 || edit.StartLine != 0 || edit.EndLine != 0 {
			return ce, fmt.Errorf("operation %d (append) does not take line coordinates", requestIndex)
		}
		ce.start, ce.end = lineCount, lineCount
		ce.replacement = splitLineEditText(edit.Text)
	default:
		return ce, fmt.Errorf("operation %d has unsupported op %q", requestIndex, edit.Op)
	}
	return ce, nil
}

func rejectConflictingLineEdits(edits []compiledLineEdit) error {
	for i := 0; i < len(edits); i++ {
		for j := i + 1; j < len(edits); j++ {
			if lineEditsConflict(edits[i], edits[j]) {
				return fmt.Errorf("operations %d and %d overlap or use the same insertion boundary", edits[i].requestIndex, edits[j].requestIndex)
			}
		}
	}
	return nil
}

func lineEditsConflict(a, b compiledLineEdit) bool {
	aPoint := a.start == a.end
	bPoint := b.start == b.end
	switch {
	case aPoint && bPoint:
		return a.start == b.start
	case aPoint:
		return a.start > b.start && a.start < b.end
	case bPoint:
		return b.start > a.start && b.start < a.end
	default:
		return a.start < b.end && b.start < a.end
	}
}
