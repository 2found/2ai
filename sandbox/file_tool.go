package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

const (
	ToolReadFile  = "read_file"
	ToolWriteFile = "write_file"
)

const (
	// maxReadFileBytes is the byte budget for one read window. The window is
	// selected first (offset/limit), then trimmed to this budget — so paging with
	// offset can always reach every line of the file.
	maxReadFileBytes = 64 * 1024
	// maxReadWholeFileBytes bounds how large a file the tool will load at all;
	// beyond this the caller is told to grep for the region instead.
	maxReadWholeFileBytes = 16 * 1024 * 1024
)

// ReadFileTool reads a workspace file. sb decides where the read happens, not
// what it produces: the windowing, line numbering and truncation notice below
// run identically over either substrate.
type ReadFileTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewReadFileTool builds read_file over the given sandbox. sb is optional: nil
// reads the file directly from the host filesystem under the Workspace guards
// (the default), non-nil reads it from inside the sandbox with the workspace
// bind-mounted.
func NewReadFileTool(sb agentcore.Sandbox, workspace *Workspace) *ReadFileTool {
	return &ReadFileTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *ReadFileTool) Name() string { return ToolReadFile }

func (t *ReadFileTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolReadFile,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Read a UTF-8 text file from the agent workspace. Content is returned with " +
			"cat -n style line numbers so you can cite exact lines. Use offset and limit to read a " +
			"window of a large file; a truncated read ends with the exact offset to continue from. " +
			"The result includes a content_hash required by edit_file and edit_lines to reject stale edits. " +
			"Use byte_offset and byte_limit to page through files above 16MB or the suffix of an oversized line. " +
			"The path must be relative and cannot escape the workspace.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "Workspace-relative file path."},
				"offset": map[string]any{
					"type":        "integer",
					"description": "1-based line number to start reading from. Defaults to 1.",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Maximum number of lines to return. Defaults to the whole file (capped for size).",
				},
				"byte_offset": map[string]any{"type": "integer", "minimum": 0, "description": "0-based byte offset; selects byte paging instead of line paging."},
				"byte_limit":  map[string]any{"type": "integer", "minimum": 1, "maximum": maxReadFileBytes, "description": "Byte page size, at most 64KB. Requires byte_offset."},
			},
			"required": []string{"path"},
		},
	}
}

func (t *ReadFileTool) Parallel() bool { return true }

func (t *ReadFileTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Path       string `json:"path"`
		Offset     int    `json:"offset"`
		Limit      int    `json:"limit"`
		ByteOffset *int64 `json:"byte_offset"`
		ByteLimit  int    `json:"byte_limit"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("read_file: invalid arguments: %w", err)
	}
	_, rel, err := t.workspace.Resolve(in.Path)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	info, err := t.fs.Stat(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	if info.IsDir {
		return "", fmt.Errorf("read_file: %s is a directory", rel)
	}
	if in.ByteOffset != nil {
		if *in.ByteOffset < 0 || *in.ByteOffset > info.Size || in.Offset != 0 || in.Limit != 0 {
			return "", fmt.Errorf("read_file: byte_offset must be within file bounds and cannot combine with line offset/limit")
		}
		limit := in.ByteLimit
		if limit <= 0 {
			limit = maxReadFileBytes
		}
		limit = min(limit, maxReadFileBytes)
		// Include three preceding bytes so a requested offset in a UTF-8 rune
		// can snap back to the start rather than losing part of the character.
		start := max(int64(0), *in.ByteOffset-3)
		data, hash, err := t.fs.ReadRange(ctx, rel, start, limit+7)
		if err != nil {
			return "", fmt.Errorf("read_file: %w", err)
		}
		index := int(*in.ByteOffset - start)
		for index > 0 && index < len(data) && data[index]&0xc0 == 0x80 {
			index--
		}
		data = data[min(index, len(data)):]
		start += int64(index)
		content := clampUTF8(string(data), limit)
		if content == "" && len(data) > 0 {
			_, size := utf8.DecodeRune(data)
			content = string(data[:size])
		}
		if !utf8.ValidString(content) {
			return "", fmt.Errorf("read_file: byte window is not UTF-8 text")
		}
		next := start + int64(len(content))
		partial := start > 0 || next < info.Size
		if err := t.workspace.recordRead(rel, hash, partial); err != nil {
			return "", err
		}
		return fmt.Sprintf("path: %s\ncontent_hash: %s\nbytes: %d\nbyte_offset: %d\nnext_byte_offset: %d\ntruncated: %t\ncontent:\n%s\n[Use byte_offset=%d to continue.]", rel, hash, info.Size, start, next, partial, content, next), nil
	}
	if in.ByteLimit != 0 {
		return "", fmt.Errorf("read_file: byte_limit requires byte_offset")
	}
	if info.Size > maxReadWholeFileBytes {
		return "", fmt.Errorf("read_file: %s is %d bytes, over the %dMB line-read cap — use byte_offset=0 and byte_limit=65536 to page the file",
			rel, info.Size, maxReadWholeFileBytes/(1024*1024))
	}
	data, err := t.fs.ReadFile(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("read_file: %s is not UTF-8 text", rel)
	}

	// Window the requested line range FIRST (1-based offset, optional limit),
	// then trim the selected window to the byte budget — never the file as a
	// whole, so offset-paging can reach every line no matter the file size.
	lines := strings.Split(string(data), "\n")
	// A trailing newline yields a final empty element; drop it so the line count
	// and numbering match cat -n rather than reporting a phantom blank line.
	if len(lines) > 1 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	total := len(lines)
	start := in.Offset
	if start < 1 {
		start = 1
	}
	if start > total {
		return "", fmt.Errorf("read_file: offset %d is beyond end of file (%d lines)", in.Offset, total)
	}
	end := total
	if in.Limit > 0 && start+in.Limit-1 < end {
		end = start + in.Limit - 1
	}

	// Emit numbered lines cat -n style until the byte budget runs out. Always
	// emit at least one line so a single oversized line can't yield an empty
	// window — it is clamped to the budget instead.
	var content strings.Builder
	clamped := false
	var nextByte int
	for _, line := range lines[:start-1] {
		nextByte += len(line) + 1
	}
	last := start - 1
	for i := start; i <= end; i++ {
		line := lines[i-1]
		if content.Len()+len(line) > maxReadFileBytes {
			if i == start {
				fmt.Fprintf(&content, "%6d\t%s\n", i, clampUTF8(line, maxReadFileBytes))
				fmt.Fprintf(&content, "[line %d is %d bytes; showing its first %dKB]\n", i, len(line), maxReadFileBytes/1024)
				last = i
				clamped = true
				nextByte += len(clampUTF8(line, maxReadFileBytes))
			}
			break
		}
		fmt.Fprintf(&content, "%6d\t%s\n", i, line)
		last = i
		nextByte += len(line) + 1
	}

	var b strings.Builder
	fmt.Fprintf(&b, "path: %s\ncontent_hash: %s\nbytes: %d\nlines: %d", rel, fileContentHash(data), info.Size, total)
	partial := last < total || start > 1 || clamped
	if err := t.workspace.recordRead(rel, fileContentHash(data), partial); err != nil {
		return "", err
	}
	if partial {
		b.WriteString("\ntruncated: true")
	}
	b.WriteString("\ncontent:\n")
	b.WriteString(content.String())
	if clamped {
		fmt.Fprintf(&b, "\n[Use byte_offset=%d and byte_limit=65536 to read the rest of this line.]", nextByte)
	}
	if last < total {
		fmt.Fprintf(&b, "\n[Showing lines %d-%d of %d. Use offset=%d to continue.]", start, last, total, last+1)
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

// clampUTF8 cuts s to at most max bytes without splitting a UTF-8 sequence.
func clampUTF8(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

type WriteFileTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewWriteFileTool builds write_file over the given sandbox. sb is optional: nil
// writes directly to the host filesystem under the Workspace guards (the
// default), non-nil writes from inside the sandbox with the workspace
// bind-mounted — where the content rides stdin, never argv or env.
func NewWriteFileTool(sb agentcore.Sandbox, workspace *Workspace) *WriteFileTool {
	return &WriteFileTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *WriteFileTool) Name() string { return ToolWriteFile }

func (t *WriteFileTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:        ToolWriteFile,
		Description: "Write a UTF-8 text file inside the agent workspace. The result includes a content_hash for a subsequent edit_file or edit_lines call. Parent directories are created; paths must be relative and cannot escape the workspace.",
		Strict:      agentcore.ToolStrictEnabled,
		Parameters: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties": map[string]any{
				"path":                    map[string]any{"type": "string", "description": "Workspace-relative file path."},
				"content":                 map[string]any{"type": "string", "description": "Complete file content to write."},
				"expected_hash":           map[string]any{"type": "string", "description": "Required with allow_partial_overwrite; hash of the current complete file."},
				"allow_partial_overwrite": map[string]any{"type": "boolean", "description": "Explicitly confirm a full-file replacement after a partial read. Requires expected_hash. Prefer edit_file/edit_lines for partial reads."},
			},
			"required": []string{"path", "content"},
		},
	}
}

func (t *WriteFileTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Path                  string `json:"path"`
		Content               string `json:"content"`
		ExpectedHash          string `json:"expected_hash"`
		AllowPartialOverwrite bool   `json:"allow_partial_overwrite"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("write_file: invalid arguments: %w", err)
	}
	_, rel, err := t.workspace.Resolve(in.Path)
	if err != nil {
		return "", fmt.Errorf("write_file: %w", err)
	}
	if t.workspace.partialRead(rel) != "" && !in.AllowPartialOverwrite {
		return "", fmt.Errorf("write_file: %s was only partially read; use edit_file/edit_lines, read the full file, or explicitly confirm allow_partial_overwrite with expected_hash", rel)
	}
	if in.ExpectedHash != "" || in.AllowPartialOverwrite {
		hash, err := normalizeFileHash(in.ExpectedHash)
		if err != nil {
			return "", fmt.Errorf("write_file: expected_hash %w", err)
		}
		_, actual, err := t.fs.ReadRange(ctx, rel, 0, 0)
		if err != nil {
			return "", err
		}
		if hash != actual {
			return "", fmt.Errorf("write_file: stale snapshot for %s; re-read the file", rel)
		}
	}
	if err := t.fs.WriteFile(ctx, rel, []byte(in.Content)); err != nil {
		return "", fmt.Errorf("write_file: %w", err)
	}
	_ = t.workspace.recordRead(rel, "", false)
	return fmt.Sprintf("path: %s\ncontent_hash: %s\nbytes_written: %d", rel, fileContentHash([]byte(in.Content)), len(in.Content)), nil
}
