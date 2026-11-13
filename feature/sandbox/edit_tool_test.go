package sandbox

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEditFileReplacesUniqueMatch(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "hello world\n")

	edit := NewEditFileTool(nil, ws)
	out, err := edit.Run(context.Background(), editArgs(t, ws, "a.txt", "world", "there", false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	if !strings.Contains(out, "replacements: 1") {
		t.Fatalf("edit output = %q", out)
	}
	got := mustRead(t, ws, "a.txt")
	if got != "hello there\n" {
		t.Fatalf("content = %q", got)
	}
	if !strings.Contains(out, "content_hash: "+fileContentHash([]byte(got))) {
		t.Fatalf("edit output does not carry the next snapshot hash: %q", out)
	}
}

func TestEditFileSchemaRequiresSnapshotHash(t *testing.T) {
	required, ok := NewEditFileTool(nil, mustWorkspace(t)).Schema().Parameters["required"].([]string)
	if !ok {
		t.Fatal("edit_file required schema is not []string")
	}
	for _, name := range required {
		if name == "expected_hash" {
			return
		}
	}
	t.Fatalf("edit_file schema does not require expected_hash: %v", required)
}

func TestEditFileRejectsAmbiguousMatch(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "x x x")
	edit := NewEditFileTool(nil, ws)
	args := editArgs(t, ws, "a.txt", "x", "y", false)
	if _, err := edit.Run(context.Background(), args); err == nil {
		t.Fatal("expected ambiguous match to fail")
	}
	// replace_all makes it succeed.
	if _, err := edit.Run(context.Background(), editArgs(t, ws, "a.txt", "x", "y", true)); err != nil {
		t.Fatalf("replace_all Run: %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "y y y" {
		t.Fatalf("content = %q", got)
	}
}

func TestEditFileMissingAndIdentical(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "abc")
	edit := NewEditFileTool(nil, ws)
	if _, err := edit.Run(context.Background(), editArgs(t, ws, "a.txt", "zzz", "q", false)); err == nil {
		t.Fatal("expected not-found to fail")
	}
	if _, err := edit.Run(context.Background(), editArgs(t, ws, "a.txt", "abc", "abc", false)); err == nil {
		t.Fatal("expected identical strings to fail")
	}
}

func TestEditFileRejectsStaleSnapshotWithoutMutating(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "mode=safe\ntimeout=30\n")
	staleHash := fileContentHash([]byte(mustRead(t, ws, "a.txt")))

	// A user changes a different line while the model is thinking. The old
	// target still exists, so old_string matching alone would silently apply.
	mustWrite(t, ws, "a.txt", "mode=safe\ntimeout=5\n")
	args, err := json.Marshal(map[string]any{
		"path": "a.txt", "expected_hash": staleHash,
		"old_string": "mode=safe", "new_string": "mode=fast",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	_, err = NewEditFileTool(nil, ws).Run(context.Background(), string(args))
	if err == nil || !strings.Contains(err.Error(), "stale snapshot") || !strings.Contains(err.Error(), "re-read") {
		t.Fatalf("stale edit error = %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "mode=safe\ntimeout=5\n" {
		t.Fatalf("stale edit mutated file: %q", got)
	}
}

func TestEditFileRevalidatesSnapshotImmediatelyBeforeWrite(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "mode=safe\ntimeout=30\n")
	originalHash := fileContentHash([]byte(mustRead(t, ws, "a.txt")))
	base := hostFS{ws: ws}
	guarded := &mutatingWorkspaceFS{
		workspaceFS: base,
		beforeSecondRead: func() {
			mustWrite(t, ws, "a.txt", "mode=safe\ntimeout=5\n")
		},
	}
	edit := &EditFileTool{workspace: ws, fs: guarded}
	args, err := json.Marshal(map[string]any{
		"path": "a.txt", "expected_hash": originalHash,
		"old_string": "mode=safe", "new_string": "mode=fast",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	_, err = edit.Run(context.Background(), string(args))
	if err == nil || !strings.Contains(err.Error(), "changed while preparing edit") {
		t.Fatalf("concurrent edit error = %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "mode=safe\ntimeout=5\n" {
		t.Fatalf("concurrent edit was overwritten: %q", got)
	}
}

func TestEditFileRequiresWellFormedSnapshotHash(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "abc")
	edit := NewEditFileTool(nil, ws)
	for _, args := range []string{
		`{"path":"a.txt","old_string":"a","new_string":"b"}`,
		`{"path":"a.txt","expected_hash":"not-a-hash","old_string":"a","new_string":"b"}`,
	} {
		if _, err := edit.Run(context.Background(), args); err == nil || !strings.Contains(err.Error(), "expected_hash") {
			t.Fatalf("Run(%s) error = %v, want expected_hash diagnostic", args, err)
		}
	}
}

func TestEditFileRejectsEscape(t *testing.T) {
	ws := mustWorkspace(t)
	edit := NewEditFileTool(nil, ws)
	if _, err := edit.Run(context.Background(), `{"path":"../x","old_string":"a","new_string":"b"}`); err == nil {
		t.Fatal("expected escape to fail")
	}
}

// --- shared helpers ---

func mustWorkspace(t *testing.T) *Workspace {
	t.Helper()
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	return ws
}

func mustWrite(t *testing.T, ws *Workspace, rel, content string) {
	t.Helper()
	abs := filepath.Join(ws.Root(), rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func mustRead(t *testing.T, ws *Workspace, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(ws.Root(), rel))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	return string(data)
}

func editArgs(t *testing.T, ws *Workspace, path, oldString, newString string, replaceAll bool) string {
	t.Helper()
	content := mustRead(t, ws, path)
	raw, err := json.Marshal(map[string]any{
		"path": path, "expected_hash": fileContentHash([]byte(content)),
		"old_string": oldString, "new_string": newString, "replace_all": replaceAll,
	})
	if err != nil {
		t.Fatalf("marshal edit args: %v", err)
	}
	return string(raw)
}

// mutatingWorkspaceFS injects a concurrent change between edit_file's initial
// snapshot read and its final pre-write revalidation.
type mutatingWorkspaceFS struct {
	workspaceFS
	reads            int
	beforeSecondRead func()
}

func (f *mutatingWorkspaceFS) ReadFile(ctx context.Context, rel string) ([]byte, error) {
	f.reads++
	if f.reads == 2 && f.beforeSecondRead != nil {
		f.beforeSecondRead()
	}
	return f.workspaceFS.ReadFile(ctx, rel)
}
