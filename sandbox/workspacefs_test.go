package sandbox

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

// The file and search tools have two I/O substrates but one implementation, so
// the contract worth testing is that they agree: over the same tree, the
// sandbox path must return byte-identical output to the host path.
//
// HostSandbox is the stand-in backend here. It is a real agentcore.Sandbox — it
// runs the helper scripts through /bin/sh with the workspace mapped to the
// mount — so these tests exercise the actual find/wc/cat helpers and the framed
// stream parser, not a mock of them. What they do not exercise is Docker; the
// container-only concerns (image contents, uid on the bind mount) belong to
// docker_test.go.

// fixtureTree seeds a workspace whose shape stresses the parts of the walk that
// differ between substrates: nested directories, a pruned dependency cache, an
// entry that sorts differently depending on whether you compare whole paths or
// path segments ("src" vs "src.md"), and a non-UTF8 file grep must skip.
func fixtureTree(t *testing.T) *Workspace {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"README.md":                "hello world\nneedle here\n",
		"src.md":                   "needle in a sibling that sorts after the src dir\n",
		"src/a.go":                 "package main\n// needle\nfunc main() {}\n",
		"src/b.go":                 "package main\nno match here\n",
		"src/deep/c.go":            "// needle deep\n",
		"src/deep/nested/d.txt":    "needle\nneedle\nneedle\n",
		"node_modules/pkg/skip.go": "needle that must never be walked\n",
		".git/config":              "needle in vcs metadata\n",
	}
	for rel, content := range files {
		abs := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", rel, err)
		}
		if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", rel, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "binary.bin"), []byte{0xff, 0xfe, 'n', 'e', 0x00}, 0o644); err != nil {
		t.Fatalf("write binary: %v", err)
	}
	ws, err := NewWorkspace(dir)
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	return ws
}

// bothSubstrates runs fn once per substrate and returns (host, sandbox) output.
// The nil is a literal untyped nil, not a typed nil pointer, so the tools take
// the host path rather than an interface holding a nil backend.
func bothSubstrates(t *testing.T, fn func(sb agentcore.Sandbox) (string, error)) (string, string) {
	t.Helper()
	host, err := fn(nil)
	if err != nil {
		t.Fatalf("host substrate: %v", err)
	}
	boxed, err := fn(NewHostSandbox())
	if err != nil {
		t.Fatalf("sandbox substrate: %v", err)
	}
	return host, boxed
}

func TestGrepIdenticalAcrossSubstrates(t *testing.T) {
	ws := fixtureTree(t)
	for _, args := range []string{
		`{"pattern":"needle"}`,
		`{"pattern":"needle","context":1}`,
		`{"pattern":"needle","glob":"**/*.go"}`,
		`{"pattern":"needle","path":"src"}`,
		`{"pattern":"NEEDLE","case_insensitive":true}`,
		`{"pattern":"needle","limit":2}`,
		`{"pattern":"nothing matches this"}`,
	} {
		host, boxed := bothSubstrates(t, func(sb agentcore.Sandbox) (string, error) {
			return NewGrepTool(sb, ws).Run(context.Background(), args)
		})
		if host != boxed {
			t.Errorf("grep %s differs between substrates:\n--- host ---\n%s\n--- sandbox ---\n%s", args, host, boxed)
		}
		if strings.Contains(host, "node_modules") || strings.Contains(host, ".git/") {
			t.Errorf("grep %s walked a pruned directory:\n%s", args, host)
		}
	}
}

func TestGlobIdenticalAcrossSubstrates(t *testing.T) {
	ws := fixtureTree(t)
	for _, args := range []string{
		`{"pattern":"**/*.go"}`,
		`{"pattern":"*.md"}`,
		`{"pattern":"src/**/*.go"}`,
		`{"pattern":"**/*.rs"}`,
		`{"pattern":"**/*.go","path":"src/deep"}`,
	} {
		host, boxed := bothSubstrates(t, func(sb agentcore.Sandbox) (string, error) {
			return NewGlobTool(sb, ws).Run(context.Background(), args)
		})
		if host != boxed {
			t.Errorf("glob %s differs between substrates:\n--- host ---\n%s\n--- sandbox ---\n%s", args, host, boxed)
		}
	}
}

func TestReadFileIdenticalAcrossSubstrates(t *testing.T) {
	ws := fixtureTree(t)
	for _, args := range []string{
		`{"path":"README.md"}`,
		`{"path":"src/deep/nested/d.txt","offset":2}`,
		`{"path":"src/deep/nested/d.txt","offset":1,"limit":2}`,
		`{"path":"src/a.go"}`,
	} {
		host, boxed := bothSubstrates(t, func(sb agentcore.Sandbox) (string, error) {
			return NewReadFileTool(sb, ws).Run(context.Background(), args)
		})
		if host != boxed {
			t.Errorf("read_file %s differs between substrates:\n--- host ---\n%s\n--- sandbox ---\n%s", args, host, boxed)
		}
	}
}

// The error cases have to agree on *whether* they fail, even where the message
// wording comes from a different layer.
func TestReadFileErrorsOnBothSubstrates(t *testing.T) {
	ws := fixtureTree(t)
	for _, args := range []string{
		`{"path":"missing.txt"}`,
		`{"path":"src"}`,           // a directory
		`{"path":"../escape.txt"}`, // traversal
		`{"path":"/etc/passwd"}`,   // absolute
	} {
		for _, sb := range []struct {
			name string
			tool *ReadFileTool
		}{
			{"host", NewReadFileTool(nil, ws)},
			{"sandbox", NewReadFileTool(NewHostSandbox(), ws)},
		} {
			if _, err := sb.tool.Run(context.Background(), args); err == nil {
				t.Errorf("%s substrate: read_file %s should have failed", sb.name, args)
			}
		}
	}
}

func TestWriteAndEditRoundTripOnSandboxSubstrate(t *testing.T) {
	ws := fixtureTree(t)
	sb := NewHostSandbox()

	// A write through the sandbox substrate must land on the real file, creating
	// its parent directories, and be readable back through either substrate.
	body := "line one\nline two\n"
	if _, err := NewWriteFileTool(sb, ws).Run(context.Background(),
		fmt.Sprintf(`{"path":"out/new.txt","content":%q}`, body)); err != nil {
		t.Fatalf("write_file: %v", err)
	}
	onDisk, err := os.ReadFile(filepath.Join(ws.Root(), "out", "new.txt"))
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(onDisk) != body {
		t.Fatalf("written content = %q, want %q", onDisk, body)
	}

	// edit_file reads and rewrites through the same substrate.
	out, err := NewEditFileTool(sb, ws).Run(context.Background(),
		editArgs(t, ws, "out/new.txt", "line two", "line 2", false))
	if err != nil {
		t.Fatalf("edit_file: %v", err)
	}
	if !strings.Contains(out, "replacements: 1") {
		t.Fatalf("edit_file output = %q", out)
	}
	edited, _ := os.ReadFile(filepath.Join(ws.Root(), "out", "new.txt"))
	if string(edited) != "line one\nline 2\n" {
		t.Fatalf("edited content = %q", edited)
	}
}

// Binary content must survive the framed stdin/stdout transfer intact — the
// reason the helpers frame by byte count instead of splitting on a delimiter.
func TestSandboxSubstratePreservesExactBytes(t *testing.T) {
	ws := fixtureTree(t)
	fs := sandboxFS{sb: NewHostSandbox(), ws: ws}
	// NUL, a lone 0xff, an embedded newline, no trailing newline: everything a
	// delimiter-based transfer would mangle.
	content := []byte("no trailing newline\x00\x01\xffand a \"quote\"\nplus \\backslash")
	if err := fs.WriteFile(context.Background(), "odd.bin", content); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	onDisk, err := os.ReadFile(filepath.Join(ws.Root(), "odd.bin"))
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(onDisk) != string(content) {
		t.Fatalf("write changed the bytes:\n got %q\nwant %q", onDisk, content)
	}
	got, err := fs.ReadFile(context.Background(), "odd.bin")
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("round trip changed the bytes:\n got %q\nwant %q", got, content)
	}
}

// walkOrderLess is the reason the two substrates agree on ordering at all: find
// walks in directory order, WalkDir in per-level sorted order, and only a
// segment-wise comparison reproduces the latter.
func TestWalkOrderLessMatchesWalkDir(t *testing.T) {
	if !walkOrderLess("src/a.go", "src.md") {
		t.Error(`"src/a.go" must sort before "src.md" (segment-wise, like WalkDir)`)
	}
	if walkOrderLess("src.md", "src/a.go") {
		t.Error("comparison must be antisymmetric")
	}
	if !walkOrderLess("a", "a/b") {
		t.Error("a shorter path must sort before its own prefix extension")
	}
	if !walkOrderLess("a/b.go", "a/c.go") {
		t.Error("same-directory entries sort by name")
	}
}

// The snapshot budget must fail loudly with an actionable message rather than
// silently returning a partial tree that grep would report as "no matches".
func TestSandboxReadEachRefusesOversizedSnapshot(t *testing.T) {
	dir := t.TempDir()
	big := strings.Repeat("x", 512*1024)
	for i := 0; i < 4; i++ {
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.txt", i)), []byte(big), 0o644); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	ws, err := NewWorkspace(dir)
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	fs := sandboxFS{sb: NewHostSandbox(), ws: ws}
	rels, err := fs.List(context.Background(), "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	err = fs.readEachWithBudget(context.Background(), rels, maxGrepFileBytes, 1024*1024, func(string, []byte) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "narrow the search") {
		t.Fatalf("oversized snapshot error = %v, want a 'narrow the search' message", err)
	}
}
