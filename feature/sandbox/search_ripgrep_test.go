package sandbox

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// requireRipgrep returns the rg binary or skips: the ripgrep engine is optional
// and the Go engine covers machines without it.
func requireRipgrep(t *testing.T) string {
	t.Helper()
	path, err := exec.LookPath("rg")
	if err != nil {
		t.Skip("rg is not on PATH")
	}
	return path
}

// The two engines are one tool: every row, page and notice must be the same.
func TestGrepRipgrepMatchesTheGoEngine(t *testing.T) {
	rg := requireRipgrep(t)
	ws := fixtureTree(t)
	mustWrite(t, ws, "ignored/.gitignore", "*.log\n!keep.log\n")
	mustWrite(t, ws, "ignored/drop.log", "needle dropped\n")
	mustWrite(t, ws, "ignored/keep.log", "needle kept\n")
	mustWrite(t, ws, ".hidden/h.txt", "needle hidden\n")
	mustWrite(t, ws, "crlf.txt", "needle\r\nline two\r\n")
	mustWrite(t, ws, "many.txt", strings.Repeat("needle row\nfiller\n", 30))
	mustWrite(t, ws, "src/span.go", "package src\n\nfunc Spread(a int) {\n\tneedle(a)\n}\n\nfunc Other() {}\n")
	mustWrite(t, ws, "web/app.TS", "const needle = 1\n")
	for _, args := range []string{
		`{"pattern":"func Spread.*?\\n}","multiline":true}`,
		`{"pattern":"func Spread.*?\\n}","multiline":true,"context":1}`,
		`{"pattern":"func Spread.*?\\n}","multiline":true,"output_mode":"count"}`,
		`{"pattern":"needle\\(a\\)\\n}\\n\\nfunc","multiline":true,"limit":2}`,
		`{"pattern":"^func","multiline":true}`,
		`{"pattern":"needle","type":"go"}`,
		`{"pattern":"needle","type":"ts"}`,
		`{"pattern":"needle","type":"md","output_mode":"files_with_matches"}`,
		`{"pattern":"needle"}`,
		`{"pattern":"needle","context":1}`,
		`{"pattern":"needle","context":3,"limit":5}`,
		`{"pattern":"needle","glob":"**/*.go"}`,
		`{"pattern":"needle","glob":"*.md"}`,
		`{"pattern":"needle","path":"src"}`,
		`{"pattern":"needle","path":"src/deep"}`,
		`{"pattern":"NEEDLE","case_insensitive":true}`,
		`{"pattern":"needle","limit":2}`,
		`{"pattern":"needle","limit":3,"offset":4}`,
		`{"pattern":"needle","offset":1000}`,
		`{"pattern":"a.go","literal":true}`,
		`{"pattern":"^needle$"}`,
		`{"pattern":"needle","output_mode":"files_with_matches"}`,
		`{"pattern":"needle","output_mode":"files_with_matches","limit":2,"offset":1}`,
		`{"pattern":"needle","output_mode":"count"}`,
		`{"pattern":"needle","output_mode":"count","limit":1}`,
		`{"pattern":"nothing matches this"}`,
	} {
		fast, err := NewGrepTool(nil, ws, WithGrepRipgrep(rg)).Run(context.Background(), args)
		if err != nil {
			t.Fatalf("ripgrep %s: %v", args, err)
		}
		slow, err := NewGrepTool(nil, ws, WithGrepRipgrep("")).Run(context.Background(), args)
		if err != nil {
			t.Fatalf("go %s: %v", args, err)
		}
		if fast != slow {
			t.Errorf("grep %s differs between engines:\n--- ripgrep ---\n%s\n--- go ---\n%s", args, fast, slow)
		}
		if strings.Contains(fast, "drop.log") || strings.Contains(fast, "node_modules") || strings.Contains(fast, ".git/") {
			t.Errorf("grep %s searched an ignored path:\n%s", args, fast)
		}
	}
}

// A scope path that is a symlink out of the workspace must not let rg follow
// it; the call goes to the Go engine, which refuses it.
func TestGrepRipgrepNeverFollowsAScopeSymlinkOut(t *testing.T) {
	rg := requireRipgrep(t)
	ws := mustWorkspace(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("needle secret\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(ws.Root(), "link")); err != nil {
		t.Fatal(err)
	}
	out, err := NewGrepTool(nil, ws, WithGrepRipgrep(rg)).Run(context.Background(), `{"pattern":"needle","path":"link"}`)
	if err == nil && strings.Contains(out, "secret") {
		t.Fatalf("grep followed a symlink out of the workspace: %q", out)
	}
	whole, err := NewGrepTool(nil, ws, WithGrepRipgrep(rg)).Run(context.Background(), `{"pattern":"needle"}`)
	if err != nil || strings.Contains(whole, "secret") {
		t.Fatalf("a workspace search followed a symlink: %q %v", whole, err)
	}
}

// A pattern only RE2 accepts, or a broken rg, falls back to the Go engine
// instead of failing the call.
func TestGrepRipgrepFailureFallsBackToGo(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "needle\n")
	broken := filepath.Join(t.TempDir(), "rg")
	if err := os.WriteFile(broken, []byte("#!/bin/sh\nexit 2\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	out, err := NewGrepTool(nil, ws, WithGrepRipgrep(broken)).Run(context.Background(), `{"pattern":"needle"}`)
	if err != nil || out != "a.txt:1:needle" {
		t.Fatalf("fallback = %q, %v", out, err)
	}
}

// Paging through a file with more matches than a page holds stays exact.
func TestGrepOffsetPagesThroughOneLargeFile(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "big.txt", strings.Repeat("hit\n", 450))
	tool := NewGrepTool(nil, ws)
	first, _ := tool.Run(context.Background(), `{"pattern":"hit"}`)
	if !strings.Contains(first, "big.txt:200:hit") || !strings.Contains(first, "offset=200") {
		t.Fatalf("first page = %q", first[len(first)-200:])
	}
	last, _ := tool.Run(context.Background(), `{"pattern":"hit","offset":400}`)
	if !strings.HasPrefix(last, "big.txt:401:hit") || !strings.HasSuffix(last, "big.txt:450:hit") || strings.Contains(last, "truncated") {
		t.Fatalf("last page = %q", last)
	}
}

// ripgrep streams a file the Go engine would skip at 2MB, so a needle at its
// end is found and no coverage notice is printed.
func TestGrepRipgrepSearchesPastTheGoFileBound(t *testing.T) {
	rg := requireRipgrep(t)
	ws := mustWorkspace(t)
	mustWrite(t, ws, "huge.txt", strings.Repeat("filler\n", 310000)+"NEEDLE_AT_END\n")
	out, err := NewGrepTool(nil, ws, WithGrepRipgrep(rg)).Run(context.Background(), `{"pattern":"NEEDLE_AT_END"}`)
	if err != nil || out != "huge.txt:310001:NEEDLE_AT_END" {
		t.Fatalf("ripgrep over a large file = %q, %v", out, err)
	}
}
