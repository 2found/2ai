package sandbox

import (
	"context"
	"strings"
	"testing"
)

func TestGrepFindsMatchesWithLocation(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "src/a.go", "package a\nfunc Foo() {}\n")
	mustWrite(t, ws, "src/b.txt", "nothing here\nFoo mention\n")

	grep := NewGrepTool(nil, ws)
	out, err := grep.Run(context.Background(), `{"pattern":"Foo"}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if !strings.Contains(out, "src/a.go:2:") || !strings.Contains(out, "src/b.txt:2:") {
		t.Fatalf("grep output = %q", out)
	}
}

func TestGrepGlobFilterAndCaseInsensitive(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.go", "TODO fix\n")
	mustWrite(t, ws, "b.md", "todo doc\n")

	grep := NewGrepTool(nil, ws)
	out, err := grep.Run(context.Background(), `{"pattern":"todo","glob":"*.go","case_insensitive":true}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if !strings.Contains(out, "a.go") || strings.Contains(out, "b.md") {
		t.Fatalf("glob filter failed: %q", out)
	}
}

func TestGrepNoMatch(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "hello")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"zzz"}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if out != "no matches" {
		t.Fatalf("expected no matches, got %q", out)
	}
}

func TestGlobMatchesDoublestar(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "src/deep/x.go", "")
	mustWrite(t, ws, "src/y.go", "")
	mustWrite(t, ws, "z.txt", "")

	glob := NewGlobTool(nil, ws)
	out, err := glob.Run(context.Background(), `{"pattern":"**/*.go"}`)
	if err != nil {
		t.Fatalf("glob Run: %v", err)
	}
	if !strings.Contains(out, "src/deep/x.go") || !strings.Contains(out, "src/y.go") {
		t.Fatalf("glob output = %q", out)
	}
	if strings.Contains(out, "z.txt") {
		t.Fatalf("glob matched non-go file: %q", out)
	}
}

func TestGlobSingleStarDoesNotCrossSlash(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.go", "")
	mustWrite(t, ws, "sub/b.go", "")

	out, err := NewGlobTool(nil, ws).Run(context.Background(), `{"pattern":"*.go"}`)
	if err != nil {
		t.Fatalf("glob Run: %v", err)
	}
	if !strings.Contains(out, "a.go") || strings.Contains(out, "sub/b.go") {
		t.Fatalf("single-star crossed slash: %q", out)
	}
}

// TestGrepContextLines pins grep -C style output: context lines with '-'
// separators around ':' match rows, overlapping windows merged, groups split
// by "--".
func TestGrepContextLines(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "l1\nl2\nneedle A\nl4\nl5\nl6\nl7\nl8\nneedle B\nl10\n")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"needle","context":1}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	for _, want := range []string{"a.txt-2-l2", "a.txt:3:needle A", "a.txt-4-l4", "a.txt-8-l8", "a.txt:9:needle B", "a.txt-10-l10", "--"} {
		if !strings.Contains(out, want) {
			t.Fatalf("context output missing %q: %q", want, out)
		}
	}
	if strings.Contains(out, "-6-") {
		t.Fatalf("context leaked lines outside any window: %q", out)
	}
}

// TestGrepContextMergesOverlappingWindows: adjacent matches must form one
// group, with each matched line still marked ':'.
func TestGrepContextMergesOverlappingWindows(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "x\nhit1\nmid\nhit2\ny\n")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"hit","context":1}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if strings.Count(out, "--") != 0 {
		t.Fatalf("overlapping windows must merge into one group: %q", out)
	}
	if !strings.Contains(out, "a.txt:2:hit1") || !strings.Contains(out, "a.txt:4:hit2") || !strings.Contains(out, "a.txt-3-mid") {
		t.Fatalf("merged group malformed: %q", out)
	}
}

// TestGrepLiteralMode: regex metacharacters in the pattern are matched
// verbatim when literal is set.
func TestGrepLiteralMode(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "price is $1.99 (sale)\nprice is X1Y99 Zsale?\n")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"$1.99 (sale)","literal":true}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if !strings.Contains(out, "a.txt:1:") || strings.Contains(out, "a.txt:2:") {
		t.Fatalf("literal mode failed: %q", out)
	}
}

// TestGrepLimitParam: a caller limit below the hard cap truncates with an
// actionable notice.
func TestGrepLimitParam(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "m\nm\nm\nm\nm\n")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"m","limit":2}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if strings.Count(out, "a.txt:") != 2 {
		t.Fatalf("limit not honored: %q", out)
	}
	if !strings.Contains(out, "truncated at 2 matches") {
		t.Fatalf("expected truncation notice: %q", out)
	}
}

func TestGrepSkipsDependencyDirs(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "node_modules/dep/x.js", "needle")
	mustWrite(t, ws, "app.js", "needle")
	out, err := NewGrepTool(nil, ws).Run(context.Background(), `{"pattern":"needle"}`)
	if err != nil {
		t.Fatalf("grep Run: %v", err)
	}
	if strings.Contains(out, "node_modules") {
		t.Fatalf("grep walked node_modules: %q", out)
	}
	if !strings.Contains(out, "app.js") {
		t.Fatalf("grep missed app.js: %q", out)
	}
}
