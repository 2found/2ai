package sandbox

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// glob lists the most recently modified file first, as Claude Code's Glob
// does, and a capped list keeps the newest files.
func TestGlobListsNewestFirst(t *testing.T) {
	ws := newTestWorkspace(t)
	now := time.Now()
	for i, name := range []string{"a.go", "b.go", "c.go"} {
		p := filepath.Join(ws.Root(), name)
		if err := os.WriteFile(p, []byte("package x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		// b newest, then c, then a.
		age := map[string]time.Duration{"a.go": 3 * time.Hour, "b.go": time.Minute, "c.go": time.Hour}[name]
		if err := os.Chtimes(p, now.Add(-age), now.Add(-age)); err != nil {
			t.Fatal(err, i)
		}
	}
	out, err := NewGlobTool(nil, ws).Run(context.Background(), `{"pattern":"*.go"}`)
	if err != nil || out != "b.go\nc.go\na.go" {
		t.Fatalf("glob = %q, %v", out, err)
	}
}

func TestGlobCapKeepsTheNewest(t *testing.T) {
	ws := newTestWorkspace(t)
	old := time.Now().Add(-time.Hour)
	for i := 0; i < maxGlobMatches+5; i++ {
		p := filepath.Join(ws.Root(), fmt.Sprintf("f%04d.txt", i))
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, old, old); err != nil {
			t.Fatal(err)
		}
	}
	fresh := filepath.Join(ws.Root(), "zz-new.txt")
	if err := os.WriteFile(fresh, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := NewGlobTool(nil, ws).Run(context.Background(), `{"pattern":"*.txt"}`)
	if err != nil || !strings.HasPrefix(out, "zz-new.txt\nf0000.txt\n") || !strings.Contains(out, "6 more files not shown") {
		t.Fatalf("glob = %.200q … %v", out, err)
	}
}
