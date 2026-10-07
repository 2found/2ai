package sandbox

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestReadFileNumbersLines(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "one\ntwo\nthree\n")
	out, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"a.txt"}`)
	if err != nil {
		t.Fatalf("read Run: %v", err)
	}
	if !strings.Contains(out, "1\tone") || !strings.Contains(out, "2\ttwo") {
		t.Fatalf("expected numbered lines, got %q", out)
	}
}

func TestReadFileOffsetLimit(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "l1\nl2\nl3\nl4\nl5\n")
	out, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"a.txt","offset":2,"limit":2}`)
	if err != nil {
		t.Fatalf("read Run: %v", err)
	}
	if !strings.Contains(out, "2\tl2") || !strings.Contains(out, "3\tl3") {
		t.Fatalf("window missing expected lines: %q", out)
	}
	if strings.Contains(out, "\tl1") || strings.Contains(out, "\tl4") {
		t.Fatalf("window leaked out-of-range lines: %q", out)
	}
	if !strings.Contains(out, "truncated: true") {
		t.Fatalf("expected truncated marker: %q", out)
	}
	if !strings.Contains(out, "Use offset=4 to continue.") {
		t.Fatalf("expected actionable continuation hint: %q", out)
	}
}

// TestReadFilePagesPastByteBudget is the paging regression: the 64KB byte
// budget used to be applied to the whole file before windowing, so lines past
// 64KB were unreachable at any offset. The budget must apply to the selected
// window instead.
func TestReadFilePagesPastByteBudget(t *testing.T) {
	ws := mustWorkspace(t)
	// ~100KB file: 2000 lines of ~50 bytes. Line 1500 sits past the 64KB mark.
	var b strings.Builder
	for i := 1; i <= 2000; i++ {
		fmt.Fprintf(&b, "line-%04d %s\n", i, strings.Repeat("x", 40))
	}
	mustWrite(t, ws, "big.txt", b.String())

	out, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"big.txt","offset":1500,"limit":3}`)
	if err != nil {
		t.Fatalf("read Run: %v", err)
	}
	for _, want := range []string{"1500\tline-1500", "1502\tline-1502"} {
		if !strings.Contains(out, want) {
			t.Fatalf("window past 64KB unreachable, missing %q in %q", want, out)
		}
	}
	if !strings.Contains(out, "Use offset=1503 to continue.") {
		t.Fatalf("expected continuation hint: %q", out)
	}
}

// TestReadFileOffsetBeyondEOFErrors pins the actionable error over the old
// silent-empty-content behavior.
func TestReadFileOffsetBeyondEOFErrors(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "one\ntwo\n")
	_, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"a.txt","offset":99}`)
	if err == nil || !strings.Contains(err.Error(), "beyond end of file (2 lines)") {
		t.Fatalf("expected beyond-EOF error, got %v", err)
	}
}

// TestReadFileClampsOversizedSingleLine: a single line larger than the budget
// must yield a clamped window with a note, never an empty result.
func TestReadFileClampsOversizedSingleLine(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "wide.txt", strings.Repeat("y", 80*1024)+"\nshort\n")
	out, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"wide.txt"}`)
	if err != nil {
		t.Fatalf("read Run: %v", err)
	}
	if !strings.Contains(out, "showing its first 64KB") {
		t.Fatalf("expected oversized-line note: %q", out)
	}
	if !strings.Contains(out, "Use offset=2 to continue.") {
		t.Fatalf("expected continuation past the clamped line: %q", out)
	}
}
