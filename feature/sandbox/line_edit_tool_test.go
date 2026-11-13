package sandbox

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestEditLinesAppliesMultipleOriginalCoordinateEdits(t *testing.T) {
	ws := mustWorkspace(t)
	original := "one\ntwo\nthree\nfour\nfive\nsix\n"
	mustWrite(t, ws, "a.txt", original)

	out, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		// Deliberately out of order: every coordinate belongs to original.
		{"op": "delete", "start_line": 6, "end_line": 6},
		{"op": "replace", "start_line": 2, "end_line": 3, "text": "TWO\nTHREE+\n"},
		{"op": "insert_after", "line": 4, "text": "inserted"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	want := "one\nTWO\nTHREE+\nfour\ninserted\nfive\n"
	if got := mustRead(t, ws, "a.txt"); got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
	if !strings.Contains(out, "edits_applied: 3") || !strings.Contains(out, "content_hash: "+fileContentHash([]byte(want))) {
		t.Fatalf("output = %q", out)
	}
}

func TestEditLinesSupportsAdjacentRangesAndAppend(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb\nc"
	mustWrite(t, ws, "a.txt", original)

	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		{"op": "replace", "start_line": 1, "end_line": 1, "text": "A"},
		{"op": "replace", "start_line": 2, "end_line": 2, "text": "B"},
		{"op": "append", "text": "d\ne"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got, want := mustRead(t, ws, "a.txt"), "A\nB\nc\nd\ne"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestEditLinesPreservesMissingFinalNewline(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb"
	mustWrite(t, ws, "a.txt", original)
	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		{"op": "replace", "start_line": 1, "end_line": 1, "text": "A\n"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "A\nb" {
		t.Fatalf("content = %q", got)
	}
}

func TestEditLinesDeleteEveryLineProducesEmptyFile(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb\n"
	mustWrite(t, ws, "a.txt", original)
	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		{"op": "delete", "start_line": 1, "end_line": 2},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "" {
		t.Fatalf("content = %q", got)
	}
}

func TestEditLinesCanAppendToEmptyFile(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "empty.txt", "")
	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, "", "empty.txt", []map[string]any{
		{"op": "append", "text": "first\nsecond\n"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got := mustRead(t, ws, "empty.txt"); got != "first\nsecond" {
		t.Fatalf("content = %q", got)
	}
}

func TestEditLinesPreservesBOMCRLFAndFinalNewline(t *testing.T) {
	ws := mustWorkspace(t)
	original := "\uFEFFa\r\nb\r\nc\r\n"
	mustWrite(t, ws, "windows.txt", original)

	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "windows.txt", []map[string]any{
		{"op": "replace", "start_line": 2, "end_line": 2, "text": "B"},
		{"op": "insert_after", "line": 3, "text": "d"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got, want := mustRead(t, ws, "windows.txt"), "\uFEFFa\r\nB\r\nc\r\nd\r\n"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestEditLinesRejectsConflictsWithoutMutation(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb\nc\nd\n"
	mustWrite(t, ws, "a.txt", original)

	cases := []struct {
		name  string
		edits []map[string]any
	}{
		{"overlapping ranges", []map[string]any{
			{"op": "replace", "start_line": 1, "end_line": 2, "text": "x"},
			{"op": "delete", "start_line": 2, "end_line": 3},
		}},
		{"same insertion boundary", []map[string]any{
			{"op": "insert_after", "line": 2, "text": "x"},
			{"op": "insert_before", "line": 3, "text": "y"},
		}},
		{"insert inside range", []map[string]any{
			{"op": "replace", "start_line": 2, "end_line": 3, "text": "x"},
			{"op": "insert_after", "line": 2, "text": "y"},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", tc.edits))
			if err == nil || !strings.Contains(err.Error(), "overlap or use the same insertion boundary") {
				t.Fatalf("error = %v", err)
			}
			if got := mustRead(t, ws, "a.txt"); got != original {
				t.Fatalf("rejected edits mutated file: %q", got)
			}
		})
	}
}

func TestEditLinesAllowsInsertionAtRangeBoundaries(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb\nc\nd\n"
	mustWrite(t, ws, "a.txt", original)
	_, err := NewEditLinesTool(nil, ws).Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		{"op": "insert_before", "line": 2, "text": "before"},
		{"op": "replace", "start_line": 2, "end_line": 3, "text": "BC"},
		{"op": "insert_after", "line": 3, "text": "after"},
	}))
	if err != nil {
		t.Fatalf("edit_lines: %v", err)
	}
	if got, want := mustRead(t, ws, "a.txt"), "a\nbefore\nBC\nafter\nd\n"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestEditLinesRejectsInvalidInputs(t *testing.T) {
	ws := mustWorkspace(t)
	original := "a\nb\n"
	mustWrite(t, ws, "a.txt", original)
	tool := NewEditLinesTool(nil, ws)

	cases := []struct {
		name  string
		edits []map[string]any
		want  string
	}{
		{"empty", nil, "at least one"},
		{"reversed range", []map[string]any{{"op": "delete", "start_line": 2, "end_line": 1}}, "invalid range"},
		{"past eof", []map[string]any{{"op": "replace", "start_line": 2, "end_line": 3, "text": "x"}}, "invalid range"},
		{"bad insertion", []map[string]any{{"op": "insert_after", "line": 3, "text": "x"}}, "invalid line"},
		{"unknown op", []map[string]any{{"op": "move", "line": 1}}, "unsupported op"},
		{"delete text", []map[string]any{{"op": "delete", "start_line": 1, "end_line": 1, "text": "x"}}, "must not include text"},
		{"no change", []map[string]any{{"op": "replace", "start_line": 1, "end_line": 1, "text": "a"}}, "make no changes"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tool.Run(context.Background(), lineEditArgs(t, original, "a.txt", tc.edits))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
			if got := mustRead(t, ws, "a.txt"); got != original {
				t.Fatalf("invalid edit mutated file: %q", got)
			}
		})
	}
}

func TestEditLinesRejectsStaleAndConcurrentSnapshots(t *testing.T) {
	t.Run("stale", func(t *testing.T) {
		ws := mustWorkspace(t)
		original := "a\nb\n"
		mustWrite(t, ws, "a.txt", original)
		args := lineEditArgs(t, original, "a.txt", []map[string]any{{"op": "replace", "start_line": 1, "end_line": 1, "text": "A"}})
		mustWrite(t, ws, "a.txt", "a\nchanged\n")
		_, err := NewEditLinesTool(nil, ws).Run(context.Background(), args)
		if err == nil || !strings.Contains(err.Error(), "stale snapshot") {
			t.Fatalf("error = %v", err)
		}
		if got := mustRead(t, ws, "a.txt"); got != "a\nchanged\n" {
			t.Fatalf("stale edit mutated file: %q", got)
		}
	})

	t.Run("concurrent", func(t *testing.T) {
		ws := mustWorkspace(t)
		original := "a\nb\n"
		mustWrite(t, ws, "a.txt", original)
		guarded := &mutatingWorkspaceFS{
			workspaceFS:      hostFS{ws: ws},
			beforeSecondRead: func() { mustWrite(t, ws, "a.txt", "a\nchanged\n") },
		}
		tool := &EditLinesTool{workspace: ws, fs: guarded}
		_, err := tool.Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
			{"op": "replace", "start_line": 1, "end_line": 1, "text": "A"},
		}))
		if err == nil || !strings.Contains(err.Error(), "changed while preparing edits") {
			t.Fatalf("error = %v", err)
		}
		if got := mustRead(t, ws, "a.txt"); got != "a\nchanged\n" {
			t.Fatalf("concurrent edit was overwritten: %q", got)
		}
	})
}

func TestEditLinesSchemaAndSandboxSubstrate(t *testing.T) {
	ws := mustWorkspace(t)
	tool := NewEditLinesTool(NewHostSandbox(), ws)
	if tool.Name() != ToolEditLines {
		t.Fatalf("name = %q", tool.Name())
	}
	required, ok := tool.Schema().Parameters["required"].([]string)
	if !ok || !containsAll(required, "path", "expected_hash", "edits") {
		t.Fatalf("required schema = %#v", tool.Schema().Parameters["required"])
	}

	original := "one\ntwo\n"
	mustWrite(t, ws, "a.txt", original)
	_, err := tool.Run(context.Background(), lineEditArgs(t, original, "a.txt", []map[string]any{
		{"op": "replace", "start_line": 2, "end_line": 2, "text": "TWO"},
	}))
	if err != nil {
		t.Fatalf("sandbox edit_lines: %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "one\nTWO\n" {
		t.Fatalf("content = %q", got)
	}
}

func lineEditArgs(t *testing.T, content, path string, edits []map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"path": path, "expected_hash": fileContentHash([]byte(content)), "edits": edits,
	})
	if err != nil {
		t.Fatalf("marshal line edit args: %v", err)
	}
	return string(raw)
}

func containsAll(values []string, wanted ...string) bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	for _, value := range wanted {
		if !set[value] {
			return false
		}
	}
	return true
}
