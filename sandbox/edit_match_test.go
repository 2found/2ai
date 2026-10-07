package sandbox

import (
	"context"
	"strings"
	"testing"
)

// TestEditFuzzySmartQuotesAndDashes: the model quotes a line back with ASCII
// quotes/dashes but the file has unicode ones — the fuzzy pass must match, and
// lines outside the edit must keep their original (unicode) bytes.
func TestEditFuzzySmartQuotesAndDashes(t *testing.T) {
	ws := mustWorkspace(t)
	// Line 1 uses smart quotes + en dash; line 2 also has smart quotes but is untouched.
	mustWrite(t, ws, "a.txt", "msg := “Hello – world”\nkeep := ‘untouched’\n")

	out, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", `msg := "Hello - world"`, `msg := "Bye"`, false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	if !strings.Contains(out, "match: fuzzy") || !strings.Contains(out, "replacements: 1") {
		t.Fatalf("expected fuzzy match note, got %q", out)
	}
	got := mustRead(t, ws, "a.txt")
	if !strings.Contains(got, "msg := \"Bye\"") {
		t.Fatalf("replacement missing: %q", got)
	}
	if !strings.Contains(got, "keep := ‘untouched’") {
		t.Fatalf("untouched line lost its original bytes: %q", got)
	}
}

// TestEditFuzzyTrailingWhitespace: trailing spaces in the file that the model
// (correctly) does not reproduce must not block the edit.
func TestEditFuzzyTrailingWhitespace(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.go", "func A() {}   \nfunc B() {}\t\nfunc C() {}\n")

	_, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.go", "func A() {}\nfunc B() {}", "func A() {}\nfunc B2() {}", false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	got := mustRead(t, ws, "a.go")
	if !strings.Contains(got, "func B2() {}") {
		t.Fatalf("edit not applied: %q", got)
	}
	// Line C sits outside the edited span: byte-identical, trailing tab and all... it has none.
	if !strings.HasSuffix(got, "func C() {}\n") {
		t.Fatalf("untouched trailing line changed: %q", got)
	}
}

// TestEditFuzzyPreservesUntouchedTrailingWhitespace: lines outside the matched
// span keep their original bytes even when they carry the same class of
// mangling the fuzzy pass normalizes.
func TestEditFuzzyPreservesUntouchedTrailingWhitespace(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "first   \nsecond line \nthird\n")

	_, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", "third", "THIRD", false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	got := mustRead(t, ws, "a.txt")
	// Exact match on "third" — every other line must be byte-identical.
	if got != "first   \nsecond line \nTHIRD\n" {
		t.Fatalf("untouched lines rewritten: %q", got)
	}
}

// TestEditCRLFFileRoundTrips: a CRLF file is edited with LF-style old/new
// strings and stays CRLF on disk.
func TestEditCRLFFileRoundTrips(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "alpha\r\nbeta\r\ngamma\r\n")

	out, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", "beta", "BETA", false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	if !strings.Contains(out, "match: exact") {
		t.Fatalf("CRLF-only difference should still be the exact path: %q", out)
	}
	if got := mustRead(t, ws, "a.txt"); got != "alpha\r\nBETA\r\ngamma\r\n" {
		t.Fatalf("CRLF not preserved: %q", got)
	}
}

// TestEditBOMPreserved: a UTF-8 BOM must survive the edit without the model
// having to include it in old_string.
func TestEditBOMPreserved(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "\uFEFFhello world\n")

	if _, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", "hello world", "hello there", false)); err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	if got := mustRead(t, ws, "a.txt"); got != "\uFEFFhello there\n" {
		t.Fatalf("BOM lost or content wrong: %q", got)
	}
}

// TestEditFuzzyAmbiguousRejected: multiple fuzzy matches without replace_all
// are refused, mirroring the exact path's uniqueness contract.
func TestEditFuzzyAmbiguousRejected(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "say “hi”\nsay “hi”\n")

	_, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", `say "hi"`, `say "yo"`, false))
	if err == nil || !strings.Contains(err.Error(), "fuzzy normalization") {
		t.Fatalf("expected fuzzy-ambiguous rejection, got %v", err)
	}
}

// TestEditFuzzyReplaceAll: replace_all applies the fuzzy match everywhere and
// reports the count.
func TestEditFuzzyReplaceAll(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "say “hi”\nmiddle\nsay “hi”\n")

	out, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", `say "hi"`, `say "yo"`, true))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	if !strings.Contains(out, "replacements: 2") || !strings.Contains(out, "match: fuzzy") {
		t.Fatalf("expected 2 fuzzy replacements, got %q", out)
	}
	got := mustRead(t, ws, "a.txt")
	if strings.Count(got, "say \"yo\"") != 2 || !strings.Contains(got, "middle") {
		t.Fatalf("replace_all fuzzy failed: %q", got)
	}
}

// TestEditFuzzyStillNotFound: fuzzy is a fallback, not a similarity search —
// genuinely different text must still be rejected.
func TestEditFuzzyStillNotFound(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "completely different content\n")

	_, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", "nothing like this", "x", false))
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected not-found, got %v", err)
	}
}

// TestEditFuzzyMultilineSpanPreservesNeighbors: a multi-line fuzzy edit
// rewrites only the matched span's lines; neighbors keep original bytes.
func TestEditFuzzyMultilineSpanPreservesNeighbors(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "a.txt", "before’s line\nspan one \nspan “two”\nafter’s line\n")

	_, err := NewEditFileTool(nil, ws).Run(context.Background(),
		editArgs(t, ws, "a.txt", "span one\nspan \"two\"", "replaced", false))
	if err != nil {
		t.Fatalf("edit Run: %v", err)
	}
	got := mustRead(t, ws, "a.txt")
	if got != "before’s line\nreplaced\nafter’s line\n" {
		t.Fatalf("multiline fuzzy span wrong: %q", got)
	}
}
