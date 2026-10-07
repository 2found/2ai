package spill

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

// failingSpillStore rejects every save, so the policy must degrade to plain
// truncation instead of failing the tool call.
type failingSpillStore struct{}

func (failingSpillStore) SaveText(context.Context, SpillRequest) (SpillRecord, error) {
	return SpillRecord{}, errors.New("storage down")
}
func (failingSpillStore) ReadText(context.Context, string, int, int) (SpillSlice, error) {
	return SpillSlice{}, ErrSpillNotFound
}

func begin(t *testing.T, p Plugin, session string) *spillRun {
	t.Helper()
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{SessionID: session, Limits: agentcore.DefaultLimits()})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	if ext == nil {
		t.Fatal("expected the plugin to accept the run")
	}
	return ext.(*spillRun)
}

func newTestPolicy(t *testing.T, max int) *spillRun {
	t.Helper()
	return begin(t, Plugin{Store: NewMemorySpillStore(), MaxInlineBytes: max}, "sess-1")
}

// bound reproduces exactly what the loop does with a tool result: ask the
// extension, and fall back to the loop's own head+tail truncation when it has
// no opinion. Testing through this — rather than through apply — is what keeps
// the "spilling never makes a result bigger, and never hides one" claims true
// of what the model ACTUALLY sees.
func bound(p *spillRun, call agentcore.ToolCall, out string) (string, string) {
	d := p.InterceptToolResult(context.Background(), call, out, nil)
	if !d.Replace {
		return agentcore.TruncateMiddle(out, p.maxInline), d.ResultRef
	}
	return d.Result, d.ResultRef
}

func TestSpill_UnderCapIsUntouched(t *testing.T) {
	p := newTestPolicy(t, 1000)
	out, loc := bound(p, agentcore.ToolCall{ID: "c1", Name: "run_sql"}, "small result")
	if out != "small result" {
		t.Fatalf("result was modified: %q", out)
	}
	if loc != "" {
		t.Fatalf("nothing should have spilled, got locator %q", loc)
	}
}

func TestSpill_OversizedIsSavedWholeAndPreviewFitsCap(t *testing.T) {
	const cap = 400
	store := NewMemorySpillStore()
	p := begin(t, Plugin{Store: store, MaxInlineBytes: cap}, "sess-1")

	full := strings.Repeat("A", 2000) + "TAIL_MARKER"
	out, loc := bound(p, agentcore.ToolCall{ID: "c1", Name: "run_sql"}, full)

	if loc == "" {
		t.Fatal("expected a spill locator")
	}
	// The whole point: the replacement never exceeds the cap.
	if len(out) > cap {
		t.Fatalf("replacement %d bytes exceeds cap %d", len(out), cap)
	}
	// And spilling never makes a result bigger than it was.
	if len(out) >= len(full) {
		t.Fatalf("replacement %d bytes is not smaller than the original %d", len(out), len(full))
	}
	if !strings.Contains(out, loc) {
		t.Fatalf("notice must carry the locator; got %q", out)
	}
	if !strings.Contains(out, "TAIL_MARKER") {
		t.Fatal("preview must retain the tail — the end usually carries the signal")
	}
	// The full text is recoverable verbatim.
	slice, err := store.ReadText(context.Background(), loc, 0, len(full))
	if err != nil {
		t.Fatalf("ReadText: %v", err)
	}
	if slice.Content != full {
		t.Fatal("stored artifact does not match the original result")
	}
	if slice.Total != len(full) {
		t.Fatalf("Total = %d, want %d", slice.Total, len(full))
	}
}

func TestSpill_NoticeReportsExactOmission(t *testing.T) {
	p := newTestPolicy(t, 500)
	full := strings.Repeat("B", 5000)
	out, _ := bound(p, agentcore.ToolCall{ID: "c1", Name: "web_fetch"}, full)

	// preview bytes + omitted bytes must reconstruct the original length, or the
	// number shown to the model is a lie it will reason from.
	preview := out[:strings.Index(out, "\n\n(Omitted")]
	previewBytes := len(strings.ReplaceAll(preview, "\n…\n", ""))
	var omitted, total int
	if _, err := fmtSscan(out, &omitted, &total); err != nil {
		t.Fatalf("could not parse notice from %q: %v", out, err)
	}
	if total != len(full) {
		t.Fatalf("notice total = %d, want %d", total, len(full))
	}
	if previewBytes+omitted != len(full) {
		t.Fatalf("preview %d + omitted %d != original %d", previewBytes, omitted, len(full))
	}
}

// fmtSscan pulls the two counts out of "(Omitted N of M bytes. …)".
func fmtSscan(s string, omitted, total *int) (int, error) {
	i := strings.Index(s, "(Omitted ")
	if i < 0 {
		return 0, errors.New("no notice")
	}
	rest := s[i+len("(Omitted "):]
	var o, tt int
	n, err := sscanTwo(rest, &o, &tt)
	*omitted, *total = o, tt
	return n, err
}

func sscanTwo(s string, a, b *int) (int, error) {
	parts := strings.Fields(s)
	if len(parts) < 3 {
		return 0, errors.New("short notice")
	}
	if _, err := parseInt(parts[0], a); err != nil {
		return 0, err
	}
	if _, err := parseInt(parts[2], b); err != nil {
		return 0, err
	}
	return 2, nil
}

func parseInt(s string, out *int) (int, error) {
	v := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0, errors.New("not a number: " + s)
		}
		v = v*10 + int(r-'0')
	}
	*out = v
	return v, nil
}

func TestSpill_SaveFailureFallsBackToTruncation(t *testing.T) {
	p := begin(t, Plugin{Store: failingSpillStore{}, MaxInlineBytes: 300}, "sess-1")
	full := strings.Repeat("C", 4000)
	out, loc := bound(p, agentcore.ToolCall{ID: "c1", Name: "run_sql"}, full)

	if loc != "" {
		t.Fatalf("a failed save must not report a locator, got %q", loc)
	}
	if len(out) > 300 {
		t.Fatalf("fallback truncation exceeded the cap: %d bytes", len(out))
	}
	if strings.Contains(out, "Full result saved at") {
		t.Fatal("must not promise a locator the store never wrote")
	}
	if out == "" {
		t.Fatal("a spill failure must never hide the result")
	}
}

func TestSpill_ReadSpillToolIsNeverSpilled(t *testing.T) {
	p := newTestPolicy(t, 200)
	full := strings.Repeat("D", 4000)
	out, loc := bound(p, agentcore.ToolCall{ID: "c1", Name: readSpillToolName}, full)
	if loc != "" {
		t.Fatal("spilling a read_spill result would loop: read -> spill -> read")
	}
	if len(out) > 200 {
		t.Fatalf("excluded tool result still must respect the cap, got %d", len(out))
	}
}

func TestSpill_TinyCapEmitsNoticeOnlyAndNeverOverruns(t *testing.T) {
	// A cap smaller than the notice must not produce an over-cap replacement.
	for _, cap := range []int{1, 20, 60, 120} {
		p := newTestPolicy(t, cap)
		full := strings.Repeat("E", 9000)
		out, _ := bound(p, agentcore.ToolCall{ID: "c1", Name: "t"}, full)
		if len(out) > cap {
			t.Fatalf("cap %d: replacement is %d bytes", cap, len(out))
		}
	}
}

func TestSpill_PreviewNeverSplitsARune(t *testing.T) {
	p := newTestPolicy(t, 300)
	full := strings.Repeat("Cứ đi rồi sẽ tới — ", 400) // multi-byte throughout
	out, _ := bound(p, agentcore.ToolCall{ID: "c1", Name: "t"}, full)
	if !utf8.ValidString(out) {
		t.Fatal("preview split a UTF-8 rune")
	}
}

func TestSpill_ReadToolPagesAndFencesBySession(t *testing.T) {
	store := NewMemorySpillStore()
	p := begin(t, Plugin{Store: store, MaxInlineBytes: 300}, "sess-1")
	full := strings.Repeat("F", 3000) + "END"
	_, loc := bound(p, agentcore.ToolCall{ID: "c1", Name: "run_sql"}, full)

	tool := &readSpillTool{policy: p}
	args, _ := json.Marshal(map[string]any{"locator": loc, "offset": 0, "limit": 100})
	got, err := tool.Run(context.Background(), string(args))
	if err != nil {
		t.Fatalf("read_spill: %v", err)
	}
	if !strings.Contains(got, "of 3003") {
		t.Fatalf("header must report the artifact total; got %q", got[:60])
	}
	if strings.Contains(got, "end of artifact") {
		t.Fatal("a 100-byte window of a 3003-byte artifact is not EOF")
	}

	// Paging to the end reports EOF and returns the tail.
	args, _ = json.Marshal(map[string]any{"locator": loc, "offset": 2950, "limit": 100})
	got, err = tool.Run(context.Background(), string(args))
	if err != nil {
		t.Fatalf("read_spill tail: %v", err)
	}
	if !strings.Contains(got, "END") || !strings.Contains(got, "end of artifact") {
		t.Fatalf("tail read wrong: %q", got)
	}

	// The fence: another session's run cannot read this locator.
	other := begin(t, Plugin{Store: store, MaxInlineBytes: 300}, "sess-2")
	args, _ = json.Marshal(map[string]any{"locator": loc})
	if _, err := (&readSpillTool{policy: other}).Run(context.Background(), string(args)); !errors.Is(err, ErrSpillNotFound) {
		t.Fatalf("cross-session read must be not-found, got %v", err)
	}
}

func TestSpill_ReadToolRejectsMissingLocator(t *testing.T) {
	p := newTestPolicy(t, 300)
	tool := &readSpillTool{policy: p}
	if _, err := tool.Run(context.Background(), `{"locator":"  "}`); err == nil {
		t.Fatal("expected an error for a blank locator")
	}
	if _, err := tool.Run(context.Background(), `not json`); err == nil {
		t.Fatal("expected an error for malformed arguments")
	}
}

// A plugin that cannot do its job must DECLINE the run (nil extension), not
// install itself and then no-op: an extension the loop holds but that never
// acts is indistinguishable, from the outside, from one that is broken.
func TestSpill_DeclinesRunItCannotServe(t *testing.T) {
	cases := []struct {
		name   string
		plugin Plugin
		limits agentcore.Limits
	}{
		{"no store", Plugin{}, agentcore.DefaultLimits()},
		{"no cap", Plugin{Store: NewMemorySpillStore()}, agentcore.Limits{MaxToolResultLen: 0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ext, err := tc.plugin.BeginRun(context.Background(), agentcore.RunInfo{SessionID: "s", Limits: tc.limits})
			if err != nil {
				t.Fatalf("declining is not an error: %v", err)
			}
			if ext != nil {
				t.Fatal("expected the plugin to decline the run")
			}
		})
	}
}

// A failed call is left entirely alone: the loop is about to replace its
// "result" with the error text, so spilling it would spend storage on a string
// nobody reads.
func TestSpill_FailedCallIsNotSpilled(t *testing.T) {
	p := newTestPolicy(t, 200)
	d := p.InterceptToolResult(context.Background(), agentcore.ToolCall{ID: "c1", Name: "run_sql"}, strings.Repeat("H", 9000), errors.New("boom"))
	if d.Replace || d.ResultRef != "" {
		t.Fatalf("a failed call must pass through untouched, got %+v", d)
	}
}
