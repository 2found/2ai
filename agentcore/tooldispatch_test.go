package agentcore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

const testRichPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

// namedTool is a minimal Tool used to assert the ToolSet registry contract
// (ordering, lookup, overwrite) without standing up a provider/loop.
type namedTool struct {
	name string
	desc string
}

func (t namedTool) Name() string { return t.name }
func (t namedTool) Schema() ToolSchema {
	return ToolSchema{Name: t.name, Description: t.desc}
}
func (namedTool) Run(context.Context, string) (string, error) { return "", nil }

type resultRefInterceptor struct{}

func (resultRefInterceptor) InterceptToolResult(context.Context, ToolCall, string, error) ToolResultDecision {
	return ToolResultDecision{
		Result: "bounded preview", Replace: true,
		Meta: "cache=miss", ResultRef: "spill_dispatch_1",
	}
}

type richResultTool struct {
	fail  bool
	parts []ContentPart
}

func (richResultTool) Name() string { return "rich" }
func (richResultTool) Schema() ToolSchema {
	return ToolSchema{Name: "rich", Parameters: map[string]any{"type": "object"}}
}
func (richResultTool) Run(context.Context, string) (string, error) {
	return "legacy path must not run", nil
}
func (t richResultTool) RunRich(context.Context, string) (ToolOutput, error) {
	parts := t.parts
	if parts == nil {
		parts = []ContentPart{{
			Type: ContentPartImage, MIMEType: "image/png", Data: testRichPNG,
		}}
	}
	out := ToolOutput{Content: "plot ready", Parts: parts}
	if t.fail {
		return out, errors.New("render failed")
	}
	return out, nil
}

func TestRichToolResultIsBoundedAtDispatchBoundary(t *testing.T) {
	parts := make([]ContentPart, 0, maxRichToolImages+2)
	for i := 0; i < maxRichToolImages+1; i++ {
		parts = append(parts, ContentPart{Type: ContentPartImage, MIMEType: "image/png", Data: testRichPNG})
	}
	parts = append(parts, ContentPart{Type: ContentPartImage, MIMEType: "text/plain", Data: testRichPNG})
	out := (&Agent{}).runToolCall(
		context.Background(), &extensionSet{}, map[string]bool{"rich": true},
		NewToolSet(richResultTool{parts: parts}), ToolCall{ID: "c1", Name: "rich", Arguments: `{}`},
		DefaultLimits(), nil,
	)
	if len(out.message.ContentParts) != maxRichToolImages {
		t.Fatalf("rich part count = %d, want %d", len(out.message.ContentParts), maxRichToolImages)
	}
	if !strings.Contains(out.message.Content, "2 rich content part(s) omitted") {
		t.Fatalf("omission was not visible to the model: %q", out.message.Content)
	}
	if !strings.Contains(out.trace.ResultMeta, "8 rich parts") {
		t.Fatalf("rich result was not reflected in trace metadata: %q", out.trace.ResultMeta)
	}
}

func TestRichToolRejectsMislabeledImageBytes(t *testing.T) {
	out := (&Agent{}).runToolCall(
		context.Background(), &extensionSet{}, map[string]bool{"rich": true},
		NewToolSet(richResultTool{parts: []ContentPart{{
			Type: ContentPartImage, MIMEType: "image/png", Data: "bm90IGEgcG5n",
		}}}), ToolCall{ID: "c1", Name: "rich", Arguments: `{}`},
		DefaultLimits(), nil,
	)
	if len(out.message.ContentParts) != 0 || !strings.Contains(out.message.Content, "1 rich content part(s) omitted") {
		t.Fatalf("invalid image reached canonical history: %+v", out.message)
	}
}

func TestRichToolResultReachesCanonicalToolMessage(t *testing.T) {
	out := (&Agent{}).runToolCall(
		context.Background(), &extensionSet{}, map[string]bool{"rich": true},
		NewToolSet(richResultTool{}), ToolCall{ID: "c1", Name: "rich", Arguments: `{}`},
		DefaultLimits(), nil,
	)
	if !strings.HasPrefix(out.message.Content, "plot ready\n[Image normalized from 1x1 to 200x200") || len(out.message.ContentParts) != 1 {
		t.Fatalf("rich result = %+v", out.message)
	}
	if got := out.message.ContentParts[0]; got.Type != ContentPartImage || got.MIMEType != "image/png" || got.Data == testRichPNG {
		t.Fatalf("image part = %+v", got)
	}
	if out.trace.ResultMeta == "" || !out.executed {
		t.Fatalf("rich tool skipped ordinary accounting: %+v", out)
	}
}

func TestRichToolDropsPartsOnError(t *testing.T) {
	out := (&Agent{}).runToolCall(
		context.Background(), &extensionSet{}, map[string]bool{"rich": true},
		NewToolSet(richResultTool{fail: true}), ToolCall{ID: "c1", Name: "rich", Arguments: `{}`},
		DefaultLimits(), nil,
	)
	if len(out.message.ContentParts) != 0 || !strings.Contains(out.message.Content, "render failed") {
		t.Fatalf("failed rich result leaked parts: %+v", out.message)
	}
}

func TestToolResultSeparatesTraceMetadataFromRecoverableReference(t *testing.T) {
	out := (&Agent{}).runToolCall(
		context.Background(),
		&extensionSet{toolIntcp: []ToolInterceptor{resultRefInterceptor{}}},
		map[string]bool{"query": true},
		NewToolSet(namedTool{name: "query"}),
		ToolCall{ID: "c1", Name: "query", Arguments: `{}`},
		DefaultLimits(),
		nil,
	)
	if out.message.ResultRef != "spill_dispatch_1" || out.trace.SpillLocator != "spill_dispatch_1" {
		t.Fatalf("recoverable ref did not reach message and trace: message=%+v trace=%+v", out.message, out.trace)
	}
	if !strings.Contains(out.trace.ResultMeta, "cache=miss") || strings.Contains(out.message.Content, "cache=miss") {
		t.Fatalf("trace-only metadata crossed the wrong boundary: message=%+v trace=%+v", out.message, out.trace)
	}
}

// TestToolSetPreservesRegistrationOrder verifies Names/Schemas reflect insertion
// order — the order the model is shown its tools in.
func TestToolSetPreservesRegistrationOrder(t *testing.T) {
	ts := NewToolSet(namedTool{name: "a"}, namedTool{name: "b"}, namedTool{name: "c"})
	if got := ts.Names(); strings.Join(got, ",") != "a,b,c" {
		t.Fatalf("names = %v, want [a b c]", got)
	}
	schemas := ts.Schemas()
	if len(schemas) != 3 || schemas[0].Name != "a" || schemas[2].Name != "c" {
		t.Fatalf("schemas out of order: %+v", schemas)
	}
}

// TestToolSetAddOverwritesInPlace verifies re-adding a name replaces the tool but
// keeps its original position (Add's overwrite branch), so a per-agent override of
// a same-named default does not reshuffle the catalog the model sees.
func TestToolSetAddOverwritesInPlace(t *testing.T) {
	ts := NewToolSet(namedTool{name: "a"}, namedTool{name: "b"})
	ts.Add(namedTool{name: "a", desc: "v2"})

	if got := ts.Names(); strings.Join(got, ",") != "a,b" {
		t.Fatalf("overwrite changed order/count: %v", got)
	}
	got, ok := ts.Get("a")
	if !ok {
		t.Fatal("Get(a) missing after overwrite")
	}
	if got.Schema().Description != "v2" {
		t.Fatalf("overwrite did not replace the tool: %q", got.Schema().Description)
	}
}

// TestToolSetGetMiss verifies a lookup for an unregistered name fails closed.
func TestToolSetGetMiss(t *testing.T) {
	ts := NewToolSet(namedTool{name: "a"})
	if _, ok := ts.Get("missing"); ok {
		t.Fatal("Get(missing) should report not-found")
	}
}

func TestTruncateBytesShortStringUnchanged(t *testing.T) {
	const s = "hello"
	if got := truncateBytes(s, 1024); got != s {
		t.Fatalf("short string altered: %q", got)
	}
	// maxBytes <= 0 disables truncation entirely.
	long := strings.Repeat("x", 100)
	if got := truncateBytes(long, 0); got != long {
		t.Fatalf("maxBytes=0 should disable truncation, got %d bytes", len(got))
	}
}

func TestTruncateBytesAppendsMarkerWhenCut(t *testing.T) {
	long := strings.Repeat("x", 1000)
	got := truncateBytes(long, 64)
	if len(got) > 64 {
		t.Fatalf("result exceeds maxBytes: %d", len(got))
	}
	if !strings.HasSuffix(got, "[truncated]") {
		t.Fatalf("expected truncation marker, got %q", got)
	}
}

// TestTruncateBytesDoesNotSplitRune guards the UTF-8 back-off: cutting in the
// middle of a multibyte rune must never yield invalid UTF-8.
func TestTruncateBytesDoesNotSplitRune(t *testing.T) {
	// "世" is 3 bytes each; a byte budget that lands mid-rune exercises RuneStart.
	s := strings.Repeat("世", 100)
	for budget := 20; budget < 40; budget++ {
		got := truncateBytes(s, budget)
		if !utf8.ValidString(got) {
			t.Fatalf("budget %d produced invalid UTF-8: %q", budget, got)
		}
	}
}

// schemaTool advertises a JSON Schema requiring a string "sql" field and an
// optional enum "mode". It records the args it actually ran with.
type schemaTool struct {
	called  int
	lastArg string
}

func (s *schemaTool) Name() string { return "run_query" }
func (s *schemaTool) Schema() ToolSchema {
	return ToolSchema{
		Name:        "run_query",
		Description: "run a SQL query",
		Parameters: map[string]any{
			"type":     "object",
			"required": []any{"sql"},
			"properties": map[string]any{
				"sql":  map[string]any{"type": "string"},
				"mode": map[string]any{"type": "string", "enum": []any{"read", "write"}},
			},
		},
	}
}
func (s *schemaTool) Run(_ context.Context, args string) (string, error) {
	s.called++
	s.lastArg = args
	return "ok", nil
}

func TestValidateArgs_RequiredAndTypes(t *testing.T) {
	tool := &schemaTool{}
	schema := tool.Schema().Parameters
	cases := []struct {
		name    string
		args    string
		wantErr bool
	}{
		{"valid", `{"sql":"select 1"}`, false},
		{"valid with enum", `{"sql":"select 1","mode":"read"}`, false},
		{"missing required", `{"mode":"read"}`, true},
		{"wrong type", `{"sql":123}`, true},
		{"bad enum", `{"sql":"x","mode":"delete"}`, true},
		{"unknown field ok", `{"sql":"x","extra":true}`, false},
		{"malformed json", `{"sql":`, true},
		{"empty needs required", ``, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := NewToolSet().validateToolArgs("<test>", c.args, schema)
			if (err != nil) != c.wantErr {
				t.Fatalf("validateToolArgs(%q) err=%v, wantErr=%v", c.args, err, c.wantErr)
			}
		})
	}
}

// TestValidateArgs_UnionTypes verifies list-valued type constraints — the
// nullable form ["string","null"] — accept any member and reject the rest,
// instead of silently passing everything (pi #7243's nullable-schema gap).
func TestValidateArgs_UnionTypes(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tag":   map[string]any{"type": []any{"string", "null"}},
			"items": map[string]any{"type": []any{"array", "null"}},
		},
	}
	cases := []struct {
		name    string
		args    string
		wantErr bool
	}{
		{"string member ok", `{"tag":"a"}`, false},
		{"null member ok", `{"tag":null}`, false},
		{"non-member rejected", `{"tag":7}`, true},
		{"array member ok", `{"items":[1,2]}`, false},
		{"object not in union", `{"items":{"a":1}}`, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := NewToolSet().validateToolArgs("<test>", c.args, schema)
			if (err != nil) != c.wantErr {
				t.Fatalf("validateToolArgs(%q) err=%v, wantErr=%v", c.args, err, c.wantErr)
			}
		})
	}
}

func TestValidateArgs_FullJSONSchema(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"required":             []string{"profile", "choice"},
		"additionalProperties": false,
		"properties": map[string]any{
			"profile": map[string]any{
				"type":                 "object",
				"required":             []string{"name", "roles"},
				"additionalProperties": false,
				"properties": map[string]any{
					"name": map[string]any{"type": "string", "minLength": 3},
					"roles": map[string]any{
						"type":     "array",
						"minItems": 1,
						"items":    map[string]any{"type": "string", "enum": []string{"reader", "writer"}},
					},
				},
			},
			"choice": map[string]any{
				"oneOf": []any{
					map[string]any{"type": "integer", "minimum": 1},
					map[string]any{"type": "string", "pattern": "^[a-z]+$"},
				},
			},
		},
	}
	cases := []struct {
		name    string
		args    string
		wantErr bool
	}{
		{"valid integer branch", `{"profile":{"name":"Ada","roles":["writer"]},"choice":2}`, false},
		{"valid string branch", `{"profile":{"name":"Ada","roles":["reader"]},"choice":"alpha"}`, false},
		{"nested required", `{"profile":{"name":"Ada"},"choice":2}`, true},
		{"nested additional property", `{"profile":{"name":"Ada","roles":["reader"],"admin":true},"choice":2}`, true},
		{"nested string bound", `{"profile":{"name":"Al","roles":["reader"]},"choice":2}`, true},
		{"array item enum", `{"profile":{"name":"Ada","roles":["owner"]},"choice":2}`, true},
		{"array minimum", `{"profile":{"name":"Ada","roles":[]},"choice":2}`, true},
		{"oneOf numeric bound", `{"profile":{"name":"Ada","roles":["reader"]},"choice":0}`, true},
		{"oneOf string pattern", `{"profile":{"name":"Ada","roles":["reader"]},"choice":"UPPER"}`, true},
		{"root additional property", `{"profile":{"name":"Ada","roles":["reader"]},"choice":2,"extra":true}`, true},
		{"object schema rejects scalar args", `[]`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := NewToolSet().validateToolArgs("<test>", tc.args, schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateToolArgs(%s) err=%v, wantErr=%v", tc.args, err, tc.wantErr)
			}
		})
	}
}

func TestValidateArgs_InvalidSchemaFailsClosed(t *testing.T) {
	invalid := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"mixed": map[string]any{"type": []any{"string", 7, "null"}},
		},
	}
	if err := NewToolSet().validateToolArgs("<test>", `{"mixed":"a"}`, invalid); err == nil || !strings.Contains(err.Error(), "does not compile") {
		t.Fatalf("invalid schema error = %v, want compile failure", err)
	}
}

func TestValidateArgs_ExternalRefFailsWithoutNetworkLoader(t *testing.T) {
	schema := map[string]any{"$ref": "https://example.invalid/tool.json"}
	if err := NewToolSet().validateToolArgs("<test>", `{}`, schema); err == nil || !strings.Contains(err.Error(), "no URLLoader set") {
		t.Fatalf("external ref error = %v, want disabled-loader failure", err)
	}
}

func TestToolValidatorCacheTracksSchemaFingerprint(t *testing.T) {
	ts := NewToolSet()
	first := map[string]any{
		"type": "object", "required": []string{"a"},
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
	}
	second := map[string]any{
		"type": "object", "required": []string{"b"},
		"properties": map[string]any{"b": map[string]any{"type": "integer"}},
	}
	if err := ts.validateToolArgs("dynamic", `{"a":"ok"}`, first); err != nil {
		t.Fatalf("first schema: %v", err)
	}
	if err := ts.validateToolArgs("dynamic", `{"a":"ok"}`, second); err == nil {
		t.Fatal("same-name schema change reused stale validator")
	}
	if err := ts.validateToolArgs("dynamic", `{"b":2}`, second); err != nil {
		t.Fatalf("second schema: %v", err)
	}
}

func TestToolValidatorCacheIsBounded(t *testing.T) {
	ts := NewToolSet()
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	for i := 0; i < maxToolValidatorCacheEntries+20; i++ {
		if err := ts.validateToolArgs(fmt.Sprintf("dynamic_%d", i), `{}`, schema); err != nil {
			t.Fatal(err)
		}
	}
	if got := len(ts.validators.entries); got > maxToolValidatorCacheEntries {
		t.Fatalf("validator cache grew to %d entries, cap=%d", got, maxToolValidatorCacheEntries)
	}
}

// TestLoopRejectsBadArgs verifies a schema-invalid tool call is blocked before
// execution, the precise reason reaches the model, and the tool never runs.
func TestLoopRejectsBadArgs(t *testing.T) {
	tool := &schemaTool{}
	recorder := &nativeRecorder{}
	faux := recordedNativeProvider(recorder,
		AssistantToolCall("c1", "run_query", `{"mode":"read"}`), // missing required sql
		AssistantText("sorry, I omitted the sql field"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("run_query"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "run a query")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.called != 0 {
		t.Fatalf("schema-invalid tool must not execute, got %d calls", tool.called)
	}
	if len(res.Tools) != 1 || res.Tools[0].Allowed {
		t.Fatalf("expected 1 blocked trace, got %+v", res.Tools)
	}
	// The second turn's request must carry the validation error as a tool result
	// so the model can self-correct.
	last := recorder.all()[recorder.count()-1]
	var sawError bool
	for _, m := range last.Messages {
		if m.Role == RoleTool && contains(m.Content, "required") {
			sawError = true
		}
	}
	if !sawError {
		t.Fatalf("expected validation error fed back to model, messages=%+v", last.Messages)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

// TestSerializeToolArgsFatValueDoesNotStarveSiblings is the defect this helper
// closes: with only the per-call cap, truncateMiddle kept the head and tail of
// the raw blob, so an argument sitting BETWEEN two fat values — its name
// included — never reached the summarizer, which was then asked to describe a
// call it could not read. Per-value capping first keeps every key.
func TestSerializeToolArgsFatValueDoesNotStarveSiblings(t *testing.T) {
	raw := `{"content":"` + bigText(40_000) + `","mode":"overwrite","patch":"` + bigText(40_000) + `"}`

	out := serializeToolArgs(raw)

	// The sandwiched argument is the one the old whole-blob truncation lost.
	if !strings.Contains(out, "mode") || !strings.Contains(out, "overwrite") {
		t.Fatalf("sandwiched argument was starved out of the serialization: %q", out)
	}
	for _, key := range []string{"content", "mode", "patch"} {
		if !strings.Contains(out, key+"=") {
			t.Fatalf("argument %q missing from the serialization: %q", key, out)
		}
	}
}

// TestSerializeToolArgsRespectsPerCallCap pins that the outer bound still
// holds: many fat values must not add up past maxSerializedToolArgs, which is
// what keeps the summarization request itself small.
func TestSerializeToolArgsRespectsPerCallCap(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{`)
	for i, key := range []string{"a", "b", "c", "d", "e", "f", "g", "h"} {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`"` + key + `":"` + bigText(5_000) + `"`)
	}
	b.WriteString(`}`)

	out := serializeToolArgs(b.String())

	if len(out) > maxSerializedToolArgs {
		t.Fatalf("serialized args exceeded the per-call cap: %d bytes", len(out))
	}
}

// TestSerializeToolArgsPerValueCap pins the inner bound independently of the
// outer one: a single fat value is cut to maxSerializedToolArgValue, not left
// to spend the whole call budget.
func TestSerializeToolArgsPerValueCap(t *testing.T) {
	out := serializeToolArgs(`{"q":"` + bigText(50_000) + `"}`)

	if len(out) > maxSerializedToolArgValue+64 {
		t.Fatalf("single value was not capped per-value: %d bytes", len(out))
	}
	if !strings.Contains(out, "truncated") {
		t.Fatalf("expected a truncation marker: %q", out)
	}
}

// TestSerializeToolArgsDeterministicKeyOrder guards a cache property, not a
// style preference: the summarization request rides the provider's prefix
// cache, and Go's randomized map iteration would churn it on every compaction.
func TestSerializeToolArgsDeterministicKeyOrder(t *testing.T) {
	raw := `{"zeta":1,"alpha":2,"mu":3,"beta":4}`

	first := serializeToolArgs(raw)
	if want := `alpha=2, beta=4, mu=3, zeta=1`; first != want {
		t.Fatalf("keys not rendered in sorted order:\n got %q\nwant %q", first, want)
	}
	for i := 0; i < 50; i++ {
		if got := serializeToolArgs(raw); got != first {
			t.Fatalf("rendering not deterministic on run %d: %q != %q", i, got, first)
		}
	}
}

// TestSerializeToolArgsNonObjectFallsThrough keeps the documented escape hatch:
// arguments that are not a JSON object behave byte-for-byte as before the
// per-value cap existed.
func TestSerializeToolArgsNonObjectFallsThrough(t *testing.T) {
	cases := []string{
		`"just a string"`,
		`[1,2,3]`,
		`42`,
		`null`,
		``,
		`{"unterminated": `,
		`not json at all`,
	}
	for _, raw := range cases {
		if got, want := serializeToolArgs(raw), truncateMiddle(raw, maxSerializedToolArgs); got != want {
			t.Fatalf("non-object %q: got %q, want the old whole-blob truncation %q", raw, got, want)
		}
	}
}

// TestSerializeToolArgsMalformedOversizedFallsThrough covers the fallback on a
// blob big enough for the old path to actually truncate, so the fall-through is
// pinned against truncateMiddle's output and not just against identity.
func TestSerializeToolArgsMalformedOversizedFallsThrough(t *testing.T) {
	raw := `{"path":"x.ts","content":"` + bigText(40_000) // never closed

	got := serializeToolArgs(raw)

	if want := truncateMiddle(raw, maxSerializedToolArgs); got != want {
		t.Fatalf("malformed JSON did not fall through unchanged:\n got %q\nwant %q", got, want)
	}
	if len(got) > maxSerializedToolArgs {
		t.Fatalf("fallback exceeded the per-call cap: %d bytes", len(got))
	}
}

// TestTruncatedResponseNeverExecutesTools pins the truncated-output guard (pi's
// failToolCallsFromTruncatedMessage): a response cut off by the output-token
// limit can carry tool calls whose arguments are silently incomplete — the
// stream ended mid-JSON and the truncated tail may still parse. None of them
// run; each is answered with an error telling the model to re-issue, and the
// refusal is traced as not-allowed without spending tool-call budget.
func TestTruncatedResponseNeverExecutesTools(t *testing.T) {
	tool := &echoTool{name: "write_file"}
	faux := scriptedNativeProvider(
		ChatResponse{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "c1", Name: "write_file", Arguments: `{"path":"a.go","content":"package main`},
			}},
			StopReason: "length",
		},
		AssistantText("re-issued with complete arguments"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("write_file"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "write the file")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.called != 0 {
		t.Fatalf("a tool call from a truncated response must never execute, got %d calls", tool.called)
	}
	if res.Final != "re-issued with complete arguments" {
		t.Fatalf("final = %q", res.Final)
	}
	if len(res.Tools) != 1 || res.Tools[0].Allowed {
		t.Fatalf("expected 1 refused trace, got %+v", res.Tools)
	}
	// The refusal reaches the model as a tool result, so the transcript stays
	// provider-valid and the model can re-issue the call.
	var sawRefusal bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && m.ToolCallID == "c1" && strings.Contains(m.Content, "truncated") {
			sawRefusal = true
		}
	}
	if !sawRefusal {
		t.Fatal("truncated call was not answered with a re-issue refusal")
	}
}

// TestTruncatedStopReasons covers the vendor-verbatim spellings the guard must
// catch: OpenAI-wire "length", Anthropic's "max_tokens", Codex's "incomplete".
func TestTruncatedStopReasons(t *testing.T) {
	for _, reason := range []string{"length", "max_tokens", "incomplete"} {
		if !isTruncatedStop(reason) {
			t.Fatalf("stop reason %q must be recognized as truncated", reason)
		}
	}
	for _, reason := range []string{"stop", "tool_calls", "error", "aborted", ""} {
		if isTruncatedStop(reason) {
			t.Fatalf("stop reason %q is not a truncation", reason)
		}
	}
}

// TestTruncatedFinalAnswerStillReturned verifies the guard only intercepts
// tool calls: a length-stopped response with no calls is still the run's
// answer (truncated text is better than none, and nothing unsafe can run).
func TestTruncatedFinalAnswerStillReturned(t *testing.T) {
	faux := scriptedNativeProvider(ChatResponse{
		Message:    Message{Role: RoleAssistant, Content: "partial ans"},
		StopReason: "length",
	})
	agent, err := New(Config{NativeProvider: faux, Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Prompt(context.Background(), "say something")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if res.Final != "partial ans" {
		t.Fatalf("final = %q, want the truncated text", res.Final)
	}
}



func TestTruncateMiddleKeepsHeadAndTail(t *testing.T) {
	s := "HEAD-" + strings.Repeat("x", 4096) + "-TAIL"
	got := truncateMiddle(s, 512)
	if len(got) > 512 {
		t.Fatalf("result exceeds budget: %d bytes", len(got))
	}
	if !strings.HasPrefix(got, "HEAD-") {
		t.Fatalf("head lost: %q", got[:16])
	}
	if !strings.HasSuffix(got, "-TAIL") {
		t.Fatalf("tail lost: %q", got[len(got)-16:])
	}
	if !strings.Contains(got, "bytes truncated") {
		t.Fatalf("missing omission marker: %q", got)
	}
}

func TestTruncateMiddleNoopWithinBudget(t *testing.T) {
	s := "short result"
	if got := truncateMiddle(s, 1024); got != s {
		t.Fatalf("modified in-budget string: %q", got)
	}
	if got := truncateMiddle(s, 0); got != s {
		t.Fatalf("maxBytes 0 must disable truncation: %q", got)
	}
}

func TestTruncateMiddleUTF8Safe(t *testing.T) {
	s := strings.Repeat("héllo wörld ", 400)
	for budget := 100; budget <= 400; budget += 37 {
		got := truncateMiddle(s, budget)
		if !utf8.ValidString(got) {
			t.Fatalf("budget %d produced invalid UTF-8", budget)
		}
		if len(got) > budget {
			t.Fatalf("budget %d exceeded: %d bytes", budget, len(got))
		}
	}
}

// TestReasoningEffortThreadedIntoRequests proves the per-agent knob reaches
// every provider call of a run.
func TestReasoningEffortThreadedIntoRequests(t *testing.T) {
	recorder := &nativeRecorder{}
	agent, err := New(Config{
		NativeProvider:  recordedNativeProvider(recorder, AssistantText("ok")),
		Model:           "faux-1",
		Tools:           NewToolSet(),
		Policy:          NewAllowList(),
		ReasoningEffort: "high",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(t.Context(), "hello"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	got := recorder.all()
	if len(got) == 0 || got[0].Options["reasoning"] != "high" {
		t.Fatalf("reasoning effort not threaded: %+v", got)
	}
}

// recordingTool captures the exact argument string it was handed, so a test can
// assert what reached the tool vs. what was traced.
type recordingTool struct {
	name     string
	lastArgs string
	called   int
}

func (r *recordingTool) Name() string { return r.name }
func (r *recordingTool) Schema() ToolSchema {
	return ToolSchema{Name: r.name, Description: "rec", Parameters: map[string]any{"type": "object"}}
}
func (r *recordingTool) Run(_ context.Context, args string) (string, error) {
	r.called++
	r.lastArgs = args
	return "ok", nil
}

// stubResolver is a test CredentialResolver driven by a func.
type stubResolver struct {
	resolve func(string) (string, error)
}

func (s stubResolver) Resolve(_ context.Context, args string) (string, error) {
	return s.resolve(args)
}

// TestCredentialResolverInjectsAtTrustBoundary proves the core F7 property: the
// resolved secret reaches the tool, but the trace (and therefore the persisted
// record and the model-visible call) keeps the {{cred:NAME}} placeholder — the
// literal is never observable outside the executing tool.
func TestCredentialResolverInjectsAtTrustBoundary(t *testing.T) {
	tool := &recordingTool{name: "call_api"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "call_api", `{"key":"{{cred:API_KEY}}"}`),
		AssistantText("done"),
	)
	env := DefaultEnv()
	env.Credentials = stubResolver{resolve: func(args string) (string, error) {
		return strings.ReplaceAll(args, "{{cred:API_KEY}}", "sk-secret-value"), nil
	}}
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("call_api"),
		Env:            &env,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// The tool received the resolved secret.
	if want := `{"key":"sk-secret-value"}`; tool.lastArgs != want {
		t.Fatalf("tool args: want %q, got %q", want, tool.lastArgs)
	}
	// The trace kept the placeholder — no secret leaked into the persisted record.
	if len(res.Tools) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(res.Tools))
	}
	if got := res.Tools[0].Args; got != `{"key":"{{cred:API_KEY}}"}` {
		t.Fatalf("trace args leaked or changed: %q", got)
	}
	if strings.Contains(res.Tools[0].Args, "sk-secret-value") {
		t.Fatal("secret value leaked into the tool trace")
	}
}

// TestCredentialResolverFailsClosed verifies a resolver error blocks the call
// (the tool never runs) and the reason is returned to the model.
func TestCredentialResolverFailsClosed(t *testing.T) {
	tool := &recordingTool{name: "call_api"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "call_api", `{"key":"{{cred:MISSING}}"}`),
		AssistantText("understood"),
	)
	env := DefaultEnv()
	env.Credentials = stubResolver{resolve: func(string) (string, error) {
		return "", errors.New("unknown credential \"MISSING\"")
	}}
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("call_api"),
		Env:            &env,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if tool.called != 0 {
		t.Fatalf("tool must not run when resolution fails, got %d calls", tool.called)
	}
	if len(res.Tools) != 1 || res.Tools[0].Allowed {
		t.Fatalf("expected 1 blocked trace, got %+v", res.Tools)
	}
	var sawBlock bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && strings.Contains(m.Content, "blocked:") {
			sawBlock = true
		}
	}
	if !sawBlock {
		t.Fatal("block reason was not returned to the model")
	}
}

// TestNoCredentialResolverPassesArgsThrough is the default-off path: with no
// resolver wired, arguments reach the tool byte-for-byte unchanged.
func TestNoCredentialResolverPassesArgsThrough(t *testing.T) {
	tool := &recordingTool{name: "call_api"}
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", "call_api", `{"key":"{{cred:API_KEY}}"}`),
		AssistantText("done"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Tools:          NewToolSet(tool),
		Policy:         NewAllowList("call_api"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := agent.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if want := `{"key":"{{cred:API_KEY}}"}`; tool.lastArgs != want {
		t.Fatalf("args should pass through unchanged: want %q, got %q", want, tool.lastArgs)
	}
}
