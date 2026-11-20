package sandbox

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

// fixProbe is a host tool for eval cells: ms sleeps, bytes pads the text, img
// shows that many pictures, fail errors, ext marks the result untrusted.
type fixProbe struct {
	name     string
	parallel bool
}

func (p *fixProbe) Name() string   { return p.name }
func (p *fixProbe) Parallel() bool { return p.parallel }
func (p *fixProbe) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: p.name, Description: "probe", Parameters: map[string]any{"type": "object", "properties": map[string]any{
		"ms": map[string]any{"type": "integer"}, "bytes": map[string]any{"type": "integer"}, "value": map[string]any{"type": "string"},
		"fail": map[string]any{"type": "boolean"}, "img": map[string]any{"type": "integer"}, "ext": map[string]any{"type": "boolean"},
	}}}
}
func (p *fixProbe) Run(ctx context.Context, args string) (string, error) {
	out, err := p.RunRich(ctx, args)
	return out.Content, err
}
func (p *fixProbe) RunRich(ctx context.Context, args string) (agentcore.ToolOutput, error) {
	var in struct {
		Ms, Bytes, Img int
		Value          string
		Fail, Ext      bool
	}
	_ = json.Unmarshal([]byte(args), &in)
	if in.Ms > 0 {
		select {
		case <-time.After(time.Duration(in.Ms) * time.Millisecond):
		case <-ctx.Done():
			return agentcore.ToolOutput{}, ctx.Err()
		}
	}
	if in.Fail {
		return agentcore.ToolOutput{}, errors.New("boom from tool")
	}
	out := agentcore.ToolOutput{Content: "ok:" + in.Value + strings.Repeat("z", in.Bytes), External: in.Ext}
	for i := 0; i < in.Img; i++ {
		out.Parts = append(out.Parts, agentcore.ContentPart{Type: agentcore.ContentPartImage, MIMEType: "image/png", Data: testEvalPNG})
	}
	return out, nil
}

// softEval answers a failed cell as its result, as 2agent's eval wrapper does,
// so a test sees what the model would: the error text and the pictures.
type softEval struct{ *EvalTool }

func (s softEval) Run(ctx context.Context, args string) (string, error) {
	out, err := s.RunRich(ctx, args)
	return out.Content, err
}
func (s softEval) RunRich(ctx context.Context, args string) (agentcore.ToolOutput, error) {
	out, err := s.EvalTool.RunRich(ctx, args)
	if err != nil && ctx.Err() == nil {
		out.Content = err.Error()
		return out, nil
	}
	return out, err
}

type fixCell struct {
	code    string
	timeout int
	reset   bool
}

// fixRun runs cells as successive eval calls of one conversation inside an
// agent run (so they can call host tools) and returns each call's message.
func fixRun(t *testing.T, language string, cfg EvalConfig, limits *agentcore.Limits, cells ...fixCell) []agentcore.Message {
	t.Helper()
	var eval *EvalTool
	if language == "javascript" {
		eval, _, _ = testJavaScriptEvalTool(t, cfg)
	} else {
		eval, _, _ = testEvalTool(t, cfg)
	}
	var responses []agentcore.ChatResponse
	for i, cell := range cells {
		args := map[string]any{"language": language, "code": cell.code}
		if cell.timeout > 0 {
			args["timeout_seconds"] = cell.timeout
		}
		if cell.reset {
			args["reset"] = true
		}
		encoded, _ := json.Marshal(args)
		responses = append(responses, agentcore.AssistantToolCall(fmt.Sprintf("c%d", i), ToolEval, string(encoded)))
	}
	responses = append(responses, agentcore.AssistantText("done"))
	agent, err := agentcore.New(agentcore.Config{
		NativeProvider: sandboxNativeFixture(responses...), Model: "faux",
		Tools:  agentcore.NewToolSet(softEval{eval}, &fixProbe{name: "p"}, &fixProbe{name: "pp", parallel: true}),
		Policy: agentcore.NewAllowList(ToolEval, "p", "pp"),
		Limits: limits,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(evalContext("fix-"+t.Name()), "go")
	if err != nil {
		t.Fatal(err)
	}
	var out []agentcore.Message
	for _, message := range result.Messages {
		if message.Role == agentcore.RoleTool && message.Name == ToolEval {
			out = append(out, message)
		}
	}
	if len(out) != len(cells) {
		t.Fatalf("got %d eval results for %d cells", len(out), len(cells))
	}
	return out
}

func mustContain(t *testing.T, what, text string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(text, want) {
			t.Fatalf("%s = %q, want it to contain %q", what, text, want)
		}
	}
}

func mustNotContain(t *testing.T, what, text string, unwanted ...string) {
	t.Helper()
	for _, bad := range unwanted {
		if strings.Contains(text, bad) {
			t.Fatalf("%s = %q, must not contain %q", what, text, bad)
		}
	}
}

// A kernel that is not reading its stdin (a busy cell) must not keep the host
// from killing it: the result writes it never drained used to hold the write
// lock that closing the process waited for, so a timed-out cell hung the whole
// call and left the kernel spinning.
func TestEvalTimeoutOfABusyKernelDoesNotHang(t *testing.T) {
	done := make(chan []agentcore.Message, 1)
	go func() {
		done <- fixRun(t, "javascript", EvalConfig{TimeoutSeconds: 2}, nil,
			fixCell{code: `for (let i = 0; i < 6; i++) tool.pp({bytes: 30000}); const t0 = Date.now(); while (Date.now() - t0 < 100000) {}`},
			fixCell{code: `1 + 1`})
	}()
	select {
	case messages := <-done:
		mustContain(t, "timed out cell", messages[0].Content, "timed out")
		mustContain(t, "next cell", messages[1].Content, "2", "[new JavaScript session")
	case <-time.After(20 * time.Second):
		t.Fatal("eval never returned after a cell timed out with its kernel not reading stdin")
	}
}

// A Python display or final expression too large for a frame is cut with a
// note, as JavaScript's is, and does not cost the kernel.
func TestEvalPythonDisplayIsBounded(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil,
		fixCell{code: `big = "x" * 30_000_000; big`},
		fixCell{code: `display("y" * 30_000_000); len(big)`},
		fixCell{code: `display({"rows": ["r" * 100] * 5000}); 1`},
	)
	mustContain(t, "final expression", messages[0].Content, "display truncated", "narrower slice")
	mustNotContain(t, "final expression", messages[0].Content, "kernel discarded", "[new Python")
	mustContain(t, "display", messages[1].Content, "display truncated", "30000000")
	mustNotContain(t, "display", messages[1].Content, "kernel discarded", "[new Python")
	mustContain(t, "json display", messages[2].Content, "rrrr")
}

// A protocol line over the frame bound is dropped, not read whole into memory
// and not fatal: a tool call too large to read is answered with an error.
func TestEvalOversizeFramesFailCleanly(t *testing.T) {
	for _, test := range []struct{ language, code string }{
		{"python", "kept = 1\ntry:\n    tool('p', {'value': 'x' * (26 * 1024 * 1024)})\nexcept Exception as e:\n    print(type(e).__name__, e.kind, str(e)[:100])\nkept"},
		{"javascript", "var kept = 1\ntry { await tool.p({value: 'x'.repeat(26 * 1024 * 1024)}) } catch (e) { console.log(e.name, e.kind, e.message.slice(0, 100)) }\nkept"},
	} {
		t.Run(test.language, func(t *testing.T) {
			messages := fixRun(t, test.language, EvalConfig{TimeoutSeconds: 20}, nil, fixCell{code: test.code}, fixCell{code: `kept`})
			mustContain(t, "oversize call", messages[0].Content, "ToolError invalid", "frame bound")
			mustContain(t, "next cell", messages[1].Content, "1")
			mustNotContain(t, "next cell", messages[1].Content, "[new ")
		})
	}
}

func TestReadEvalLineBoundsWhatItHolds(t *testing.T) {
	line := `{"type": "tool_call", "id": "cell-3", "request_id": "cell-3/tool-1", "name": "p", "arguments": {"v": "` + strings.Repeat("x", 5000) + `"}}` + "\n"
	reader := bufioReader(line + "next\n")
	got, head, size, over, err := readEvalLine(reader, 1000)
	if err != nil || !over || got != nil || size != len(line) || len(head) != 512 {
		t.Fatalf("oversize line: got=%d over=%v size=%d head=%d err=%v", len(got), over, size, len(head), err)
	}
	frame := oversizeFrame(head, size)
	if frame.Type != "oversize" || frame.Name != "tool_call" || frame.ID != "cell-3" || frame.RequestID != "cell-3/tool-1" {
		t.Fatalf("oversize frame = %+v", frame)
	}
	got, _, _, over, err = readEvalLine(reader, 1000)
	if err != nil || over || string(got) != "next\n" {
		t.Fatalf("line after an oversize one: %q over=%v err=%v", got, over, err)
	}
}

// When a cell times out its output so far and its pictures are still the
// model's only account of what it did.
func TestEvalTimeoutKeepsPartialOutputAndPictures(t *testing.T) {
	for _, test := range []struct{ language, code string }{
		{"python", "print('progress-marker')\ntool('p', {'img': 1, 'value': 'shot'})\nimport time\ntime.sleep(30)"},
		{"javascript", "console.log('progress-marker')\nawait tool.p({img: 1, value: 'shot'})\nawait new Promise(r => setTimeout(r, 30000))"},
	} {
		t.Run(test.language, func(t *testing.T) {
			messages := fixRun(t, test.language, EvalConfig{TimeoutSeconds: 2}, nil, fixCell{code: test.code})
			mustContain(t, "timeout", messages[0].Content, "timed out", "progress-marker")
			if len(messages[0].ContentParts) != 1 {
				t.Fatalf("pictures kept = %d, want 1: %q", len(messages[0].ContentParts), messages[0].Content)
			}
		})
	}
}

// A kernel that died between cells is replaced and the cell runs: it never
// reached the dead one, so "not replayed" would be a lie.
func TestEvalKernelDiedWhileIdleRunsTheCellOnAFreshKernel(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("died-idle")
	if out, err := tool.Run(ctx, `{"language":"python","code":"import threading, os; threading.Timer(0.2, lambda: os._exit(1)).start(); v = 1"}`); err != nil || strings.Contains(out, "[new") {
		t.Fatalf("first cell: %q %v", out, err)
	}
	time.Sleep(800 * time.Millisecond)
	out, err := tool.Run(ctx, `{"language":"python","code":"2 + 2"}`)
	if err != nil {
		t.Fatalf("cell after the kernel died: %v", err)
	}
	mustContain(t, "cell after death", out, "[new Python session", "4")
}

// A kernel whose stdin is gone before the cell is written never saw the cell,
// so the cell runs once on a fresh kernel.
func TestEvalCellNotDeliveredRunsOnAFreshKernel(t *testing.T) {
	tool, registry, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("not-delivered")
	if _, err := tool.Run(ctx, `{"language":"python","code":"v = 1"}`); err != nil {
		t.Fatal(err)
	}
	key := tool.sessionKey(ctx, "python")
	registry.mu.Lock()
	kernel := registry.sessions[key]
	registry.mu.Unlock()
	_ = kernel.process.in.Close() // the next write fails, whether or not the process has exited by then
	out, err := tool.Run(ctx, `{"language":"python","code":"3 + 4"}`)
	if err != nil {
		t.Fatalf("cell after the kernel lost its stdin: %v", err)
	}
	mustContain(t, "cell on a fresh kernel", out, "[new Python session", "7")

	broken, err := tool.startProcess(evalRuntimeSpec{name: "python", label: "Python", command: tool.config.Python.Command, runner: pythonEvalRunner})
	if err != nil {
		t.Fatal(err)
	}
	defer broken.terminate()
	_ = broken.in.Close()
	if _, err := broken.execute(context.Background(), "1", 4096, 1, nil); !errors.Is(err, errEvalNotSent) {
		t.Fatalf("execute on a kernel without stdin = %v, want errEvalNotSent", err)
	}
}

func TestEvalPythonKeepsPrintAndDisplayInOrder(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil, fixCell{code: "for i in range(25):\n    print('p%d' % i)\n    display({'d': i})\nprint('end')"})
	content := messages[0].Content
	at := -1
	for i := 0; i < 25; i++ {
		for _, marker := range []string{fmt.Sprintf("p%d\n", i), fmt.Sprintf("\"d\": %d\n", i)} {
			next := strings.Index(content[at+1:], marker)
			if next < 0 {
				t.Fatalf("marker %q out of order or missing after offset %d in %q", marker, at, content)
			}
			at += 1 + next
		}
	}
}

func TestEvalOutputKeepsUTF8AtChunkBoundaries(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil, fixCell{code: "print('a' + 'é' * 12000)"})
	if strings.Contains(messages[0].Content, "\ufffd") || strings.Count(messages[0].Content, "é") != 12000 {
		t.Fatalf("python output lost characters: %d replacement characters, %d é", strings.Count(messages[0].Content, "\ufffd"), strings.Count(messages[0].Content, "é"))
	}
	for _, pad := range []int{8190, 8191} {
		messages = fixRun(t, "javascript", EvalConfig{}, nil, fixCell{code: fmt.Sprintf("console.log('a'.repeat(%d) + String.fromCodePoint(0x1F600).repeat(4)); 0", pad)})
		if strings.Contains(messages[0].Content, "\ufffd") || strings.Count(messages[0].Content, "😀") != 4 {
			t.Fatalf("javascript output (pad %d) lost characters: %d replacement characters, %d emoji", pad, strings.Count(messages[0].Content, "\ufffd"), strings.Count(messages[0].Content, "😀"))
		}
	}
}

// Host tools can be called from threads, concurrently with tool_all and
// atool, and from JavaScript with Promise.all, when the tool may run beside
// itself (a ParallelTool).
func TestEvalHostToolCallsRunConcurrently(t *testing.T) {
	const cell = `
import time, asyncio
from concurrent.futures import ThreadPoolExecutor
t0 = time.time()
batch = tool_all([("pp", {"ms": 400, "value": str(i)}) for i in range(3)])
t_all = time.time() - t0
t0 = time.time()
async def one(i): return await atool("pp", {"ms": 400, "value": "a%d" % i})
gathered = await asyncio.gather(*[one(i) for i in range(3)])
t_gather = time.time() - t0
t0 = time.time()
with ThreadPoolExecutor(3) as pool:
    threaded = list(pool.map(lambda i: tool("pp", {"ms": 400, "value": "t%d" % i}), range(3)))
t_threads = time.time() - t0
failed = tool_all([("pp", {"fail": True}), ("pp", {"value": "fine"})])
print(batch, gathered, threaded, [type(f).__name__ for f in failed], failed[0].kind, failed[1])
print("fast", max(t_all, t_gather, t_threads) < 1.0, round(max(t_all, t_gather, t_threads), 2))
`
	messages := fixRun(t, "python", EvalConfig{TimeoutSeconds: 20}, nil, fixCell{code: cell})
	mustContain(t, "python concurrency", messages[0].Content, "'ok:0', 'ok:1', 'ok:2'", "'ok:a0'", "'ok:t2'", "['ToolError', 'str'] failed ok:fine", "fast True")

	messages = fixRun(t, "javascript", EvalConfig{TimeoutSeconds: 20}, nil, fixCell{code: `
const t0 = Date.now();
const all = await Promise.all([0, 1, 2].map(i => tool.pp({ms: 400, value: "j" + i})));
console.log(all.join(","), Date.now() - t0 < 1000);`})
	mustContain(t, "javascript concurrency", messages[0].Content, "ok:j0,ok:j1,ok:j2", "true")
}

func TestEvalJavaScriptDeclarationsCanBeRepeatedAcrossCells(t *testing.T) {
	cells := []string{
		"const x = 1; x", "const x = 2; x",
		"let y = 1; y", "let y = 2; y",
		"class A { v() { return 1 } }\nnew A().v()", "class A { v() { return 2 } }\nnew A().v()",
		"class B extends A { v() { return super.v() + 10 } }\nnew B().v()", "class B extends A { v() { return super.v() + 20 } }\nnew B().v()",
		"import path from 'node:path'\npath.sep", "import path from 'node:path'\npath.sep",
		"import { join as j } from 'node:path'; j('a', 'b')", "import { join as j } from 'node:path'; j('a', 'c')",
		"const { a, b: c } = { a: 1, b: 2 }; a + c", "const { a, b: c } = { a: 10, b: 20 }; a + c",
		"const r = await Promise.resolve(5); r", "const r = await Promise.resolve(6); r",
		"const K = class Foo { }; typeof K", "const K = class Foo { }; typeof K",
		"const s = `const q = ${1}`; s",
		"// const hidden = 1\nconst t = 'let u = 2'; t",
		"const f = () => class Inner {}; typeof f()",
		"for (const i of [1, 2]) {} \nconst after = 3; after",
		"const re = /a\\/b/g; const half = 10 / 2 / 1; [re.test('a/b'), half].join()",
		"const o = { class: 1, let: 2, const: 3 }; o.class + o.let + o.const",
		"if (true) { const inner = 1; var outer = inner }\nouter",
		"const g = function* () { yield 1 }; [...g()].length",
		"let a1 = 1, b1 = 2\nclass Z {}\nclass Z2 extends Z {}\n[a1, b1, typeof Z2].join()",
	}
	limits := agentcore.DefaultLimits()
	limits.MaxTurns, limits.MaxToolCalls = 100, 100
	messages := fixRun(t, "javascript", EvalConfig{}, &limits, func() []fixCell {
		var out []fixCell
		for _, c := range cells {
			out = append(out, fixCell{code: c})
		}
		return out
	}()...)
	want := []string{"1", "2", "1", "2", "1", "2", "12", "22", "/", "/", "a/b", "a/c", "3", "30", "5", "6", "function", "function", "const q = 1", "let u = 2", "function", "3", "true,5", "6", "1", "1", "1,2,function"}
	for i, message := range messages {
		if strings.Contains(message.Content, "failed") || !strings.Contains(message.Content, want[i]) {
			t.Fatalf("cell %d %q = %q, want %q", i, cells[i], message.Content, want[i])
		}
	}
}

func TestEvalScheduleAllowsTheMaximumCellTimeout(t *testing.T) {
	cfg, err := ParseEvalConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TimeoutSeconds != 30 || cfg.MaxTimeoutSeconds != MaxEvalCellTimeoutSeconds {
		t.Fatalf("default config = %+v", cfg)
	}
	tool, _, _ := testEvalTool(t, cfg)
	schema := tool.Schema()
	property := schema.Parameters["properties"].(map[string]any)["timeout_seconds"].(map[string]any)
	if property["maximum"] != MaxEvalCellTimeoutSeconds {
		t.Fatalf("timeout_seconds maximum = %v, want %d", property["maximum"], MaxEvalCellTimeoutSeconds)
	}
	for _, want := range []string{"200 calls per cell", "ToolError", "tool_all", "atool", "Promise.all", "1.5 MB", "may be up to 300"} {
		mustContain(t, "description", schema.Description, want)
	}
	if out, err := tool.Run(evalContext("long-timeout"), `{"language":"python","code":"1+1","timeout_seconds":120}`); err != nil || !strings.Contains(out, "2") {
		t.Fatalf("a 120 s timeout was refused: %q %v", out, err)
	}
	if _, err := tool.Run(evalContext("long-timeout"), `{"language":"python","code":"1+1","timeout_seconds":301}`); err == nil || !strings.Contains(err.Error(), "between 1 and 300") {
		t.Fatalf("a 301 s timeout was allowed: %v", err)
	}
}

func TestEvalToolErrorsNameTheirKind(t *testing.T) {
	limits := agentcore.DefaultLimits()
	limits.MaxToolCalls = 4
	py := `
def kind(**args):
    try:
        tool("p", args)
    except ToolError as e:
        return (type(e).__name__, isinstance(e, RuntimeError), e.kind)
    return "no error"
print(kind(fail=True))
print(kind(ms="slow"))
print(kind(value="x"), kind(value="x"), kind(value="x"))
try:
    tool("nope", {})
except ToolError as e:
    print(e.kind)
try:
    tool("bad name", {})
except ToolError as e:
    print(e.kind)
`
	messages := fixRun(t, "python", EvalConfig{MaxBridgeCalls: 6}, &limits, fixCell{code: py})
	mustContain(t, "python kinds", messages[0].Content, "('ToolError', True, 'failed')", "('ToolError', True, 'blocked')", "'budget'")
	mustContain(t, "python unknown tool and bad name", messages[0].Content, "blocked", "invalid")

	js := `
const kind = async (name, args) => { try { await tool(name, args) } catch (e) { return [e.name, e instanceof Error, e.kind].join("/") } return "no error" };
console.log(await kind("p", {fail: true}));
console.log(await kind("p", {ms: "slow"}));
console.log(await kind("bad name", {}));
let n = 0, last = "";
for (let i = 0; i < 7; i++) last = await kind("p", {value: "x"});
console.log(last);`
	messages = fixRun(t, "javascript", EvalConfig{MaxBridgeCalls: 4}, nil, fixCell{code: js})
	mustContain(t, "javascript kinds", messages[0].Content, "ToolError/true/failed", "ToolError/true/blocked", "ToolError/true/invalid", "ToolError/true/limit")
}

func TestEvalHostToolResultOverTheBoundIsRefusedNotCut(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil, fixCell{code: `
small = tool("p", {"bytes": 200000})
print(len(small), small.endswith("z"))
try:
    tool("p", {"bytes": 1700000})
except ToolError as e:
    print(e.kind, str(e)[:200])`})
	mustContain(t, "result sizes", messages[0].Content, "200003 True", "limit host tool result is", "ask the tool for less")
}

func TestEvalExternalContentReachesTheOutput(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil,
		fixCell{code: `print(tool("p", {"value": "plain"}))`},
		fixCell{code: `print(tool("p", {"value": "fetched page", "ext": True}))`},
		fixCell{code: `tool("p", {"ext": True}); raise ValueError("after an untrusted call")`},
	)
	mustNotContain(t, "trusted cell", messages[0].Content, "<external_content")
	mustContain(t, "untrusted cell", messages[1].Content, `<external_content tool="eval">`, "fetched page")
	mustContain(t, "failed untrusted cell", messages[2].Content, `<external_content tool="eval">`, "ValueError")
}

func TestEvalNewSessionNoticeOnlyAfterALostKernel(t *testing.T) {
	python, err := execLookPath("python3")
	if err != nil {
		t.Skip("python3 is not available")
	}
	cfg := EvalConfig{TimeoutSeconds: 2}
	cfg.Python.Command = python
	registry := NewEvalSessionRegistry(4, 300*time.Millisecond)
	t.Cleanup(registry.Close)
	tool, err := NewEvalTool(NewHostSandbox(), mustWorkspace(t), registry, "ns", cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := evalContext("notice")
	run := func(args string) string {
		t.Helper()
		out, _ := tool.Run(ctx, args)
		return out
	}
	mustNotContain(t, "first cell of a session", run(`{"language":"python","code":"v = 1"}`), "[new")
	mustNotContain(t, "second cell", run(`{"language":"python","code":"v"}`), "[new")
	time.Sleep(900 * time.Millisecond) // the idle reaper takes the kernel
	mustContain(t, "cell after the idle reap", run(`{"language":"python","code":"1"}`), "[new Python session")
	mustNotContain(t, "cell after the notice", run(`{"language":"python","code":"1"}`), "[new")
	run(`{"language":"python","code":"import time; time.sleep(30)"}`) // times out
	mustContain(t, "cell after a timeout", run(`{"language":"python","code":"1"}`), "[new Python session")
	run(`{"language":"python","code":"1"}`)
	mustNotContain(t, "reset after use", run(`{"language":"python","code":"1","reset":true}`), "[new")
}

func TestEvalTracebacksOmitTheRunnersOwnFrames(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil,
		fixCell{code: "def f():\n    return 1 / 0\nf()"},
		fixCell{code: "tool('p', {'fail': True})"},
		fixCell{code: "def broken(:\n    pass"},
		fixCell{code: "exec('1/0')"},
	)
	mustContain(t, "python traceback", messages[0].Content, `File "<cell>", line 3`, `File "<cell>", line 2, in f`, "ZeroDivisionError")
	mustNotContain(t, "python traceback", messages[0].Content, "<string>", "_run_cell")
	mustContain(t, "tool traceback", messages[1].Content, `File "<cell>", line 1`, "ToolError")
	mustNotContain(t, "tool traceback", messages[1].Content, "<string>", "_call_tool")
	mustContain(t, "syntax error", messages[2].Content, "SyntaxError")
	mustNotContain(t, "syntax error", messages[2].Content, "_run_cell")
	mustContain(t, "exec frame", messages[3].Content, `File "<string>", line 1`) // the cell's own exec stays

	messages = fixRun(t, "javascript", EvalConfig{}, nil,
		fixCell{code: "function f() { throw new Error('inside f') }\nf()"},
		fixCell{code: "require('node:fs').readFileSync('/no/such/file')"},
	)
	mustContain(t, "javascript stack", messages[0].Content, "inside f", "at f (<cell>")
	mustNotContain(t, "javascript stack", messages[0].Content, "node:repl", "node:vm", "ContextifyScript", "[eval]", "node:domain")
	mustContain(t, "library frame stays", messages[1].Content, "ENOENT", "node:fs")
}

func TestEvalOutputHeadersOnlyWhereTheyInform(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{}, nil,
		fixCell{code: "print('only stdout')"},
		fixCell{code: "1 + 1"},
		fixCell{code: "print('out'); 3"},
		fixCell{code: "import sys; print('warn', file=sys.stderr)"},
		fixCell{code: "pass"},
	)
	if got := messages[0].Content; got != "only stdout" {
		t.Fatalf("stdout only = %q", got)
	}
	if got := messages[1].Content; got != "2" {
		t.Fatalf("result only = %q", got)
	}
	mustContain(t, "stdout and result", messages[2].Content, "[stdout]\nout", "[result]\n3")
	if got := messages[3].Content; got != "[stderr]\nwarn" {
		t.Fatalf("stderr only = %q", got)
	}
	if got := messages[4].Content; got != "(no output)" {
		t.Fatalf("no output = %q", got)
	}
}

func TestEvalTruncationNoteSaysHowToGetTheRest(t *testing.T) {
	messages := fixRun(t, "python", EvalConfig{MaxOutputBytes: 4096}, nil, fixCell{code: "print('H' * 3000); print('M' * 3000); print('T' * 3000)"})
	mustContain(t, "bounded output", messages[0].Content, "bytes omitted", "narrower slice", "write the rest to a file", "HHHH", "TTTT")
	if len(messages[0].Content) > 4400 {
		t.Fatalf("output %d bytes, want a bound near 4096", len(messages[0].Content))
	}
}

func bufioReader(s string) *bufio.Reader { return bufio.NewReaderSize(strings.NewReader(s), 16) }

func execLookPath(name string) (string, error) { return exec.LookPath(name) }
