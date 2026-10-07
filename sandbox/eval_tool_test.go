package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func testEvalTool(t *testing.T, cfg EvalConfig) (*EvalTool, *EvalSessionRegistry, *Workspace) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not available")
	}
	ws := mustWorkspace(t)
	cfg.Python.Command = python
	if cfg.TimeoutSeconds == 0 {
		cfg.TimeoutSeconds = 5
	}
	if cfg.MaxOutputBytes == 0 {
		cfg.MaxOutputBytes = 50 * 1024
	}
	registry := NewEvalSessionRegistry(4, time.Minute)
	t.Cleanup(registry.Close)
	tool, err := NewEvalTool(NewHostSandbox(), ws, registry, "tenant\x00project\x00agent", cfg)
	if err != nil {
		t.Fatalf("NewEvalTool: %v", err)
	}
	return tool, registry, ws
}

func testJavaScriptEvalTool(t *testing.T, cfg EvalConfig) (*EvalTool, *EvalSessionRegistry, *Workspace) {
	t.Helper()
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not available")
	}
	ws := mustWorkspace(t)
	cfg.JavaScript.Command = node
	if cfg.TimeoutSeconds == 0 {
		cfg.TimeoutSeconds = 5
	}
	if cfg.MaxOutputBytes == 0 {
		cfg.MaxOutputBytes = 50 * 1024
	}
	registry := NewEvalSessionRegistry(4, time.Minute)
	t.Cleanup(registry.Close)
	tool, err := NewEvalTool(NewHostSandbox(), ws, registry, "tenant\x00project\x00agent", cfg)
	if err != nil {
		t.Fatalf("NewEvalTool: %v", err)
	}
	return tool, registry, ws
}

func requireNodeEvalTypeScript(t *testing.T, moduleHooks bool) {
	t.Helper()
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not available")
	}
	condition := `typeof require("node:module").stripTypeScriptTypes === "function"`
	if moduleHooks {
		condition += ` && typeof require("node:module").registerHooks === "function"`
	}
	if err := exec.Command(node, "-e", `process.exit(`+condition+` ? 0 : 1)`).Run(); err != nil {
		feature := "TypeScript transforms"
		if moduleHooks {
			feature = "TypeScript module hooks"
		}
		t.Skipf("installed Node does not support eval %s", feature)
	}
}

type captureEvalProcessSandbox struct {
	stubSandbox
	startErr error
}

func (s *captureEvalProcessSandbox) Start(_ context.Context, req agentcore.SandboxExec) (agentcore.SandboxProcess, error) {
	s.last = req
	return nil, s.startErr
}

func evalContext(id string) context.Context {
	return agentcore.WithSandboxSession(context.Background(), id)
}

func TestEvalToolPersistsStateResetsAndIsolatesSessions(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctxA := evalContext("conversation-a")
	ctxB := evalContext("conversation-b")

	if out, err := tool.Run(ctxA, `{"language":"python","code":"value = 40\nprint('ready')"}`); err != nil || !strings.Contains(out, "ready") {
		t.Fatalf("seed: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctxA, `{"language":"python","code":"value += 2\nvalue"}`); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("reuse: out=%q err=%v", out, err)
	}
	if _, err := tool.Run(ctxB, `{"language":"python","code":"value"}`); err == nil || !strings.Contains(err.Error(), "NameError") {
		t.Fatalf("isolated session error = %v", err)
	}
	if out, err := tool.Run(ctxA, `{"language":"python","reset":true,"code":"'value' in globals()"}`); err != nil || !strings.Contains(out, "False") {
		t.Fatalf("reset: out=%q err=%v", out, err)
	}
}

func TestEvalToolKeepsPartialStateAfterRuntimeError(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("partial-state")
	_, err := tool.Run(ctx, `{"language":"python","code":"before_error = 7\nraise RuntimeError('boom')"}`)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("runtime error = %v", err)
	}
	out, err := tool.Run(ctx, `{"language":"python","code":"before_error"}`)
	if err != nil || !strings.Contains(out, "7") {
		t.Fatalf("partial state: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalPersistsStateResetsAndIsolatesSessions(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctxA := evalContext("javascript-a")
	ctxB := evalContext("javascript-b")

	if out, err := tool.Run(ctxA, `{"language":"javascript","code":"let value = 40; console.log('ready')"}`); err != nil || !strings.Contains(out, "ready") {
		t.Fatalf("seed: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctxA, `{"language":"javascript","code":"value += 2; value"}`); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("reuse: out=%q err=%v", out, err)
	}
	if _, err := tool.Run(ctxB, `{"language":"javascript","code":"value"}`); err == nil || !strings.Contains(err.Error(), "ReferenceError") {
		t.Fatalf("isolated session error = %v", err)
	}
	if out, err := tool.Run(ctxA, `{"language":"javascript","reset":true,"code":"typeof value"}`); err != nil || !strings.Contains(out, "undefined") {
		t.Fatalf("reset: out=%q err=%v", out, err)
	}
}

func TestEvalLanguageStateAndResetAreIndependent(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is not available")
	}
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("language-isolation")
	if _, err := tool.Run(ctx, `{"language":"python","code":"language_value = 17"}`); err != nil {
		t.Fatalf("seed Python: %v", err)
	}
	if _, err := tool.Run(ctx, `{"language":"javascript","code":"globalThis.languageValue = 23"}`); err != nil {
		t.Fatalf("seed JavaScript: %v", err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","reset":true,"code":"typeof languageValue"}`); err != nil || !strings.Contains(out, "undefined") {
		t.Fatalf("reset JavaScript: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctx, `{"language":"python","code":"language_value"}`); err != nil || !strings.Contains(out, "17") {
		t.Fatalf("JavaScript reset touched Python: out=%q err=%v", out, err)
	}
}

func TestEvalSchemaAdvertisesBothPersistentLanguages(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	properties, ok := tool.Schema().Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema properties = %#v", tool.Schema().Parameters["properties"])
	}
	language, ok := properties["language"].(map[string]any)
	if !ok {
		t.Fatalf("language schema = %#v", properties["language"])
	}
	values, ok := language["enum"].([]string)
	if !ok || strings.Join(values, ",") != "python,javascript" {
		t.Fatalf("language enum = %#v", language["enum"])
	}
}

func TestJavaScriptEvalServerLaunchUsesDedicatedProtocolFDAndConfiguredImage(t *testing.T) {
	startErr := errors.New("stop after capture")
	sb := &captureEvalProcessSandbox{startErr: startErr}
	ws := mustWorkspace(t)
	registry := NewEvalSessionRegistry(1, time.Minute)
	t.Cleanup(registry.Close)
	tool, err := NewEvalTool(sb, ws, registry, "namespace", EvalConfig{
		JavaScript: EvalJavaScriptConfig{
			Command: "/usr/local/bin/node", Args: []string{"--no-deprecation"}, Image: "agentray-eval:test",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	spec, _ := tool.runtimeFor("javascript")
	if _, err := tool.startProcess(spec); !errors.Is(err, startErr) {
		t.Fatalf("start error = %v", err)
	}
	joined := strings.Join(sb.last.Argv, " ")
	for _, want := range []string{
		`sh -c exec "$@" 3>&1 1>/dev/null 2>/dev/null`,
		"/usr/local/bin/node --no-deprecation --eval",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("argv missing %q: %s", want, joined)
		}
	}
	if sb.last.Env["AGENTRAY_EVAL_PROTOCOL_FD"] != "3" || sb.last.Image != "agentray-eval:test" {
		t.Fatalf("server runtime request = %+v", sb.last)
	}
}

func TestEvalDockerImageRunsPythonAndJavaScriptKernels(t *testing.T) {
	image := strings.TrimSpace(os.Getenv("AGENTRAY_TEST_EVAL_IMAGE"))
	if image == "" {
		image = "agentray-eval:test"
	}
	checkCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if exec.CommandContext(checkCtx, "docker", "info").Run() != nil {
		t.Skip("Docker is unavailable")
	}
	if exec.CommandContext(checkCtx, "docker", "image", "inspect", image).Run() != nil {
		t.Skipf("eval image %q is unavailable; build it with make sandbox-build-eval EVAL_IMAGE=%s", image, image)
	}

	ws := mustWorkspace(t)
	module := `
export enum DockerScale { Answer = 42 }
export class DockerBox { constructor(public value: number) {} }
export const dockerModuleValue: number = new DockerBox(DockerScale.Answer).value
`
	if err := os.WriteFile(filepath.Join(ws.Root(), "docker-module.ts"), []byte(module), 0o600); err != nil {
		t.Fatal(err)
	}
	registry := NewEvalSessionRegistry(4, time.Minute)
	t.Cleanup(registry.Close)
	tool, err := NewEvalTool(NewDockerSandbox(), ws, registry, "docker-eval", EvalConfig{
		Python:         EvalPythonConfig{Image: image},
		JavaScript:     EvalJavaScriptConfig{Image: image},
		TimeoutSeconds: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := evalContext("docker-conversation")
	if out, err := tool.Run(ctx, `{"language":"python","code":"python_value = 6 * 7\npython_value"}`); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("Docker Python: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","code":"const javascriptValue = await Promise.resolve(42); javascriptValue"}`); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("Docker JavaScript: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","code":"javascriptValue + 1"}`); err != nil || !strings.Contains(out, "43") {
		t.Fatalf("Docker JavaScript persistence: out=%q err=%v", out, err)
	}
	code := `interface Label { value: string }
import { basename } from "node:path"
import { dockerModuleValue } from "./docker-module.ts"
const dockerLabel: Label = { value: basename("/tmp/item.txt") };
({label: dockerLabel.value, moduleValue: dockerModuleValue})`
	payload, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
	if out, err := tool.Run(ctx, string(payload)); err != nil || !strings.Contains(out, "item.txt") || !strings.Contains(out, `"moduleValue": 42`) {
		t.Fatalf("Docker JavaScript TypeScript/static import: out=%q err=%v", out, err)
	}
	probe := &evalBridgeProbeTool{}
	bridgeArgs, _ := json.Marshal(map[string]any{
		"language": "javascript", "code": `await tool.bridge_probe({value: "docker"})`,
	})
	agent, err := agentcore.New(agentcore.Config{
		Provider: agentcore.NewFauxProvider(
			agentcore.AssistantToolCall("docker-eval-call", ToolEval, string(bridgeArgs)),
			agentcore.AssistantText("done"),
		),
		Model: "faux", Tools: agentcore.NewToolSet(tool, probe),
		Policy: agentcore.NewAllowList(ToolEval, probeToolName),
	})
	if err != nil {
		t.Fatalf("Docker bridge agent: %v", err)
	}
	bridgeResult, err := agent.Prompt(evalContext("docker-bridge"), "bridge from the container")
	if err != nil {
		t.Fatalf("Docker bridge prompt: %v", err)
	}
	if probe.calls.Load() != 1 || !strings.Contains(toolMessageFor(t, bridgeResult, ToolEval).Content, "probe:docker") {
		t.Fatalf("Docker bridge failed: calls=%d result=%+v", probe.calls.Load(), bridgeResult)
	}
}

func TestJavaScriptEvalKeepsPartialStateAfterRuntimeError(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-partial")
	_, err := tool.Run(ctx, `{"language":"javascript","code":"let beforeError = 7; throw new Error('boom')"}`)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("runtime error = %v", err)
	}
	out, err := tool.Run(ctx, `{"language":"javascript","code":"beforeError"}`)
	if err != nil || !strings.Contains(out, "7") {
		t.Fatalf("partial state: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalTopLevelAwaitRejectionReturnsWithoutStrandingKernel(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-await-rejection")
	_, err := tool.Run(ctx, `{"language":"javascript","code":"await Promise.reject(new Error('async boom'))"}`)
	if err == nil || !strings.Contains(err.Error(), "async boom") {
		t.Fatalf("await rejection = %v", err)
	}
	out, err := tool.Run(ctx, `{"language":"javascript","code":"6 * 7"}`)
	if err != nil || !strings.Contains(out, "42") {
		t.Fatalf("kernel stranded after await rejection: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalFloatingRejectionIsCellErrorNotKernelCrash(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-floating-rejection")
	_, err := tool.Run(ctx, `{"language":"javascript","code":"Promise.reject(new Error('floating boom')); 1"}`)
	if err == nil || !strings.Contains(err.Error(), "floating boom") {
		t.Fatalf("floating rejection = %v", err)
	}
	out, err := tool.Run(ctx, `{"language":"javascript","code":"6 * 7"}`)
	if err != nil || !strings.Contains(out, "42") {
		t.Fatalf("kernel crashed after floating rejection: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalBoundsLargeFinalDisplayBeforeProtocolFrame(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{MaxOutputBytes: 4096})
	out, err := tool.Run(evalContext("javascript-large-display"), `{"language":"javascript","code":"'X'.repeat(2000000)"}`)
	if err != nil {
		t.Fatalf("large display: %v", err)
	}
	if len(out) > 4300 || !strings.Contains(out, "bytes omitted") || !strings.Contains(out, "XXXX") {
		t.Fatalf("bounded output len=%d output=%q", len(out), out)
	}
}

func TestJavaScriptEvalTopLevelAwaitJSONAndRichImage(t *testing.T) {
	tool, _, _ := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-rich")
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	code := `const values = await Promise.resolve([1, 2, 10]);
display({kind: "stats", mean: values.reduce((a, b) => a + b, 0) / values.length});
display({type: "image", mimeType: "image/png", data: Buffer.from("` + png + `", "base64")});
values.reduce((a, b) => a + b, 0)`
	payload, err := json.Marshal(map[string]any{"language": "javascript", "code": code})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.RunRich(ctx, string(payload))
	if err != nil {
		t.Fatalf("RunRich: %v", err)
	}
	if !strings.Contains(out.Content, `"kind": "stats"`) || !strings.Contains(out.Content, "13") {
		t.Fatalf("rich text = %q", out.Content)
	}
	if len(out.Parts) != 1 || out.Parts[0].MIMEType != "image/png" || out.Parts[0].Data != png {
		t.Fatalf("rich parts = %+v", out.Parts)
	}
}

func TestJavaScriptEvalStaticImportsTypeScriptAndPersistentBindings(t *testing.T) {
	requireNodeEvalTypeScript(t, false)
	tool, _, ws := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-typescript-imports")
	module := `
export default function twice(value) { return value * 2 }
export const answer = 21
`
	if err := os.WriteFile(filepath.Join(ws.Root(), "math.mjs"), []byte(module), 0o600); err != nil {
		t.Fatal(err)
	}
	code := `
interface Pair { left: number; right: number }
import twice, {
  answer as base
} from "./math.mjs"
const pair: Pair = { left: twice(base), right: 0 }
pair.left
`
	payload, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
	if out, err := tool.Run(ctx, string(payload)); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("TypeScript static import: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","code":"pair.left + 1"}`); err != nil || !strings.Contains(out, "43") {
		t.Fatalf("transformed binding did not persist: out=%q err=%v", out, err)
	}
	advanced := `
enum Direction { Left = 4, Right = 8 }
class Box { constructor(public value: number) {} }
const typed = ("ready" as string);
({value: new Box(Direction.Right).value, typed})
`
	payload, _ = json.Marshal(map[string]any{"language": "javascript", "code": advanced})
	out, err := tool.Run(ctx, string(payload))
	if err != nil || !strings.Contains(out, `"value": 8`) || !strings.Contains(out, `"typed": "ready"`) {
		t.Fatalf("advanced TypeScript: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalTransformsImportedTypeScriptModules(t *testing.T) {
	requireNodeEvalTypeScript(t, true)
	tool, _, ws := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-typescript-modules")
	dependency := `
export enum Scale { Double = 2 }
export class Box { constructor(public value: number) {} }
`
	module := `
import { Box, Scale } from "./typed-dependency.ts"
export const computed: number = new Box(21 * Scale.Double).value
`
	if err := os.WriteFile(filepath.Join(ws.Root(), "typed-dependency.ts"), []byte(dependency), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws.Root(), "typed-module.mts"), []byte(module), 0o600); err != nil {
		t.Fatal(err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","code":"import { computed } from './typed-module.mts'; computed"}`); err != nil || !strings.Contains(out, "42") {
		t.Fatalf("imported TypeScript module: out=%q err=%v", out, err)
	}
	if out, err := tool.Run(ctx, `{"language":"javascript","code":"computed + 1"}`); err != nil || !strings.Contains(out, "43") {
		t.Fatalf("imported binding did not persist: out=%q err=%v", out, err)
	}
}

func TestJavaScriptEvalSupportsStaticImportFormsAndLeavesImportTextUntouched(t *testing.T) {
	tool, _, ws := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-import-forms")
	module := `
import { writeFileSync } from "node:fs"
writeFileSync("static-import-side-effect.txt", "loaded")
export default function twice(value) { return value * 2 }
export const answer = 21
`
	if err := os.WriteFile(filepath.Join(ws.Root(), "forms.mjs"), []byte(module), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws.Root(), "data.json"), []byte(`{"answer":42}`), 0o600); err != nil {
		t.Fatal(err)
	}
	code := "import './forms.mjs';\n" +
		"import twice, * as math from './forms.mjs';\n" +
		"import data from './data.json' with { type: 'json' };\n" +
		"// import commentGhost from 'missing';\n" +
		"/* import blockGhost from 'missing'; */\n" +
		"const dynamicPath = await import('node:path');\n" +
		"const stringValue = \"import missing from 'nowhere'\";\n" +
		"const templateValue = `import missing from 'nowhere'`;\n" +
		"const regexValue = /import missing from ['\\\"]nowhere['\\\"]/;\n" +
		"if (true) /import ghost from ['\\\"]missing['\\\"]/.test(stringValue);\n" +
		"({value: twice(math.answer), jsonValue: data.answer, dynamicValue: dynamicPath.basename('/tmp/dynamic.txt'), textSafe: regexValue.test(stringValue) && templateValue.length > 0})"
	payload, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
	out, err := tool.Run(ctx, string(payload))
	if err != nil || !strings.Contains(out, `"value": 42`) || !strings.Contains(out, `"jsonValue": 42`) || !strings.Contains(out, `"dynamicValue": "dynamic.txt"`) || !strings.Contains(out, `"textSafe": true`) {
		t.Fatalf("static import forms: out=%q err=%v", out, err)
	}
	if data, err := os.ReadFile(filepath.Join(ws.Root(), "static-import-side-effect.txt")); err != nil || string(data) != "loaded" {
		t.Fatalf("side-effect import: data=%q err=%v", data, err)
	}
}

func TestJavaScriptEvalWorkspaceEnvironmentAndProtocolIsolation(t *testing.T) {
	const secretKey = "AGENTRAY_JS_EVAL_TEST_SECRET"
	t.Setenv(secretKey, "must-not-leak")
	tool, _, ws := testJavaScriptEvalTool(t, EvalConfig{})
	ctx := evalContext("javascript-boundary")
	code := `const fs = require("node:fs");
fs.writeFileSync("made-js.txt", "ok");
require("node:child_process").spawnSync(process.execPath, ["-e", "console.log('{\\\"type\\\":\\\"done\\\",\\\"id\\\":\\\"cell-1\\\",\\\"execution_count\\\":999}')"], {stdio: "inherit"});
console.log("real-output");
display({cwd: process.cwd(), secret: process.env.AGENTRAY_JS_EVAL_TEST_SECRET ?? null});
21 * 2`
	payload, _ := json.Marshal(map[string]any{"language": "javascript", "code": code})
	out, err := tool.Run(ctx, string(payload))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "must-not-leak") || strings.Contains(out, "execution_count") ||
		!strings.Contains(out, "real-output") || !strings.Contains(out, `"secret": null`) || !strings.Contains(out, "42") {
		t.Fatalf("boundary output = %q", out)
	}
	data, err := os.ReadFile(filepath.Join(ws.Root(), "made-js.txt"))
	if err != nil || string(data) != "ok" {
		t.Fatalf("workspace output: data=%q err=%v", data, err)
	}
}

func TestJavaScriptEvalTimeoutDiscardsKernelWithoutReplay(t *testing.T) {
	tool, _, ws := testJavaScriptEvalTool(t, EvalConfig{TimeoutSeconds: 2})
	ctx := evalContext("javascript-timeout")
	_, err := tool.Run(ctx, `{"language":"javascript","timeout_seconds":1,"code":"require('node:fs').appendFileSync('effects-js.txt', 'A'); globalThis.marker = 1; while (true) {}"}`)
	if err == nil || !strings.Contains(err.Error(), "not replayed") {
		t.Fatalf("timeout error = %v", err)
	}
	data, readErr := os.ReadFile(filepath.Join(ws.Root(), "effects-js.txt"))
	if readErr != nil || string(data) != "A" {
		t.Fatalf("cell was lost or replayed: data=%q err=%v", data, readErr)
	}
	out, err := tool.Run(ctx, `{"language":"javascript","code":"typeof marker"}`)
	if err != nil || !strings.Contains(out, "undefined") {
		t.Fatalf("fresh kernel: out=%q err=%v", out, err)
	}
}

func TestEvalToolWorkspaceAndEnvironmentBoundary(t *testing.T) {
	const secretKey = "AGENTRAY_EVAL_TEST_SECRET"
	t.Setenv(secretKey, "must-not-leak")
	tool, _, ws := testEvalTool(t, EvalConfig{})
	ctx := evalContext("environment")
	out, err := tool.Run(ctx, `{"language":"python","code":"import os\nopen('made.txt','w').write('ok')\ndisplay({'cwd': os.getcwd(), 'secret': os.getenv('AGENTRAY_EVAL_TEST_SECRET')})"}`)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "must-not-leak") || !strings.Contains(out, `"secret": null`) {
		t.Fatalf("environment leaked or missing boundary evidence: %q", out)
	}
	data, err := os.ReadFile(filepath.Join(ws.Root(), "made.txt"))
	if err != nil || string(data) != "ok" {
		t.Fatalf("workspace output: data=%q err=%v", data, err)
	}
}

func TestEvalToolReturnsRichMIMEDisplays(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("rich-display")
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	code := `import base64
class Chart:
    def _repr_markdown_(self): return "**chart ready**"
    def _repr_png_(self): return base64.b64decode("` + png + `")
display(Chart())`
	payload, err := json.Marshal(map[string]any{"language": "python", "code": code})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.RunRich(ctx, string(payload))
	if err != nil {
		t.Fatalf("RunRich: %v", err)
	}
	if !strings.Contains(out.Content, "**chart ready**") || !strings.Contains(out.Content, "rich image display") {
		t.Fatalf("rich display text = %q", out.Content)
	}
	if len(out.Parts) != 1 || out.Parts[0].Type != agentcore.ContentPartImage ||
		out.Parts[0].MIMEType != "image/png" || out.Parts[0].Data != png {
		t.Fatalf("rich display parts = %+v", out.Parts)
	}
}

func TestEvalToolNormalizesBinaryMIMEBundleImages(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("mime-bundle")
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	code := `import base64
class Bundle:
    def _repr_mimebundle_(self):
        return {"text/markdown": "bundle", "image/png": base64.b64decode("` + png + `")}
display(Bundle())`
	payload, err := json.Marshal(map[string]any{"language": "python", "code": code})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.RunRich(ctx, string(payload))
	if err != nil {
		t.Fatalf("RunRich: %v", err)
	}
	if len(out.Parts) != 1 || out.Parts[0].Data != png || !strings.Contains(out.Content, "bundle") {
		t.Fatalf("binary MIME bundle = %+v", out)
	}
}

func TestRenderEvalDisplayReportsImageLimitInsteadOfSilentlyDropping(t *testing.T) {
	bundle := map[string]json.RawMessage{
		"image/png": json.RawMessage(`"iVBORw0KGgo="`),
	}
	text, parts, used := renderEvalDisplay(bundle, 0, maxEvalImageBytes)
	if len(parts) != 0 || used != 0 || !strings.Contains(text, "per-cell image limit reached") {
		t.Fatalf("bounded display = text %q parts %+v used %d", text, parts, used)
	}
}

func TestEvalToolTimeoutDiscardsKernelWithoutReplay(t *testing.T) {
	tool, _, ws := testEvalTool(t, EvalConfig{TimeoutSeconds: 2})
	ctx := evalContext("timeout")
	_, err := tool.Run(ctx, `{"language":"python","timeout_seconds":1,"code":"import time\nopen('effects.txt','a').write('A')\ntime.sleep(10)"}`)
	if err == nil || !strings.Contains(err.Error(), "not replayed") {
		t.Fatalf("timeout error = %v", err)
	}
	data, readErr := os.ReadFile(filepath.Join(ws.Root(), "effects.txt"))
	if readErr != nil || string(data) != "A" {
		t.Fatalf("cell was lost or replayed: data=%q err=%v", data, readErr)
	}
	out, err := tool.Run(ctx, `{"language":"python","code":"'time' in globals()"}`)
	if err != nil || !strings.Contains(out, "False") {
		t.Fatalf("fresh kernel: out=%q err=%v", out, err)
	}
}

func TestEvalToolBoundsOutputAndSupportsTopLevelAwait(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{MaxOutputBytes: 4096})
	ctx := evalContext("output")
	out, err := tool.Run(ctx, `{"language":"python","code":"print('H' * 6000)\nprint('T' * 6000)"}`)
	if err != nil {
		t.Fatalf("large output: %v", err)
	}
	if len(out) > 4300 || !strings.Contains(out, "bytes omitted") || !strings.Contains(out, "HHHH") || !strings.Contains(out, "TTTT") {
		t.Fatalf("bounded output len=%d output=%q", len(out), out)
	}
	out, err = tool.Run(ctx, `{"language":"python","code":"import asyncio\nawait asyncio.sleep(0)\n6 * 7"}`)
	if err != nil || !strings.Contains(out, "42") {
		t.Fatalf("top-level await: out=%q err=%v", out, err)
	}
}

func TestEvalToolChildOutputCannotSpoofProtocol(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("protocol")
	code := `import subprocess, sys
subprocess.run([sys.executable, "-c", "print('{\\\"type\\\":\\\"done\\\",\\\"id\\\":\\\"cell-1\\\",\\\"execution_count\\\":999}')"])
print("real-output")
21 * 2`
	payload, err := json.Marshal(map[string]any{"language": "python", "code": code})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.Run(ctx, string(payload))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, `"execution_count":999`) || !strings.Contains(out, "real-output") || !strings.Contains(out, "42") {
		t.Fatalf("child output was not captured as data: %q", out)
	}
}

func TestEvalToolSerializesCellsWithinSession(t *testing.T) {
	tool, _, ws := testEvalTool(t, EvalConfig{})
	ctx := evalContext("serialized")
	firstDone := make(chan error, 1)
	go func() {
		_, err := tool.Run(ctx, `{"language":"python","code":"import time\nopen('started','w').write('yes')\nserial_value = 1\ntime.sleep(0.2)\nserial_value = 2"}`)
		firstDone <- err
	}()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(filepath.Join(ws.Root(), "started")); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("first cell did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}
	out, err := tool.Run(ctx, `{"language":"python","code":"serial_value"}`)
	if err != nil || !strings.Contains(out, "2") {
		t.Fatalf("serialized cell: out=%q err=%v", out, err)
	}
	if err := <-firstDone; err != nil {
		t.Fatalf("first cell: %v", err)
	}
}

func TestEvalToolRejectsInvalidArguments(t *testing.T) {
	tool, _, _ := testEvalTool(t, EvalConfig{})
	ctx := evalContext("invalid")
	for _, args := range []string{
		`{"language":"ruby","code":"1"}`,
		`{"language":"python","code":"1","timeout_seconds":999}`,
		`{"language":"python","code":"1","extra":true}`,
		`{"language":"python"}`,
		`{"language":"python","code":"1"} {}`,
	} {
		if _, err := tool.Run(ctx, args); err == nil {
			t.Errorf("Run(%s) unexpectedly succeeded", args)
		}
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := tool.Run(cancelled, `{"language":"python","code":"1"}`); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled error = %v", err)
	}
}
