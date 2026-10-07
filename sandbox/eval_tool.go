package sandbox

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/2found/2ai/agentcore"
)

var evalToolSequence atomic.Uint64

// EvalTool executes one Python or JavaScript cell in a retained,
// conversation-and-language-scoped subprocess. The registry—not this per-run
// tool instance—owns the process, so state survives runtime rebuilding while
// Python and JavaScript can never share globals accidentally.
type EvalTool struct {
	processSB   agentcore.ProcessSandbox
	workspace   *Workspace
	registry    *EvalSessionRegistry
	config      EvalConfig
	hosted      bool
	namespace   string
	instance    string
	fingerprint string
}

func NewEvalTool(sb agentcore.Sandbox, workspace *Workspace, registry *EvalSessionRegistry, namespace string, config EvalConfig) (*EvalTool, error) {
	if workspace == nil {
		return nil, fmt.Errorf("eval requires an agent workspace")
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	hosted := sb == nil
	if _, ok := sb.(*HostSandbox); ok {
		hosted = true
	}
	if sb == nil {
		sb = NewHostSandbox()
	}
	processSB, ok := sb.(agentcore.ProcessSandbox)
	if !ok {
		return nil, fmt.Errorf("eval requires a sandbox backend with interactive process support")
	}
	if registry == nil {
		registry = NewEvalSessionRegistry(0, 0)
	}
	encoded, _ := json.Marshal(config)
	sum := sha256.Sum256(encoded)
	return &EvalTool{
		processSB: processSB, workspace: workspace, registry: registry, config: config,
		hosted: hosted, namespace: strings.TrimSpace(namespace),
		instance:    fmt.Sprintf("eval-%d", evalToolSequence.Add(1)),
		fingerprint: hex.EncodeToString(sum[:8]),
	}, nil
}

func (t *EvalTool) Name() string { return ToolEval }

func (t *EvalTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolEval,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Execute one Python or JavaScript cell in a persistent conversation-scoped runtime. Variables, imports, functions, and objects survive later calls in the same language. " +
			"Use reset=true to discard only the selected language's state before the cell. Relative file access starts in the shared agent workspace. " +
			"JavaScript accepts static or dynamic imports and TypeScript cell syntax when the runtime is Node.js 22.13 or newer. " +
			"display(value), rich displays, and final expressions preserve Markdown, JSON, PNG, and JPEG output; vision-capable models receive images natively. " +
			"Inside a live agent run, call governed host tools with await tool.read_file({...}) or await tool(\"read_file\", {...}) in JavaScript, and tool(\"read_file\", {...}) in Python. These calls use the same policy, validation, credentials, budgets, tracing, and cancellation as direct tool calls. " +
			"Output is bounded; interactive input is unsupported. A timeout discards the kernel and never replays the cell. " +
			"The runtime is operator-provisioned and server deployments execute it inside the configured sandbox.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"language":        map[string]any{"type": "string", "enum": []string{"python", "javascript"}, "description": "Language runtime. State is isolated per language."},
				"code":            map[string]any{"type": "string", "description": "One cell. Top-level await and display(value) are supported. JavaScript also accepts static imports and TypeScript syntax; TypeScript is transformed without type checking."},
				"timeout_seconds": map[string]any{"type": "integer", "minimum": 1, "maximum": t.config.TimeoutSeconds, "description": "Optional cell timeout; defaults to the configured maximum."},
				"reset":           map[string]any{"type": "boolean", "description": "Discard the selected language's retained state before executing this cell."},
			},
			"required": []string{"language", "code"},
		},
	}
}

func (t *EvalTool) Run(ctx context.Context, args string) (string, error) {
	out, err := t.run(ctx, args)
	return out.Content, err
}

// RunRich preserves MIME displays for the agent loop. Run remains the
// compatibility path for direct callers and returns the identical text.
func (t *EvalTool) RunRich(ctx context.Context, args string) (agentcore.ToolOutput, error) {
	return t.run(ctx, args)
}

func (t *EvalTool) run(ctx context.Context, args string) (agentcore.ToolOutput, error) {
	var in struct {
		Language       string  `json:"language"`
		Code           *string `json:"code"`
		TimeoutSeconds int     `json:"timeout_seconds"`
		Reset          bool    `json:"reset"`
	}
	dec := json.NewDecoder(strings.NewReader(args))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: invalid arguments: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: invalid arguments: expected one JSON object")
	}
	runtimeSpec, ok := t.runtimeFor(in.Language)
	if !ok {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: unsupported language %q", in.Language)
	}
	if in.Code == nil {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: code is required")
	}
	timeout := in.TimeoutSeconds
	if timeout == 0 {
		timeout = t.config.TimeoutSeconds
	}
	if timeout < 1 || timeout > t.config.TimeoutSeconds {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: timeout_seconds must be between 1 and %d", t.config.TimeoutSeconds)
	}
	if err := ctx.Err(); err != nil {
		return agentcore.ToolOutput{}, err
	}

	key := t.sessionKey(ctx, runtimeSpec.name)
	kernel, err := t.registry.acquire(key, in.Reset, func() (*evalProcess, error) {
		return t.startProcess(runtimeSpec)
	})
	if err != nil {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: start %s kernel: %w", runtimeSpec.label, err)
	}
	released := false
	release := func() {
		if !released {
			t.registry.release(key, kernel)
			released = true
		}
	}
	defer release()

	cellCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	invoker, _ := agentcore.ToolInvokerFrom(cellCtx)
	result, err := kernel.execute(cellCtx, *in.Code, t.config.MaxOutputBytes, t.config.MaxBridgeCalls, invoker)
	cancel()
	control := agentcore.ToolOutput{
		Invocations:        result.Invocations,
		AdditionalContexts: result.Extra,
		Terminate:          result.Terminate,
	}
	if err != nil {
		// The cell may already have performed side effects. Kill and forget the
		// process, but never replay the code in a fresh kernel.
		t.registry.invalidate(key, kernel)
		released = true
		if contextDeadline(cellCtx, err) {
			return control, fmt.Errorf("eval: cell timed out after %ds; kernel discarded and cell not replayed", timeout)
		}
		return control, fmt.Errorf("eval: kernel discarded and cell not replayed: %w", err)
	}
	output := strings.TrimSpace(result.Output)
	if output == "" {
		output = "(no output)"
	}
	if result.RuntimeError {
		control.Content = output
		control.Parts = result.Parts
		return control, fmt.Errorf("eval: %s cell %d failed; retained state before the error may remain:\n%s", runtimeSpec.label, result.ExecutionCount, output)
	}
	control.Content = output
	control.Parts = result.Parts
	return control, nil
}

func contextDeadline(ctx context.Context, err error) bool {
	return ctx.Err() == context.DeadlineExceeded || err == context.DeadlineExceeded
}

func (t *EvalTool) sessionKey(ctx context.Context, language string) string {
	session := strings.TrimSpace(agentcore.SandboxSessionFrom(ctx))
	if session == "" {
		session = t.instance
	}
	namespace := t.namespace
	if namespace == "" {
		namespace = t.instance
	}
	return strings.Join([]string{namespace, session, t.workspace.Root(), language, t.fingerprint}, "\x00")
}

type evalRuntimeSpec struct {
	name, label, command, image, runner string
	args                                []string
}

func (t *EvalTool) runtimeFor(language string) (evalRuntimeSpec, bool) {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "python", "py":
		return evalRuntimeSpec{
			name: "python", label: "Python", command: t.config.Python.Command,
			args: t.config.Python.Args, image: t.config.Python.Image, runner: pythonEvalRunner,
		}, true
	case "javascript", "js":
		return evalRuntimeSpec{
			name: "javascript", label: "JavaScript", command: t.config.JavaScript.Command,
			args: t.config.JavaScript.Args, image: t.config.JavaScript.Image, runner: javascriptEvalRunner,
		}, true
	default:
		return evalRuntimeSpec{}, false
	}
}

func (t *EvalTool) startProcess(spec evalRuntimeSpec) (*evalProcess, error) {
	env := map[string]string{
		"HOME": sandboxWorkdir, "TMPDIR": sandboxWorkdir,
		"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
		"NODE_NO_WARNINGS": "1",
	}
	cleanup := func() {}
	if t.hosted {
		home, err := os.MkdirTemp("", "agentray-eval-home-")
		if err != nil {
			return nil, err
		}
		cleanup = func() { _ = os.RemoveAll(home) }
		env["HOME"], env["TMPDIR"] = home, home
		for _, key := range []string{"PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "NODE_PATH"} {
			if value := os.Getenv(key); value != "" {
				env[key] = value
			}
		}
	}
	argv := append([]string{spec.command}, spec.args...)
	if spec.name == "python" {
		argv = append(argv, "-u", "-c", spec.runner)
	} else if runtime.GOOS == "windows" && t.hosted {
		// A trusted Windows laptop has no POSIX fd-redirection shell. The runner
		// still captures console/process writes on stdout; hosted Linux containers
		// and POSIX laptops use the stronger dedicated fd 3 path below.
		env["AGENTRAY_EVAL_PROTOCOL_FD"] = "1"
		argv = append(argv, "--eval", spec.runner)
	} else {
		env["AGENTRAY_EVAL_PROTOCOL_FD"] = "3"
		nodeArgv := append([]string(nil), argv...)
		nodeArgv = append(nodeArgv, "--eval", spec.runner)
		argv = append([]string{"sh", "-c", `exec "$@" 3>&1 1>/dev/null 2>/dev/null`, "agentray-js-eval"}, nodeArgv...)
	}
	proc, err := t.processSB.Start(context.Background(), agentcore.SandboxExec{
		Argv: argv, Env: env, Image: spec.image,
		Mounts:  []agentcore.SandboxMount{{Source: t.workspace.Root(), Target: shellWorkdir}},
		Workdir: shellWorkdir,
		Constraints: agentcore.SandboxLimits{
			RunAsHostUser: true,
			MemoryMB:      512, CPUs: 1, PidsLimit: 128, TimeoutSeconds: evalProcessLifetime.Seconds(),
		},
	})
	if err != nil {
		cleanup()
		return nil, err
	}
	return newEvalProcess(proc, cleanup), nil
}
