package sandbox

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/2found/2ai/core/agentcore"
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
		Description: "Execute one Python or JavaScript cell in a persistent conversation-scoped runtime. Variables, imports, functions, and objects survive later calls in the same language; reset=true discards that language's state first. Relative file access starts in the shared agent workspace. " +
			"JavaScript accepts static or dynamic imports and TypeScript syntax (Node.js 22.13+); top-level const, let and class can be declared again in a later cell. " +
			"display(value), rich displays, and final expressions preserve Markdown, JSON, PNG, and JPEG output; vision-capable models receive images natively. " +
			fmt.Sprintf("Inside a live agent run a cell calls governed host tools (same policy, validation, credentials, budgets, tracing and cancellation as direct calls): JavaScript await tool.read({...}) or await tool(\"read\", {...}); Python tool(\"read\", {...}). Up to %d calls per cell. ", t.config.MaxBridgeCalls) +
			"Each call returns the tool's text as a string (parse JSON yourself); a result over 1.5 MB is refused, so ask the tool for less. Pictures a tool shows come to you with the cell's result. " +
			"A failed call raises ToolError with .kind: blocked, failed, budget, approval, limit, invalid, unavailable or aborted. " +
			"For concurrency use Promise.all in JavaScript; in Python tool_all([(name, args), ...]) returns the results in order (a failed one is a ToolError value) and await atool(name, args) works with asyncio.gather. " +
			"A result that opens with [new ... session] ran in a fresh kernel (idle time, a timeout, a crash): set up again what earlier cells defined. " +
			fmt.Sprintf("Output is bounded (print a narrower slice or write to a file); no interactive input. timeout_seconds defaults to %d and may be up to %d: a timeout discards the kernel and never replays the cell, and returns the output produced so far.", t.config.TimeoutSeconds, t.config.MaxTimeoutSeconds),
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"language":        map[string]any{"type": "string", "enum": []string{"python", "javascript"}, "description": "Language runtime. State is isolated per language."},
				"code":            map[string]any{"type": "string", "description": "One cell. Top-level await and display(value) are supported. JavaScript also accepts static imports and TypeScript syntax; TypeScript is transformed without type checking."},
				"timeout_seconds": map[string]any{"type": "integer", "minimum": 1, "maximum": t.config.MaxTimeoutSeconds, "description": fmt.Sprintf("Optional cell timeout in seconds; defaults to %d.", t.config.TimeoutSeconds)},
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
	if timeout < 1 || timeout > t.config.MaxTimeoutSeconds {
		return agentcore.ToolOutput{}, fmt.Errorf("eval: timeout_seconds must be between 1 and %d", t.config.MaxTimeoutSeconds)
	}
	if err := ctx.Err(); err != nil {
		return agentcore.ToolOutput{}, err
	}

	key := t.sessionKey(ctx, runtimeSpec.name)
	reset := in.Reset
	var (
		kernel   *evalKernel
		result   evalCellResult
		err      error
		cellCtx  context.Context
		attempts int
	)
	for {
		attempts++
		kernel, err = t.registry.acquire(key, reset, func() (*evalProcess, error) {
			return t.startProcess(runtimeSpec)
		})
		if err != nil {
			return agentcore.ToolOutput{}, fmt.Errorf("eval: start %s kernel: %w", runtimeSpec.label, err)
		}
		var cancel context.CancelFunc
		cellCtx, cancel = context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
		invoker, _ := agentcore.ToolInvokerFrom(cellCtx)
		result, err = kernel.execute(cellCtx, *in.Code, t.config.MaxOutputBytes, t.config.MaxBridgeCalls, invoker)
		cancel()
		if errors.Is(err, errEvalNotSent) && attempts == 1 {
			// The kernel was gone before it took the cell, so the cell has not run:
			// give it a fresh kernel (which says it is new) instead of failing.
			t.registry.invalidate(key, kernel)
			reset = false
			continue
		}
		break
	}
	control := agentcore.ToolOutput{
		Invocations:        result.Invocations,
		AdditionalContexts: result.Extra,
		Terminate:          result.Terminate,
		External:           result.External,
	}
	if err != nil {
		// The cell may already have performed side effects. Kill and forget the
		// process, but never replay the code in a fresh kernel. What it produced
		// before the end is the model's only account of those effects.
		t.registry.invalidate(key, kernel)
		control.Parts = result.Parts
		partial := strings.TrimSpace(result.Output)
		withPartial := func(message string) error {
			if partial == "" {
				return errors.New(message)
			}
			return errors.New(message + "\nOutput before it ended:\n" + partial)
		}
		switch {
		case contextDeadline(cellCtx, err):
			return control, withPartial(fmt.Sprintf("eval: cell timed out after %ds; kernel discarded (its state is gone) and cell not replayed", timeout))
		case errors.Is(err, errEvalNotSent):
			return control, fmt.Errorf("eval: the kernel could not take the cell, which did not run: %w", err)
		default:
			return control, withPartial(fmt.Sprintf("eval: the kernel ended while the cell ran, so the cell may have partly run; kernel discarded (its state is gone) and cell not replayed: %v", err))
		}
	}
	t.registry.release(key, kernel)
	output := strings.TrimSpace(result.Output)
	if output == "" {
		output = "(no output)"
	}
	if result.Fresh && !in.Reset {
		// A kernel that replaces one lost to idle time, a timeout or a crash looks
		// like the old one until a name is missing; say so up front.
		output = fmt.Sprintf("[new %s session: nothing earlier cells defined exists]\n%s", runtimeSpec.label, output)
	}
	control.Content = output
	control.Parts = result.Parts
	if result.RuntimeError {
		return control, fmt.Errorf("eval: %s cell %d failed; retained state before the error may remain:\n%s", runtimeSpec.label, result.ExecutionCount, output)
	}
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
