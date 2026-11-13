package sandbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"

	"github.com/2found/2ai/core/agentcore"
)

// commandProcess adapts an os/exec command to agentcore.SandboxProcess. The
// caller consumes stdout/stderr directly, so Wait reports lifecycle state and
// exit status rather than buffered output.
type commandProcess struct {
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdout   io.ReadCloser
	stderr   io.ReadCloser
	runCtx   context.Context
	cancel   context.CancelFunc
	killFn   func() error
	cleanup  func()
	errLabel string
	timeout  string

	killOnce sync.Once
	waitOnce sync.Once
	waitRes  processWaitResult
}

type processWaitResult struct {
	result agentcore.SandboxResult
	err    error
}

// startCommand wires cmd's three pipes and starts it under runCtx. On failure
// cancel has been called; the caller owns any other cleanup.
func startCommand(cmd *exec.Cmd, runCtx context.Context, cancel context.CancelFunc, label string) (*commandProcess, error) {
	fail := func(what string, err error) (*commandProcess, error) {
		cancel()
		return nil, fmt.Errorf("sandbox: %s %s: %w", label, what, err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fail("stdin pipe", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fail("stdout pipe", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fail("stderr pipe", err)
	}
	if err := cmd.Start(); err != nil {
		return fail("start", err)
	}
	return &commandProcess{
		cmd: cmd, stdin: stdin, stdout: stdout, stderr: stderr,
		runCtx: runCtx, cancel: cancel, errLabel: label,
	}, nil
}

func (p *commandProcess) Stdin() io.WriteCloser { return p.stdin }
func (p *commandProcess) Stdout() io.ReadCloser { return p.stdout }
func (p *commandProcess) Stderr() io.ReadCloser { return p.stderr }

func (p *commandProcess) Kill() error {
	var err error
	p.killOnce.Do(func() {
		p.cancel()
		if p.killFn != nil {
			err = p.killFn()
		}
	})
	return err
}

func (p *commandProcess) Wait() (agentcore.SandboxResult, error) {
	p.waitOnce.Do(func() {
		runErr := p.cmd.Wait()
		res := agentcore.SandboxResult{}
		if err := p.runCtx.Err(); err != nil {
			res.Killed = true
			res.KillReason = "cancelled"
			if errors.Is(err, context.DeadlineExceeded) {
				res.KillReason = p.timeout
			}
		} else if runErr != nil {
			var ee *exec.ExitError
			if errors.As(runErr, &ee) {
				res.ExitCode = ee.ExitCode()
			} else {
				p.waitRes.err = fmt.Errorf("sandbox: %s: %w", p.errLabel, runErr)
			}
		}
		p.waitRes.result = res
		p.cancel()
		if p.cleanup != nil {
			p.cleanup()
		}
	})
	return p.waitRes.result, p.waitRes.err
}
