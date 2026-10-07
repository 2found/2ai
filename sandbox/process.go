package sandbox

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"sync"

	"github.com/2found/2ai/agentcore"
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
		if p.runCtx.Err() != nil {
			res.Killed = true
			res.KillReason = p.timeout
		} else if runErr != nil {
			if ee, ok := runErr.(*exec.ExitError); ok {
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
