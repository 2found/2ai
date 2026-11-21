package jobs

import (
	"context"
	"fmt"
	"strings"

	"github.com/2found/2ai/core/agentcore"
)

// ToolName is the one tool the board gives the model. Its op picks what it
// does to the run's background jobs: list, status, wait and cancel, plus send
// (Plugin.AllowSend) and keep (Plugin.Survive). One tool in place of one per
// op keeps a rarely used capability to a single declaration on every request.
const ToolName = "job"

// jobsTool is the job tool: it reads op and hands the call to that op's
// handler, which reads the rest of the arguments.
type jobsTool struct{ jobTool }

// ops are the ops the board takes, in the order the schema lists them.
func (b *board) ops() []string {
	ops := []string{"list", "status", "wait", "cancel"}
	if b.allowSend {
		ops = append(ops, "send")
	}
	if b.survive != nil {
		ops = append(ops, "keep")
	}
	return ops
}

func (t jobsTool) Schema() agentcore.ToolSchema {
	b := t.board
	d := "Manage the background jobs this run started (a background shell command, an async sub-agent, a slow tool run async) by op: " + strings.Join(b.ops(), ", ") + ". " +
		"op list: every job, newest first, with its state, duration and, while it runs, its newest output line. " +
		"op status: one job; its full output once it finished, the newest part while it runs. " +
		"op wait: block until it finishes, at most timeout_seconds; a job still running then keeps going. " +
		"op cancel: stop it."
	if b.allowSend {
		d += " op send: give a running background sub-agent a message it reads at its next step, to correct course or add a constraint without cancelling it."
	}
	if s := b.survive; s != nil {
		d += fmt.Sprintf(" op keep: let a running job (a dev server, a long build) outlive this run, for at most %s, %d per conversation; its result is announced when it ends. "+
			"Sub-agents cannot be kept, and never keep work whose output this answer needs.", s.maxAge, s.maxJobs)
	}
	d += " Jobs still running when you answer are cancelled"
	if b.survive != nil {
		d += " unless kept"
	}
	d += "; wait for the ones whose results you need."
	props := map[string]any{
		"op":              map[string]any{"type": "string", "enum": b.ops()},
		"id":              map[string]any{"type": "string", "description": "The job id; every op but list needs it."},
		"output_bytes":    map[string]any{"type": "integer", "description": "status: " + statusBytesDescription(b.maxBytes)},
		"timeout_seconds": map[string]any{"type": "integer", "description": "wait: how long to wait before returning with the job still running."},
	}
	if b.allowSend {
		props["message"] = map[string]any{"type": "string", "description": "send: what to tell the sub-agent."}
	}
	return agentcore.ToolSchema{
		Name:        t.name,
		Description: d,
		Strict:      agentcore.ToolStrictEnabled,
		Parameters:  map[string]any{"type": "object", "properties": props, "required": []string{"op"}},
	}
}

func (t jobsTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Op string `json:"op"`
	}
	if err := unmarshalArgs(args, &in); err != nil {
		return "", err
	}
	op := strings.TrimSpace(in.Op)
	switch {
	case op == "list":
		return jobListTool{t.jobTool}.Run(ctx, args)
	case op == "status":
		return jobStatusTool{t.jobTool}.Run(ctx, args)
	case op == "wait":
		return jobWaitTool{t.jobTool}.Run(ctx, args)
	case op == "cancel":
		return jobCancelTool{t.jobTool}.Run(ctx, args)
	case op == "send" && t.board.allowSend:
		return jobSendTool{t.jobTool}.Run(ctx, args)
	case op == "keep" && t.board.survive != nil:
		return jobKeepTool{t.jobTool}.Run(ctx, args)
	case op == "":
		return "", fmt.Errorf("op is required: one of %s", strings.Join(t.board.ops(), ", "))
	}
	return "", fmt.Errorf("op %q is not one this run takes: one of %s", op, strings.Join(t.board.ops(), ", "))
}
