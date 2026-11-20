package agentcore

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// ToolInvocation is the audit/accounting projection of a tool call initiated
// from inside another tool. Eval kernels use it for host-tool bridges: the
// nested call does not become a second provider-authored tool message, but it
// still has to survive in the outer call's durable outcome and RunResult.
type ToolInvocation struct {
	Trace    ToolTrace `json:"trace"`
	Executed bool      `json:"executed,omitempty"`
}

// ToolInvoker is the run-owned capability for invoking another registered tool
// from inside a tool. Implementations must enter through Agent.runToolCall; a
// raw Tool.Run call would bypass validation, policy, hooks, credentials, result
// bounds, and idempotency. The capability exists only on contexts handed out by
// a live Agent run, so constructing a Tool directly does not grant delegation.
type ToolInvoker interface {
	InvokeTool(ctx context.Context, name, args string) (ToolOutput, error)
}

type toolInvokerContextKey struct{}
type toolStackContextKey struct{}
type toolBudgetContextKey struct{}
type toolBridgeGuardContextKey struct{}
type nestedToolEventsKey struct{}

// ToolInvokerFrom returns the run-owned nested invocation capability, when the
// current call is executing inside an Agent loop.
func ToolInvokerFrom(ctx context.Context) (ToolInvoker, bool) {
	if ctx == nil {
		return nil, false
	}
	invoker, ok := ctx.Value(toolInvokerContextKey{}).(ToolInvoker)
	return invoker, ok && invoker != nil
}

func withToolInvoker(ctx context.Context, invoker ToolInvoker) context.Context {
	return context.WithValue(ctx, toolInvokerContextKey{}, invoker)
}

func withToolStack(ctx context.Context, name string) context.Context {
	stack, _ := ctx.Value(toolStackContextKey{}).([]string)
	next := make([]string, len(stack)+1)
	copy(next, stack)
	next[len(stack)] = name
	return context.WithValue(ctx, toolStackContextKey{}, next)
}

// NestedToolCall reports a call made from inside another tool through the
// run's ToolInvoker (code in run_js or an eval cell), rather than by the
// model. Such a result returns to code: it is neither spilled nor cut to
// the model's result bound, nor framed as external content (the caller
// learns that from ToolOutput.External).
func NestedToolCall(ctx context.Context) bool {
	nested, _ := ctx.Value(nestedToolCallKey{}).(bool)
	return nested
}

// nestedToolCallKey marks the context of one call the ToolInvoker makes. It
// is cleared again on the context the tool itself runs with, so a run the
// tool starts (a sub-agent) makes direct calls of its own. The tool stack
// cannot tell: a sub-agent's calls inherit it.
type nestedToolCallKey struct{}

func withNestedToolCall(ctx context.Context, nested bool) context.Context {
	if NestedToolCall(ctx) == nested {
		return ctx
	}
	return context.WithValue(ctx, nestedToolCallKey{}, nested)
}

// maxNestedToolResultBytes bounds a nested result for code that set no
// bound of its own; run_js and eval refuse far less.
const maxNestedToolResultBytes = 64 << 20

func toolStack(ctx context.Context) []string {
	if ctx == nil {
		return nil
	}
	stack, _ := ctx.Value(toolStackContextKey{}).([]string)
	return stack
}

// toolExecutionBudget is shared by every direct and bridged call in one run.
// Direct and nested calls both reserve before execution, so parallel batches
// and eval cells cannot race past MaxToolCalls.
//
// Reading a skill for the first time is not charged. read_skill returns only
// this agent's own instructions and reaches nothing outside the run, yet an
// agent whose task matches two skills had to spend two calls of a small
// budget before doing any work: a support agent capped at seven calls lost
// the one it needed to file the customer's note. A second read of the same
// skill, or a read of a skill the agent does not have, is charged like any
// call, so the budget still bounds a loop.
type toolExecutionBudget struct {
	used atomic.Int64
	max  int64

	// skill names the enabled skill a read_skill call loads; nil when the run
	// has no skills.
	skill func(args string) (string, bool)
	mu    sync.Mutex
	read  map[string]bool
}

func newToolExecutionBudget(maximum int) *toolExecutionBudget {
	if maximum < 1 {
		maximum = 1
	}
	return &toolExecutionBudget{max: int64(maximum)}
}

func (b *toolExecutionBudget) exhausted() bool {
	return b == nil || b.used.Load() >= b.max
}

func (b *toolExecutionBudget) reserve() bool {
	if b == nil {
		return false
	}
	for {
		used := b.used.Load()
		if used >= b.max {
			return false
		}
		if b.used.CompareAndSwap(used, used+1) {
			return true
		}
	}
}

// reserveCall reserves one execution for a call, free for the first read of
// each skill; charged reports whether it counts. The returned func gives the
// reservation back when the call did not execute.
func (b *toolExecutionBudget) reserveCall(name, args string) (reserved, charged bool, release func()) {
	if b != nil && b.skill != nil && name == readSkillToolName {
		if id, ok := b.skill(args); ok {
			b.mu.Lock()
			first := !b.read[id]
			if first {
				if b.read == nil {
					b.read = map[string]bool{}
				}
				b.read[id] = true
			}
			b.mu.Unlock()
			if first {
				return true, false, func() {
					b.mu.Lock()
					delete(b.read, id)
					b.mu.Unlock()
				}
			}
		}
	}
	if !b.reserve() {
		return false, false, func() {}
	}
	return true, true, b.release
}

// left is how many more calls the run may charge.
func (b *toolExecutionBudget) left() int64 {
	if b == nil {
		return 0
	}
	return max(b.max-b.used.Load(), 0)
}

func (b *toolExecutionBudget) release() {
	if b != nil {
		b.used.Add(-1)
	}
}

func withToolExecutionBudget(ctx context.Context, budget *toolExecutionBudget) context.Context {
	return context.WithValue(ctx, toolBudgetContextKey{}, budget)
}

func toolExecutionBudgetFrom(ctx context.Context) *toolExecutionBudget {
	if ctx == nil {
		return nil
	}
	budget, _ := ctx.Value(toolBudgetContextKey{}).(*toolExecutionBudget)
	return budget
}

type toolBridgeGuard func(name string) (allow bool, reason string)

func withToolBridgeGuard(ctx context.Context, guard toolBridgeGuard) context.Context {
	return context.WithValue(ctx, toolBridgeGuardContextKey{}, guard)
}

func toolBridgeGuardFrom(ctx context.Context) toolBridgeGuard {
	if ctx == nil {
		return nil
	}
	guard, _ := ctx.Value(toolBridgeGuardContextKey{}).(toolBridgeGuard)
	return guard
}

const maxNestedToolDepth = 8

func validateNestedToolTarget(ctx context.Context, name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("nested tool name is required")
	}
	stack := toolStack(ctx)
	if len(stack) >= maxNestedToolDepth {
		return fmt.Errorf("nested tool depth exceeds %d", maxNestedToolDepth)
	}
	for _, active := range stack {
		if active == name {
			return fmt.Errorf("recursive nested tool call %q is not allowed", name)
		}
	}
	if guard := toolBridgeGuardFrom(ctx); guard != nil {
		if allow, reason := guard(name); !allow {
			if strings.TrimSpace(reason) == "" {
				reason = "tool is unavailable for nested invocation"
			}
			return fmt.Errorf("%s", reason)
		}
	}
	return nil
}
