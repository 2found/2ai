package agentcore

import "context"

// Decision is the outcome of a permission check. The zero value refuses the
// call, so a gate that forgets to decide fails closed. A refused or parked
// decision carries a human-readable reason that is returned to the model (not
// a silent failure) so it can adapt.
type Decision struct {
	// Allow lets the call run.
	Allow bool
	// Ask parks the run on a human approval instead of running or refusing the
	// call (Allow must be false). The run ends with a question built from the
	// tool, its arguments and Reason; the human's answer reaches the model as a
	// new user message, and the model re-issues the call if it still wants it.
	// Recording the approval is the host's job: its Policy answers Allow for
	// that call once the human has approved, which is why approval grants stay
	// application state rather than agentcore state.
	Ask bool
	// Reason explains a refusal or an approval request.
	Reason string
	// Arguments, when non-empty, replaces the call's JSON arguments for every
	// later gate and for execution (path normalization, clamping a limit,
	// stripping a field). The replacement is validated against the tool's
	// schema again and recorded on the trace.
	Arguments string
}

// Allowed is a convenience constructor for a permitted decision.
func Allowed() Decision { return Decision{Allow: true} }

// AllowedWith permits the call with replacement JSON arguments.
func AllowedWith(arguments string) Decision { return Decision{Allow: true, Arguments: arguments} }

// Blocked is a convenience constructor for a denied decision.
func Blocked(reason string) Decision { return Decision{Allow: false, Reason: reason} }

// AskApproval parks the call on a human approval (see Decision.Ask).
func AskApproval(reason string) Decision { return Decision{Ask: true, Reason: reason} }

// refused reports a decision that ends evaluation: neither allowed nor parked.
func (d Decision) refused() bool { return !d.Allow && !d.Ask }

// Policy decides whether a tool call may execute. It is default-deny: a project
// starts with no tools permitted until the user opts in. The consumer supplies
// the implementation (e.g. scope -> tool mapping for the Growth Analyst).
type Policy interface {
	// Allow is consulted after arguments are schema-validated and before
	// execution. call.Arguments holds the validated JSON, so a rule may depend
	// on what the call does, not only on which tool it names.
	Allow(ctx context.Context, call ToolCall) Decision
	// PermittedTools filters the advertised tool names down to those the policy
	// currently allows or would ask about, so the model never sees a tool it
	// cannot use.
	PermittedTools(ctx context.Context, all []string) []string
}

// DenyAll is the safe default Policy: it permits nothing. Useful as a base and
// in tests.
type DenyAll struct{}

func (DenyAll) Allow(context.Context, ToolCall) Decision          { return Blocked("no tools permitted") }
func (DenyAll) PermittedTools(context.Context, []string) []string { return nil }

// AllowList is a simple Policy permitting an explicit set of tool names. It is
// the building block consumers compose from scope definitions.
type AllowList struct {
	names map[string]bool
}

// NewAllowList builds an AllowList from the given permitted tool names.
func NewAllowList(names ...string) *AllowList {
	m := make(map[string]bool, len(names))
	for _, n := range names {
		m[n] = true
	}
	return &AllowList{names: m}
}

func (a *AllowList) Allow(_ context.Context, call ToolCall) Decision {
	if a.names[call.Name] {
		return Allowed()
	}
	return Blocked("tool '" + call.Name + "' is not permitted by the current permission scopes")
}

func (a *AllowList) PermittedTools(_ context.Context, all []string) []string {
	out := make([]string, 0, len(all))
	for _, n := range all {
		if a.names[n] {
			out = append(out, n)
		}
	}
	return out
}

// Delegation can only narrow authority.
//
// A run's effective policy is its own policy followed by the policy of every
// run that delegated to it. The chain rides the context, because a
// cross-agent delegate is a freshly built Agent whose fields know nothing of
// its caller; ctx is the only thread that survives the hop. Every policy in
// the chain must allow a call, so a delegate composed with a broader policy
// than its parent still cannot do anything the parent could not.

// policyChain is an intersection of policies: the strictest answer wins
// (refuse over ask over allow), and argument rewrites thread through in order.
type policyChain []Policy

type delegatingPoliciesKey struct{}

// withDelegatingPolicy records p as a policy every run started under ctx must
// also satisfy. The loop calls it for every tool execution, so whatever the
// tool starts (a forked child, a named delegate) inherits the caller's limits.
func withDelegatingPolicy(ctx context.Context, p Policy) context.Context {
	parents, _ := ctx.Value(delegatingPoliciesKey{}).(policyChain)
	chain := make(policyChain, 0, len(parents)+1)
	chain = append(append(chain, parents...), p)
	return context.WithValue(ctx, delegatingPoliciesKey{}, chain)
}

// effectivePolicy is this agent's policy intersected with every delegating
// ancestor's policy on ctx.
func (a *Agent) effectivePolicy(ctx context.Context) policyChain {
	parents, _ := ctx.Value(delegatingPoliciesKey{}).(policyChain)
	return append(policyChain{a.policy}, parents...)
}

// PermittedTools keeps only the names every policy in the chain permits.
func (c policyChain) PermittedTools(ctx context.Context, all []string) []string {
	for _, p := range c {
		all = p.PermittedTools(ctx, all)
	}
	return all
}

// narrowPolicy intersects parent with any number of narrowing policies.
// Without narrowing the parent's policy is kept as is.
func narrowPolicy(parent Policy, narrow []Policy) Policy {
	chain := policyChain{parent}
	for _, p := range narrow {
		if p != nil {
			chain = append(chain, p)
		}
	}
	if len(chain) == 1 {
		return parent
	}
	return chainPolicy{chain}
}

// chainPolicy exposes an intersection as a Policy for a narrowed fork.
type chainPolicy struct{ chain policyChain }

func (p chainPolicy) Allow(ctx context.Context, call ToolCall) Decision {
	d := Allowed()
	for _, policy := range p.chain {
		next := policy.Allow(ctx, call)
		d, call = mergeDecision(d, next, call)
		if d.refused() {
			return d
		}
	}
	return d
}

func (p chainPolicy) PermittedTools(ctx context.Context, all []string) []string {
	return p.chain.PermittedTools(ctx, all)
}

// mergeDecision folds the next gate's answer into the decision so far. A
// refusal ends evaluation, an approval request is sticky, and an argument
// rewrite applies to the call every later gate sees.
func mergeDecision(so Decision, next Decision, call ToolCall) (Decision, ToolCall) {
	if next.refused() {
		if next.Reason == "" {
			next.Reason = "tool call refused"
		}
		return next, call
	}
	if next.Arguments != "" {
		call.Arguments = next.Arguments
		so.Arguments = next.Arguments
	}
	if next.Ask && !so.Ask {
		so.Allow, so.Ask, so.Reason = false, true, next.Reason
	}
	return so, call
}
