package agentcore

import "context"

// Decision is the outcome of a permission check. A blocked decision carries a
// human-readable reason that is returned to the model (not a silent failure) so
// it can adapt.
type Decision struct {
	Allow  bool
	Reason string
}

// Allowed is a convenience constructor for a permitted decision.
func Allowed() Decision { return Decision{Allow: true} }

// Blocked is a convenience constructor for a denied decision.
func Blocked(reason string) Decision { return Decision{Allow: false, Reason: reason} }

// Policy decides whether a tool call may execute. It is default-deny: a project
// starts with no tools permitted until the user opts in. The consumer supplies
// the implementation (e.g. scope -> tool mapping for the Growth Analyst).
type Policy interface {
	// Allow is consulted in the beforeToolCall preflight, after arguments are
	// schema-validated and before execution.
	Allow(ctx context.Context, call ToolCall) Decision
	// PermittedTools filters the advertised tool names down to those the policy
	// currently allows, so the model never sees a tool it cannot use.
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
