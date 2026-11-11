package agentcore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strings"
)

// Env describes host-owned tool execution infrastructure. It is executor
// configuration, not an agent capability installed through plugins/extensions.
type Env struct {
	// Sandbox is the optional isolation substrate for tools that execute
	// untrusted code (shell, file, browser). nil when no such tools are wired —
	// the analytics tools never touch it. The concrete backend lives outside
	// this leaf package and is injected by the host. This field records the
	// backend for diagnostics; the host must also bind it to each tool that uses
	// it. Setting this field alone does not sandbox a tool.
	Sandbox Sandbox
	// Credentials is the optional secret resolver. When set, {{cred:NAME}}
	// placeholders in a call's arguments are resolved at the trust boundary —
	// after the call has been traced and gated, immediately before the tool
	// executes — for the tools CredentialGrants allows, and every resolved value
	// is redacted from what the tool returns. nil = no resolution; arguments
	// reach the tool unchanged.
	Credentials CredentialResolver
	// CredentialGrants say which tools (and, for network tools, which hosts)
	// may receive each credential. Default deny: a placeholder no grant covers
	// blocks the call.
	CredentialGrants []CredentialGrant
}

// CredentialResolver returns the secret value for one {{cred:NAME}}
// placeholder. The executor calls Resolve with exactly one placeholder and no
// other text, after the grant for that credential has been checked, so the
// literal secret exists only in the arguments handed to the executing tool —
// never in the model's view, the persisted trace, or the before-hooks.
//
// agentcore defines the contract only; the concrete vault is injected by the
// host via Env.Credentials, keeping this package a leaf with no
// infrastructure imports. Resolve MUST fail closed: an unknown placeholder
// returns an error, which blocks the call and feeds the reason back to the
// model rather than silently leaving the placeholder in place.
type CredentialResolver interface {
	Resolve(ctx context.Context, args string) (string, error)
}

// DefaultEnv returns an Env backed by real implementations. Both capabilities
// are host-injected and optional, so the default env is the empty one: a run
// with no sandbox and no vault is a complete, working agent that simply cannot
// execute untrusted code or resolve a secret.
func DefaultEnv() Env { return Env{} }

// sandboxSessionKey carries the persistent-session id down the context into a
// session-aware Sandbox backend. agentcore stays generic: it never learns what
// the id means; it only lets a session-capable tool (computer_use) key a
// long-lived execution environment so installed packages and written files
// survive across tool calls in one conversation — the property that turns a
// one-shot shell into a Claude-Code-level computer-use surface.
type sandboxSessionKey struct{}

// WithSandboxSession tags ctx with a stable session id a SessionSandbox uses to
// reuse one persistent container across tool calls. The consumer (the Runner)
// sets it to the conversation session id just before driving the loop; an empty
// id is a no-op and leaves tools on the ephemeral, throwaway-container path.
func WithSandboxSession(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, sandboxSessionKey{}, id)
}

// SandboxSessionFrom reads the session id set by WithSandboxSession ("" when
// absent — the safe default that keeps execution ephemeral and unshared).
func SandboxSessionFrom(ctx context.Context) string {
	if v, ok := ctx.Value(sandboxSessionKey{}).(string); ok {
		return v
	}
	return ""
}

// Sandbox executes untrusted commands in an isolated environment that cannot
// reach the host's environment, filesystem, or network unless explicitly
// granted. It is the substrate the agent's shell / file / browser tools run in,
// so a prompt-injected command cannot read server secrets, touch the DB, or
// exfiltrate over the network — the in-process Policy gate decides *whether* a
// tool may run; the Sandbox decides *what the host it runs against can see*.
//
// agentcore defines the contract only. A concrete backend (Docker container,
// gVisor/Kata, micro-VM, …) is injected by the host via Env.Sandbox, keeping
// this package a leaf with no infrastructure imports — the same boundary that
// keeps the core reusable across agents (see agentcore/README.md).
type Sandbox interface {
	// Exec runs one command to completion inside an ephemeral sandbox and
	// returns its captured output. It MUST NOT inherit the host process
	// environment; only req.Env is exposed inside. A non-zero exit code is a
	// SandboxResult, not an error — error is reserved for the sandbox itself
	// failing to run (backend unavailable, image missing).
	Exec(ctx context.Context, req SandboxExec) (SandboxResult, error)
}

// SessionSandbox is an optional capability a Sandbox backend may also implement
// to support persistent computer-use sessions. When req.Session is non-empty,
// Exec reuses one long-lived container keyed by that id so packages installed
// and files written by an earlier call are visible to a later one (e.g. `pip
// install python-docx` then a script that imports it). CloseSession tears the
// container down when the conversation ends; a backend that does not implement
// this interface simply runs every call ephemerally.
type SessionSandbox interface {
	Sandbox
	// CloseSession reaps the persistent container for id, if any. It is
	// idempotent: closing an unknown or already-closed session is not an error.
	CloseSession(id string) error
}

// ProcessSandbox is the optional interactive-process capability implemented by
// backends that can keep stdin/stdout open while a command runs. Protocol tools
// such as LSP and DAP need this; ordinary shell/file tools intentionally stay on
// the smaller run-to-completion Sandbox contract.
type ProcessSandbox interface {
	Sandbox
	// Start must enforce SandboxExec.Constraints.TimeoutSeconds for the whole
	// process lifetime and hard-kill the process group/container when it expires.
	// Retained protocol tools rely on that backend-independent lifetime bound.
	Start(ctx context.Context, req SandboxExec) (SandboxProcess, error)
}

// SandboxProcess is one started interactive command. Callers own all three
// pipes and must call Wait. Kill is idempotent and terminates the command's
// process group or container, not merely its direct child.
type SandboxProcess interface {
	Stdin() io.WriteCloser
	Stdout() io.ReadCloser
	Stderr() io.ReadCloser
	Wait() (SandboxResult, error)
	Kill() error
}

// SandboxExec is one sandboxed command request.
type SandboxExec struct {
	// Argv is the command and its arguments; Argv[0] is resolved against the
	// sandbox image's PATH, not the host's.
	Argv []string
	// Stdin is fed to the command's standard input.
	Stdin string
	// Env is the ONLY environment visible inside the sandbox. The host process
	// environment is never inherited — this is the property that stops a
	// prompt-injected command from reading DB creds or API keys.
	Env map[string]string
	// Mounts are explicit host paths exposed to the sandbox. Backends must reject
	// mounts they cannot enforce; callers provide only paths already narrowed by a
	// workspace guard.
	Mounts []SandboxMount
	// Workdir overrides the working directory the command starts in. Empty keeps
	// the backend default (the ephemeral scratch workdir). Set it to a mount
	// target so the command runs against shared workspace files.
	Workdir string
	// Session, when non-empty, requests persistent execution: a SessionSandbox
	// reuses one long-lived container keyed by this id across calls, so installed
	// packages and written files survive between tool invocations. Empty (the
	// default) runs the command in a fresh, throwaway container — the fail-safe
	// path every backend supports. The container's network/mount/limit envelope
	// is fixed by the first call that opens the session.
	Session string
	// Image, when non-empty, requests a specific sandbox image for this exec,
	// overriding the backend's default image selection. It lets distinct tools run
	// in purpose-built images within one host (e.g. computer_use in a doc-toolchain
	// image, browser_use in a Chrome image) without sharing a container. A backend
	// that cannot honor it MUST ignore it and fall back to its default — the
	// fail-safe path; agentcore stays generic and never interprets the value.
	Image string
	// Constraints are the resource + isolation caps for this execution.
	Constraints SandboxLimits
}

// SandboxMount exposes one host directory or file inside a sandboxed execution.
type SandboxMount struct {
	Source   string
	Target   string
	ReadOnly bool
}

// SandboxLimits are the isolation and resource caps applied to one execution.
// The zero value is fail-closed: no network, read-only root filesystem, and the
// default resource caps applied by the backend.
type SandboxLimits struct {
	Network bool // false (default) = no network egress at all
	// NetworkAllow confines egress, when Network is true, to these hosts
	// (+subdomains) via the sandbox filtering proxy. It is never "empty = open":
	// Network true with an empty list is refused, and unrestricted egress is the
	// explicit sandbox.AnyHost ("*") entry.
	NetworkAllow []string
	WritableFS   bool // false (default) = read-only root + small writable workdir
	// RunAsHostUser requests the host process UID/GID inside a container while
	// preserving a read-only root. It is for writable bind-mounted workspaces:
	// nobody often cannot modify a 0755 host directory, while container-root
	// would leave root-owned files behind. Backends that cannot map numeric host
	// identities may ignore it and retain their unprivileged default.
	RunAsHostUser  bool
	MemoryMB       int     // 0 = backend default
	CPUs           float64 // 0 = backend default
	PidsLimit      int     // 0 = backend default
	TimeoutSeconds float64 // 0 = backend default; negative disables the HostSandbox deadline; positive hard-kills after this elapses
}

// SandboxResult is the captured outcome of a sandboxed execution.
type SandboxResult struct {
	ExitCode   int
	Stdout     string
	Stderr     string
	Killed     bool   // true if the timeout fired and the command was killed
	KillReason string // human-readable kill cause, set when Killed
}

// Credentials at the trust boundary.
//
// The model writes {{cred:NAME}} where a secret belongs. The call is validated,
// traced and gated in that placeholder form; only the argument string handed
// to the executing tool carries the value. Three rules keep a prompt-injected
// model from turning that into exfiltration:
//
//  1. Default deny. A credential reaches a tool only under a CredentialGrant
//     naming that tool, and for a network tool only when every host the call
//     contacts is one the grant names.
//  2. Redaction. Every resolved value (raw, JSON-escaped and URL-escaped) is
//     replaced in the tool's output, partial updates and error text before an
//     interceptor, hook, trace or the model sees it.
//  3. Escaping. A value substituted into JSON arguments is JSON-escaped, so a
//     secret containing a quote cannot change the shape of the call.

// credentialPlaceholder matches {{cred:NAME}}; NAME is the only capture.
var credentialPlaceholder = regexp.MustCompile(`\{\{\s*cred:([A-Za-z0-9_.\-]{1,128})\s*\}\}`)

// CredentialGrant lets the named tools receive one credential. A placeholder
// no grant covers blocks the call.
type CredentialGrant struct {
	// Name is the credential as written in {{cred:NAME}}.
	Name string
	// Tools are the tool names that may receive it.
	Tools []string
	// Hosts, when non-empty, confines the credential to calls whose every
	// contacted host is one of these or a subdomain of one. It requires the
	// tool to implement NetworkTool; a tool that cannot say where it connects
	// never receives a host-scoped credential. Leave it empty only for a tool
	// that cannot reach arbitrary hosts.
	Hosts []string
}

// NetworkTool is an optional Tool capability: it reports the hosts one call
// will contact, from the call's placeholder-form arguments. Host-scoped
// credential grants rely on it, so the tool must also refuse to contact any
// other host (redirects included) once it is running.
type NetworkTool interface {
	NetworkHosts(args string) ([]string, error)
}

// resolveCredentials checks every placeholder in args against the grants,
// resolves the granted values, and returns the arguments the tool runs with
// plus a redactor for its output. With no resolver configured, args pass
// through unchanged and nothing is redacted.
func (e Env) resolveCredentials(ctx context.Context, tool Tool, args string) (string, *strings.Replacer, error) {
	if e.Credentials == nil {
		return args, nil, nil
	}
	matches := credentialPlaceholder.FindAllStringSubmatch(args, -1)
	if len(matches) == 0 {
		return args, nil, nil
	}
	values := map[string]string{}
	for _, m := range matches {
		name := m[1]
		if _, done := values[name]; done {
			continue
		}
		if err := e.checkGrant(name, tool, args); err != nil {
			return "", nil, err
		}
		value, err := e.Credentials.Resolve(ctx, "{{cred:"+name+"}}")
		if err != nil {
			return "", nil, err
		}
		if value == "" || credentialPlaceholder.MatchString(value) {
			return "", nil, fmt.Errorf("credential %q did not resolve", name)
		}
		values[name] = value
	}
	resolved := credentialPlaceholder.ReplaceAllStringFunc(args, func(token string) string {
		return jsonStringBody(values[credentialPlaceholder.FindStringSubmatch(token)[1]])
	})
	return resolved, secretRedactor(values), nil
}

// checkGrant reports why name may not reach tool on this call, or nil.
func (e Env) checkGrant(name string, tool Tool, args string) error {
	refusal := fmt.Errorf("credential %q is not granted to tool %q", name, tool.Name())
	for _, grant := range e.CredentialGrants {
		if grant.Name != name || !slices.Contains(grant.Tools, tool.Name()) {
			continue
		}
		if len(grant.Hosts) == 0 {
			return nil
		}
		network, ok := tool.(NetworkTool)
		if !ok {
			refusal = fmt.Errorf("credential %q is host-scoped and tool %q does not declare the hosts it contacts", name, tool.Name())
			continue
		}
		hosts, err := network.NetworkHosts(args)
		if err != nil || len(hosts) == 0 {
			refusal = fmt.Errorf("credential %q is host-scoped and the hosts of this %q call are unknown", name, tool.Name())
			continue
		}
		if allHostsGranted(hosts, grant.Hosts) {
			return nil
		}
		refusal = fmt.Errorf("credential %q is not granted for the hosts this %q call contacts", name, tool.Name())
	}
	return refusal
}

func allHostsGranted(hosts, granted []string) bool {
	for _, host := range hosts {
		host = normalizeHost(host)
		ok := false
		for _, allowed := range granted {
			allowed = normalizeHost(allowed)
			if allowed != "" && (host == allowed || strings.HasSuffix(host, "."+allowed)) {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	return true
}

func normalizeHost(host string) string {
	host = strings.TrimSpace(host)
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	return strings.TrimSuffix(strings.ToLower(host), ".")
}

// jsonStringBody escapes s for insertion inside a JSON string literal.
func jsonStringBody(s string) string {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s)
	out := strings.TrimSuffix(b.String(), "\n")
	return out[1 : len(out)-1]
}

// secretRedactor replaces every spelling of each resolved value a tool is
// likely to echo with a marker naming the credential. Longer spellings are
// matched first so one secret that contains another is redacted whole.
func secretRedactor(values map[string]string) *strings.Replacer {
	type pair struct{ old, marker string }
	var pairs []pair
	for name, value := range values {
		marker := "[redacted cred:" + name + "]"
		seen := map[string]bool{}
		for _, spelling := range []string{value, jsonStringBody(value), url.QueryEscape(value), url.PathEscape(value)} {
			if spelling != "" && !seen[spelling] {
				seen[spelling] = true
				pairs = append(pairs, pair{spelling, marker})
			}
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return len(pairs[i].old) > len(pairs[j].old) })
	args := make([]string, 0, 2*len(pairs))
	for _, p := range pairs {
		args = append(args, p.old, p.marker)
	}
	return strings.NewReplacer(args...)
}
