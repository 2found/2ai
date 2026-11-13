// Package sandbox provides isolation backends and the tools that run inside
// them, implementing agentcore.Sandbox. It is an edge package: it shells out to
// the host's container runtime, so it lives outside the agentcore leaf and is
// injected by the host (agentruntime), never imported by the core.
package sandbox

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os/exec"
	"os/user"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

// Hardening defaults. Every execution gets a fresh, throwaway container that
// drops all Linux capabilities, cannot gain new privileges, runs as an
// unprivileged UID, has a read-only root with only a small tmpfs workdir, and —
// unless explicitly granted — no network at all. These mirror the flags the AGT
// Go SDK and agent-sandbox Dockerfile apply, rebuilt here for agentray.
const (
	defaultImage     = "alpine:3.19"
	defaultMemoryMB  = 256
	defaultCPUs      = 1.0
	defaultPids      = 128
	defaultTimeoutS  = 30.0
	sandboxUID       = "65534:65534" // nobody:nogroup
	sandboxWorkdir   = "/work"
	sandboxTmpfsSize = "64m"

	// sessionMaxLifetimeS caps how long a persistent computer-use container stays
	// alive (its keepalive `sleep` argument). It self-reaps after this even if
	// CloseSession is never called, so a leaked session can't pin host resources
	// indefinitely; a later call simply recreates it.
	sessionMaxLifetimeS = 3600
	// rootUser runs a writable-FS computer-use session as container-root so
	// package managers (pip/apt/npm) can install system-wide. It is still hard
	// isolated — all Linux capabilities dropped, no-new-privileges, no host env,
	// resource caps — but it is the deliberate, policy-granted exception to the
	// nobody/read-only locked default the one-shot run_shell keeps.
	rootUser = "0:0"
)

// sessionNameRe sanitizes a session id into the [a-zA-Z0-9_.-] set docker
// container names allow, so an opaque conversation id keys a real container.
var sessionNameRe = regexp.MustCompile(`[^a-zA-Z0-9_.-]`)

// DockerSandbox runs each command in an ephemeral, hardened container via the
// host `docker` CLI. It holds no daemon connection and keeps no state between
// calls: every Exec is `docker run --rm` with a fresh container, so a
// compromised execution cannot persist or reach a sibling.
type DockerSandbox struct {
	docker  string // docker binary (default "docker")
	image   string // locked one-shot image (default defaultImage)
	cuImage string // persistent computer-use image (rich toolchain); empty = image

	// mu guards the persistent-session container registry. sessions maps a
	// sanitized session id to the running keepalive container's name; a missing
	// entry means the session is not yet opened (lazily created on first Exec).
	mu       sync.Mutex
	sessions map[string]string

	// egress caches one host-side filtering proxy per distinct allowlist;
	// network is the internal docker network allowlisted containers join,
	// created on first use (guarded by egressMu).
	egress   *egressProxyPool
	egressMu sync.Mutex
	network  egressNetwork
}

// Option configures a DockerSandbox.
type Option func(*DockerSandbox)

// WithImage overrides the sandbox image (default alpine:3.19). Use a hardened,
// minimal-PATH image in production.
func WithImage(image string) Option {
	return func(s *DockerSandbox) {
		if strings.TrimSpace(image) != "" {
			s.image = image
		}
	}
}

// WithComputerUseImage sets the image used for persistent computer-use sessions
// (a richer toolchain image with python/pandoc/office libraries). Empty leaves
// computer-use sessions on the default image. One-shot run_shell is unaffected.
func WithComputerUseImage(image string) Option {
	return func(s *DockerSandbox) {
		if strings.TrimSpace(image) != "" {
			s.cuImage = image
		}
	}
}

// WithDockerBinary overrides the docker CLI path (default "docker").
func WithDockerBinary(bin string) Option {
	return func(s *DockerSandbox) {
		if strings.TrimSpace(bin) != "" {
			s.docker = bin
		}
	}
}

// NewDockerSandbox builds a DockerSandbox with hardened defaults.
func NewDockerSandbox(opts ...Option) *DockerSandbox {
	s := &DockerSandbox{docker: "docker", image: defaultImage, sessions: map[string]string{}, egress: newEgressProxyPool()}
	for _, o := range opts {
		o(s)
	}
	return s
}

// egressNetworkArgs returns the docker run flags governing a container's network
// egress, plus any proxy env to inject:
//
//   - no network → ["--network","none"], no env.
//   - network + AnyHost → nil: the default bridge, unrestricted. Only the
//     explicit AnyHost entry gets here.
//   - network + allowlist → the container joins this sandbox's internal egress
//     network, which has no route off the bridge; the only peer it can reach is
//     the filtering proxy listening on that network's gateway address, named in
//     HTTP(S)_PROXY. A client that ignores the proxy env has nowhere to go.
//   - network + empty allowlist → an error. Open egress is never the silent
//     default.
//
// Any failure to stand up the filter is an error, never a fallback to open
// network.
func (s *DockerSandbox) egressNetworkArgs(ctx context.Context, lim agentcore.SandboxLimits) ([]string, map[string]string, error) {
	if !lim.Network {
		return []string{"--network", "none"}, nil, nil
	}
	allow := newEgressAllow(lim.NetworkAllow)
	if allow.unrestricted {
		return nil, nil, nil
	}
	if len(allow.rules) == 0 {
		return nil, nil, errNoEgressAllowlist
	}
	network, err := s.egressNetwork(ctx)
	if err != nil {
		return nil, nil, err
	}
	proxy, err := s.egress.get(allow, network.gateway)
	if err != nil {
		return nil, nil, fmt.Errorf("sandbox: egress proxy cannot listen on the egress network gateway %s "+
			"(an allowlist needs the Docker bridge on this host, i.e. a native Linux engine): %w", network.gateway, err)
	}
	proxyURL := "http://" + proxy.Addr()
	return []string{"--network", network.name}, map[string]string{
		"HTTP_PROXY":  proxyURL,
		"HTTPS_PROXY": proxyURL,
		"http_proxy":  proxyURL,
		"https_proxy": proxyURL,
		"NO_PROXY":    "localhost,127.0.0.1",
		"no_proxy":    "localhost,127.0.0.1",
	}, nil
}

// errNoEgressAllowlist is returned for Network=true with no allowlist.
var errNoEgressAllowlist = fmt.Errorf("sandbox: network granted without an egress allowlist; " +
	"list the hosts in NetworkAllow, or pass sandbox.AnyHost for unrestricted egress")

// egressNetwork is the internal docker network allowlisted containers join.
type egressNetwork struct{ name, gateway string }

// egressNetwork returns this sandbox's internal egress network, creating it on
// first use. It is --internal (Docker installs no route or NAT off the bridge)
// with inter-container traffic disabled, so allowlisted sessions cannot reach
// each other either; the host-side proxy on the gateway address is the only
// reachable peer.
func (s *DockerSandbox) egressNetwork(ctx context.Context) (egressNetwork, error) {
	s.egressMu.Lock()
	defer s.egressMu.Unlock()
	if s.network.name != "" {
		return s.network, nil
	}
	c, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	name := "agentray-egress-" + randHex(6)
	if out, err := exec.CommandContext(c, s.docker, "network", "create", "--internal", "--driver", "bridge",
		"--opt", "com.docker.network.bridge.enable_icc=false", "--label", "2ai.egress=1", name).CombinedOutput(); err != nil {
		return egressNetwork{}, fmt.Errorf("sandbox: create egress network: %w: %s", err, strings.TrimSpace(string(out)))
	}
	out, err := exec.CommandContext(c, s.docker, "network", "inspect", "-f", "{{range .IPAM.Config}}{{.Gateway}} {{end}}", name).Output()
	gateway := ""
	for _, field := range strings.Fields(string(out)) {
		if ip := net.ParseIP(field); ip != nil && ip.To4() != nil {
			gateway = field
			break
		}
	}
	if err != nil || gateway == "" {
		s.removeNetwork(name)
		return egressNetwork{}, fmt.Errorf("sandbox: egress network %s has no IPv4 gateway (inspect: %v)", name, err)
	}
	s.network = egressNetwork{name: name, gateway: gateway}
	return s.network, nil
}

func (s *DockerSandbox) removeNetwork(name string) {
	c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(c, s.docker, "network", "rm", name).Run()
}

// StopEgress tears down every cached egress proxy and the egress network.
// Called on sandbox shutdown.
func (s *DockerSandbox) StopEgress() {
	s.egress.stopAll()
	s.egressMu.Lock()
	network := s.network
	s.network = egressNetwork{}
	s.egressMu.Unlock()
	if network.name != "" {
		s.removeNetwork(network.name)
	}
}

// Image returns the configured sandbox image (used by tests / diagnostics).
func (s *DockerSandbox) Image() string { return s.image }

// Available reports whether the docker runtime is reachable.
func (s *DockerSandbox) Available(ctx context.Context) bool {
	return exec.CommandContext(ctx, s.docker, "info").Run() == nil
}

// envelopeArgs renders the hardened isolation flags every container gets —
// one-shot and session alike. Keeping one renderer is a security invariant:
// a new execution path must not silently lose a cap, mount, user, network, or
// environment restriction.
func (s *DockerSandbox) envelopeArgs(ctx context.Context, req agentcore.SandboxExec, lim agentcore.SandboxLimits) ([]string, error) {
	args := []string{
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
		"--user", sandboxUser(lim),
		"--memory", fmt.Sprintf("%dm", lim.MemoryMB),
		"--memory-swap", fmt.Sprintf("%dm", lim.MemoryMB),
		"--cpus", strconv.FormatFloat(lim.CPUs, 'f', 2, 64),
		"--pids-limit", strconv.Itoa(lim.PidsLimit),
	}
	egressArgs, egressEnv, err := s.egressNetworkArgs(ctx, lim)
	if err != nil {
		return nil, err
	}
	args = append(args, egressArgs...)
	if !lim.WritableFS {
		args = append(args, "--read-only", "--tmpfs", sandboxTmpfs(lim))
	}
	for _, m := range req.Mounts {
		if strings.TrimSpace(m.Source) == "" || strings.TrimSpace(m.Target) == "" {
			return nil, fmt.Errorf("sandbox: invalid mount")
		}
		spec := fmt.Sprintf("type=bind,src=%s,dst=%s", m.Source, m.Target)
		if m.ReadOnly {
			spec += ",readonly"
		}
		args = append(args, "--mount", spec)
	}
	args = appendEnv(args, req.Env)
	return appendEnv(args, egressEnv), nil
}

func appendEnv(args []string, env map[string]string) []string {
	for k, v := range env {
		args = append(args, "--env", k+"="+v)
	}
	return args
}

// dockerRun is one prepared docker CLI invocation: its arguments and how to
// stop what it started. Killing the docker CLI alone does not stop the
// workload — the container (one-shot) or the exec'd process tree (session)
// lives in the daemon — so kill reaches into the daemon.
type dockerRun struct {
	args    []string
	lim     agentcore.SandboxLimits
	kill    func() // stops the workload; safe to call more than once
	cleanup func() // runs after the CLI exits
}

// prepare renders req into a docker invocation: `docker run --rm` of a fresh
// hardened container, or `docker exec` into the persistent session container
// when req.Session is set.
func (s *DockerSandbox) prepare(ctx context.Context, req agentcore.SandboxExec) (dockerRun, error) {
	if len(req.Argv) == 0 {
		return dockerRun{}, fmt.Errorf("sandbox: empty argv")
	}
	lim := withDefaults(req.Constraints)
	if strings.TrimSpace(req.Session) != "" {
		name, err := s.ensureSession(ctx, req, lim)
		if err != nil {
			return dockerRun{}, err
		}
		pidfile := sessionRunDir + "/" + randHex(8) + ".pid"
		args := []string{"exec", "-i"}
		if w := strings.TrimSpace(req.Workdir); w != "" {
			args = append(args, "--workdir", w)
		}
		args = appendEnv(args, req.Env)
		args = append(args, name, "/bin/sh", "-c", sessionExecWrapper, pidfile)
		kill := func() { s.killSessionExec(name, pidfile) }
		return dockerRun{args: append(args, req.Argv...), lim: lim, kill: kill, cleanup: func() {}}, nil
	}
	name := "agentray-sbx-" + randHex(8)
	workdir := sandboxWorkdir
	if strings.TrimSpace(req.Workdir) != "" {
		workdir = req.Workdir
	}
	envelope, err := s.envelopeArgs(ctx, req, lim)
	if err != nil {
		return dockerRun{}, err
	}
	args := append([]string{"run", "--rm", "-i", "--name", name, "--workdir", workdir}, envelope...)
	image := s.image
	if strings.TrimSpace(req.Image) != "" {
		image = req.Image
	}
	args = append(append(args, image), req.Argv...)
	remove := func() { s.removeContainer(name) }
	return dockerRun{args: args, lim: lim, kill: remove, cleanup: remove}, nil
}

// sessionRunDir is a small tmpfs in every session container holding the pid
// files that let a timed-out or cancelled exec be killed inside the container.
const sessionRunDir = "/run/2ai"

// sessionExecWrapper records the shell's pid before running the command, so
// the whole process tree under it can be killed. It does not exec: staying the
// parent keeps one pid that every descendant hangs off.
const sessionExecWrapper = `echo $$ > "$0" || exit 125; "$@"; s=$?; rm -f "$0"; exit $s`

// sessionKillScript stops then kills the process tree rooted at the pid in
// $0, walking /proc children so it needs no procps in the image. Each process
// is stopped before its children are listed, so none can fork past the walk.
const sessionKillScript = `t() { kill -STOP "$1" 2>/dev/null; for c in $(cat /proc/"$1"/task/*/children 2>/dev/null); do t "$c"; done; kill -KILL "$1" 2>/dev/null; }
p=$(cat "$0" 2>/dev/null) && [ -n "$p" ] && t "$p"; rm -f "$0"`

func (s *DockerSandbox) killSessionExec(container, pidfile string) {
	c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(c, s.docker, "exec", container, "/bin/sh", "-c", sessionKillScript, pidfile).Run()
}

func (s *DockerSandbox) removeContainer(name string) {
	c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(c, s.docker, "rm", "-f", name).Run()
}

// command builds the docker CLI process for run. When ctx ends — timeout or
// caller cancellation — the workload is killed in the daemon, then the CLI.
func (s *DockerSandbox) command(ctx context.Context, run dockerRun) *exec.Cmd {
	cmd := exec.CommandContext(ctx, s.docker, run.args...)
	cmd.Cancel = func() error {
		run.kill()
		return cmd.Process.Kill()
	}
	cmd.WaitDelay = 10 * time.Second
	return cmd
}

// Exec runs req to completion in a fresh hardened container (or in the
// persistent session container when req.Session is set) and returns the
// captured output. A non-zero exit code is returned as a SandboxResult (not an
// error); error is reserved for the backend itself failing.
func (s *DockerSandbox) Exec(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	run, err := s.prepare(ctx, req)
	if err != nil {
		return agentcore.SandboxResult{}, err
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(run.lim.TimeoutSeconds*float64(time.Second)))
	defer cancel()
	cmd := s.command(runCtx, run)
	if req.Stdin != "" {
		cmd.Stdin = strings.NewReader(req.Stdin)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()
	run.cleanup()

	res := agentcore.SandboxResult{Stdout: stdout.String(), Stderr: stderr.String()}
	// A caller's cancellation is an error; only the deadline is a result.
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		res.Killed = true
		res.KillReason = fmt.Sprintf("exceeded %gs timeout", run.lim.TimeoutSeconds)
		return res, nil
	}
	if runErr != nil {
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			res.ExitCode = ee.ExitCode()
			return res, nil
		}
		return res, fmt.Errorf("sandbox: docker %s: %w", run.args[0], runErr)
	}
	return res, nil
}

// Start launches an interactive command — in a fresh hardened container, or in
// the persistent session container when req.Session is set — exposing its pipes
// until Wait. The workload is killed in the daemon on Kill, timeout, or caller
// cancellation, because terminating the docker CLI alone does not stop it.
func (s *DockerSandbox) Start(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxProcess, error) {
	if req.Stdin != "" {
		return nil, fmt.Errorf("sandbox: interactive process cannot use preloaded stdin")
	}
	run, err := s.prepare(ctx, req)
	if err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(run.lim.TimeoutSeconds*float64(time.Second)))
	cmd := s.command(runCtx, run)
	proc, err := startCommand(cmd, runCtx, cancel, "docker "+run.args[0])
	if err != nil {
		run.cleanup()
		return nil, err
	}
	proc.killFn = func() error { run.kill(); return nil }
	proc.cleanup = run.cleanup
	proc.timeout = fmt.Sprintf("exceeded %gs timeout", run.lim.TimeoutSeconds)
	return proc, nil
}

// sandboxUser picks the in-container UID. The locked default (read-only root)
// runs as unprivileged nobody. A writable-FS computer-use profile runs as
// container-root so package managers can install system-wide — still with all
// capabilities dropped, no-new-privileges, and no host env, so root *inside*
// this throwaway container cannot escalate to or see the host.
func sandboxUser(lim agentcore.SandboxLimits) string {
	if lim.WritableFS {
		return rootUser
	}
	if lim.RunAsHostUser {
		if current, err := user.Current(); err == nil {
			if _, uidErr := strconv.ParseUint(current.Uid, 10, 32); uidErr == nil {
				if _, gidErr := strconv.ParseUint(current.Gid, 10, 32); gidErr == nil {
					return current.Uid + ":" + current.Gid
				}
			}
		}
	}
	return sandboxUID
}

func sandboxTmpfs(lim agentcore.SandboxLimits) string {
	uid, gid := "65534", "65534"
	if parsedUID, parsedGID, ok := strings.Cut(sandboxUser(lim), ":"); ok {
		uid, gid = parsedUID, parsedGID
	}
	return fmt.Sprintf("%s:rw,size=%s,uid=%s,gid=%s", sandboxWorkdir, sandboxTmpfsSize, uid, gid)
}

// ensureSession returns the running keepalive container's name for the session,
// starting it on first use. If a previously-started container has since died
// (its keepalive sleep expired, or it was reaped), it is transparently recreated
// so a long conversation never wedges on a dead session.
func (s *DockerSandbox) ensureSession(ctx context.Context, req agentcore.SandboxExec, lim agentcore.SandboxLimits) (string, error) {
	key := sessionNameRe.ReplaceAllString(req.Session, "-")
	name := "agentray-ses-" + key

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.sessions[key]; ok && s.containerRunning(ctx, name) {
		return name, nil
	}
	// Stale or never-created: clear any dead container of this name first so the
	// run --name does not collide, then start fresh.
	s.removeContainer(name)

	envelope, err := s.envelopeArgs(ctx, req, lim)
	if err != nil {
		return "", err
	}
	args := append([]string{"run", "-d", "--name", name,
		"--tmpfs", sessionRunDir + ":rw,size=1m,mode=1777"}, envelope...)
	if !lim.WritableFS {
		args = append(args, "--workdir", sandboxWorkdir)
	}
	// Image selection: an explicit per-exec Image wins (browser_use's Chrome
	// image); else the configured computer-use image; else the locked default.
	image := s.image
	if s.cuImage != "" {
		image = s.cuImage
	}
	if strings.TrimSpace(req.Image) != "" {
		image = req.Image
	}
	args = append(args, image, "sleep", strconv.Itoa(sessionMaxLifetimeS))

	startCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(startCtx, s.docker, args...).CombinedOutput(); err != nil {
		return "", fmt.Errorf("sandbox: start session container: %w: %s", err, strings.TrimSpace(string(out)))
	}
	s.sessions[key] = name
	return name, nil
}

// containerRunning reports whether the named container is up (cheap inspect).
func (s *DockerSandbox) containerRunning(ctx context.Context, name string) bool {
	c, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(c, s.docker, "inspect", "-f", "{{.State.Running}}", name).Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// CloseSession reaps the persistent container for id. It is idempotent and
// satisfies agentcore.SessionSandbox so the runtime can tear a session down when
// a conversation ends.
func (s *DockerSandbox) CloseSession(id string) error {
	key := sessionNameRe.ReplaceAllString(id, "-")
	s.mu.Lock()
	name, ok := s.sessions[key]
	delete(s.sessions, key)
	s.mu.Unlock()
	if !ok {
		name = "agentray-ses-" + key
	}
	s.removeContainer(name)
	return nil
}

// withDefaults fills unset (zero) limits with the hardened defaults. It never
// loosens an explicit value.
func withDefaults(l agentcore.SandboxLimits) agentcore.SandboxLimits {
	if l.MemoryMB <= 0 {
		l.MemoryMB = defaultMemoryMB
	}
	if l.CPUs <= 0 {
		l.CPUs = defaultCPUs
	}
	if l.PidsLimit <= 0 {
		l.PidsLimit = defaultPids
	}
	if l.TimeoutSeconds <= 0 {
		l.TimeoutSeconds = defaultTimeoutS
	}
	return l
}

// randHex returns 2*n hex chars from crypto/rand for a collision-free container
// name. A broken entropy source is fatal — we will not run untrusted code under
// a predictable name.
func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(fmt.Sprintf("sandbox: crypto/rand failed: %v", err))
	}
	return hex.EncodeToString(b)
}
