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
	"fmt"
	"net"
	"os/exec"
	"os/user"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
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

	// egress caches one host-side filtering proxy per distinct allowlist (#5b).
	// nil until the first networked run with a NetworkAllow list.
	egress *egressProxyPool
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
// egress for the given limits, plus any proxy env to inject (#5b):
//
//   - no network → ["--network","none"], no env.
//   - network, no allowlist → nil (default network; current behavior).
//   - network + allowlist → route through the host-side filtering proxy: set
//     HTTP(S)_PROXY to it (reached via the host gateway) so allowlisted hosts are
//     reachable and everything else is hard-denied by the proxy. The container is
//     additionally kept on the host gateway route via --add-host so the proxy is
//     resolvable. On hosts that support an internal bridge, docker.go confines the
//     workload further; where it cannot, the proxy still hard-denies non-listed
//     hosts for any client that honors proxy env (pip/npm/apt/curl do by default).
//
// A best-effort proxy start failure falls back to no network rather than opening
// the container to the whole internet — fail closed.
func (s *DockerSandbox) egressNetworkArgs(lim agentcore.SandboxLimits) (netArgs []string, env map[string]string) {
	if !lim.Network {
		return []string{"--network", "none"}, nil
	}
	allow := newEgressAllow(lim.NetworkAllow)
	if len(allow.hosts) == 0 {
		return nil, nil // network on, no allowlist: unchanged default-network behavior
	}
	proxy, err := s.egress.get(allow)
	if err != nil || proxy == nil {
		// Could not stand up the filter; deny all egress rather than allow all.
		return []string{"--network", "none"}, nil
	}
	_, port, splitErr := net.SplitHostPort(proxy.Addr())
	if splitErr != nil {
		return []string{"--network", "none"}, nil
	}
	// host-gateway lets the container reach the proxy listening on the host.
	proxyURL := "http://host.docker.internal:" + port
	return []string{"--add-host", "host.docker.internal:host-gateway"}, map[string]string{
		"HTTP_PROXY":  proxyURL,
		"HTTPS_PROXY": proxyURL,
		"http_proxy":  proxyURL,
		"https_proxy": proxyURL,
		"NO_PROXY":    "localhost,127.0.0.1",
	}
}

// StopEgress tears down every cached egress proxy. Called on sandbox shutdown.
func (s *DockerSandbox) StopEgress() {
	if s.egress != nil {
		s.egress.stopAll()
	}
}

// Image returns the configured sandbox image (used by tests / diagnostics).
func (s *DockerSandbox) Image() string { return s.image }

// Available reports whether the docker runtime is reachable.
func (s *DockerSandbox) Available(ctx context.Context) bool {
	return exec.CommandContext(ctx, s.docker, "info").Run() == nil
}

// oneShotArgs renders the common hardened `docker run` envelope used by both
// buffered Exec and interactive Start. Keeping one renderer is a security
// invariant: adding streaming protocols must not silently lose a cap, mount,
// user, network, or environment restriction.
func (s *DockerSandbox) oneShotArgs(req agentcore.SandboxExec, lim agentcore.SandboxLimits, name string) ([]string, error) {
	workdir := sandboxWorkdir
	if strings.TrimSpace(req.Workdir) != "" {
		workdir = req.Workdir
	}
	args := []string{
		"run", "--rm", "-i",
		"--name", name,
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
		"--user", sandboxUser(lim),
		"--workdir", workdir,
		"--tmpfs", sandboxTmpfs(lim),
		"--memory", fmt.Sprintf("%dm", lim.MemoryMB),
		"--memory-swap", fmt.Sprintf("%dm", lim.MemoryMB),
		"--cpus", strconv.FormatFloat(lim.CPUs, 'f', 2, 64),
		"--pids-limit", strconv.Itoa(lim.PidsLimit),
	}
	egressArgs, egressEnv := s.egressNetworkArgs(lim)
	args = append(args, egressArgs...)
	if !lim.WritableFS {
		args = append(args, "--read-only")
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
	for k, v := range req.Env {
		args = append(args, "--env", k+"="+v)
	}
	for k, v := range egressEnv {
		args = append(args, "--env", k+"="+v)
	}
	image := s.image
	if strings.TrimSpace(req.Image) != "" {
		image = req.Image
	}
	args = append(args, image)
	args = append(args, req.Argv...)
	return args, nil
}

// Exec runs req to completion in a fresh hardened container and returns the
// captured output. A non-zero exit code is returned as a SandboxResult (not an
// error); error is reserved for the backend itself failing.
func (s *DockerSandbox) Exec(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	if len(req.Argv) == 0 {
		return agentcore.SandboxResult{}, fmt.Errorf("sandbox: empty argv")
	}
	// A session request is served by a reused, long-lived container so installs
	// and files persist across calls; the empty-session path below stays the
	// fresh-container-per-call default.
	if strings.TrimSpace(req.Session) != "" {
		return s.execSession(ctx, req)
	}
	lim := withDefaults(req.Constraints)
	name := "agentray-sbx-" + randHex(8)

	timeout := time.Duration(lim.TimeoutSeconds * float64(time.Second))
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args, err := s.oneShotArgs(req, lim, name)
	if err != nil {
		return agentcore.SandboxResult{}, err
	}

	cmd := exec.CommandContext(runCtx, s.docker, args...)
	if req.Stdin != "" {
		cmd.Stdin = strings.NewReader(req.Stdin)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	res := agentcore.SandboxResult{Stdout: stdout.String(), Stderr: stderr.String()}

	if runCtx.Err() == context.DeadlineExceeded {
		// Killing the docker CLI does not reliably reap the container, so force
		// it with a short, independent context.
		rmCtx, rmCancel := context.WithTimeout(context.Background(), 10*time.Second)
		_ = exec.CommandContext(rmCtx, s.docker, "rm", "-f", name).Run()
		rmCancel()
		res.Killed = true
		res.KillReason = fmt.Sprintf("exceeded %.0fs timeout", lim.TimeoutSeconds)
		return res, nil
	}
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			// Non-zero exit from the sandboxed command is a normal result.
			res.ExitCode = ee.ExitCode()
			return res, nil
		}
		return res, fmt.Errorf("sandbox: docker run: %w", runErr)
	}
	return res, nil
}

// Start launches an interactive command in the same fresh hardened container
// Exec uses, exposing protocol pipes until Wait. The container is force-removed
// after Wait or Kill because terminating the docker CLI alone is not sufficient
// to reap a detached child on every daemon/platform combination.
func (s *DockerSandbox) Start(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxProcess, error) {
	if len(req.Argv) == 0 {
		return nil, fmt.Errorf("sandbox: empty argv")
	}
	if req.Stdin != "" {
		return nil, fmt.Errorf("sandbox: interactive process cannot use preloaded stdin")
	}
	if strings.TrimSpace(req.Session) != "" {
		return nil, fmt.Errorf("sandbox: interactive persistent sessions are not supported")
	}
	lim := withDefaults(req.Constraints)
	name := "agentray-sbx-" + randHex(8)
	args, err := s.oneShotArgs(req, lim, name)
	if err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(lim.TimeoutSeconds*float64(time.Second)))
	cmd := exec.CommandContext(runCtx, s.docker, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("sandbox: docker stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("sandbox: docker stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("sandbox: docker stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("sandbox: docker start: %w", err)
	}
	remove := func() {
		rmCtx, rmCancel := context.WithTimeout(context.Background(), 10*time.Second)
		_ = exec.CommandContext(rmCtx, s.docker, "rm", "-f", name).Run()
		rmCancel()
	}
	return &commandProcess{
		cmd: cmd, stdin: stdin, stdout: stdout, stderr: stderr,
		runCtx: runCtx, cancel: cancel, killFn: func() error {
			remove()
			return nil
		},
		cleanup: remove, errLabel: "docker run", timeout: fmt.Sprintf("exceeded %.0fs timeout", lim.TimeoutSeconds),
	}, nil
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

// execSession runs req inside the persistent container for req.Session, creating
// it on first use. The container's isolation envelope (network, mounts, user,
// writability, resource caps) is fixed by this first call; later calls only run
// commands inside it, so `pip install X` then a script importing X works.
func (s *DockerSandbox) execSession(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	lim := withDefaults(req.Constraints)
	name, err := s.ensureSession(ctx, req, lim)
	if err != nil {
		return agentcore.SandboxResult{}, err
	}

	timeout := time.Duration(lim.TimeoutSeconds * float64(time.Second))
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := []string{"exec", "-i"}
	if w := strings.TrimSpace(req.Workdir); w != "" {
		args = append(args, "--workdir", w)
	}
	for k, v := range req.Env {
		args = append(args, "--env", k+"="+v)
	}
	args = append(args, name)
	args = append(args, req.Argv...)

	cmd := exec.CommandContext(runCtx, s.docker, args...)
	if req.Stdin != "" {
		cmd.Stdin = strings.NewReader(req.Stdin)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	res := agentcore.SandboxResult{Stdout: stdout.String(), Stderr: stderr.String()}
	if runCtx.Err() == context.DeadlineExceeded {
		// The command ran too long; the session container itself stays alive so
		// the agent can retry with a smaller step. Only this exec is killed.
		res.Killed = true
		res.KillReason = fmt.Sprintf("exceeded %.0fs timeout", lim.TimeoutSeconds)
		return res, nil
	}
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			res.ExitCode = ee.ExitCode()
			return res, nil
		}
		return res, fmt.Errorf("sandbox: docker exec: %w", runErr)
	}
	return res, nil
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
	rmCtx, rmCancel := context.WithTimeout(context.Background(), 10*time.Second)
	_ = exec.CommandContext(rmCtx, s.docker, "rm", "-f", name).Run()
	rmCancel()

	args := []string{
		"run", "-d",
		"--name", name,
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
		"--user", sandboxUser(lim),
		"--memory", fmt.Sprintf("%dm", lim.MemoryMB),
		"--memory-swap", fmt.Sprintf("%dm", lim.MemoryMB),
		"--cpus", strconv.FormatFloat(lim.CPUs, 'f', 2, 64),
		"--pids-limit", strconv.Itoa(lim.PidsLimit),
	}
	egressArgs, egressEnv := s.egressNetworkArgs(lim)
	args = append(args, egressArgs...)
	if !lim.WritableFS {
		args = append(args,
			"--read-only",
			"--tmpfs", sandboxTmpfs(lim),
			"--workdir", sandboxWorkdir,
		)
	}
	for _, m := range req.Mounts {
		if strings.TrimSpace(m.Source) == "" || strings.TrimSpace(m.Target) == "" {
			return "", fmt.Errorf("sandbox: invalid mount")
		}
		spec := fmt.Sprintf("type=bind,src=%s,dst=%s", m.Source, m.Target)
		if m.ReadOnly {
			spec += ",readonly"
		}
		args = append(args, "--mount", spec)
	}
	for k, v := range req.Env {
		args = append(args, "--env", k+"="+v)
	}
	for k, v := range egressEnv {
		args = append(args, "--env", k+"="+v)
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
	rmCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(rmCtx, s.docker, "rm", "-f", name).Run()
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
