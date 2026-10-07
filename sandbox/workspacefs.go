package sandbox

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/2found/2ai/agentcore"
)

// workspaceFS is the I/O substrate the file and search tools (read_file,
// write_file, edit_file, edit_lines, grep, glob) run on. It exists so those tools have one
// implementation and two backends: hostFS does the os.ReadFile / WalkDir calls
// they have always done, sandboxFS routes the same primitives through
// agentcore.Sandbox.Exec. Every byte of windowing, line numbering, fuzzy
// matching and grep context stays above this seam, which is what makes the two
// substrates produce identical output over identical trees.
//
// Every path is workspace-relative and slash-separated — implementations
// validate it through Workspace.Resolve and never see an absolute or escaping
// path.
type workspaceFS interface {
	// Stat reports the size of a workspace-relative path and whether it is a
	// directory. A missing path is an error.
	Stat(ctx context.Context, rel string) (fsStat, error)
	// ReadFile returns the whole contents of a workspace-relative file.
	ReadFile(ctx context.Context, rel string) ([]byte, error)
	// ReadRange returns a bounded byte window and a hash of the full file.
	ReadRange(ctx context.Context, rel string, offset int64, limit int) ([]byte, string, error)
	// WriteFile writes data at rel, creating parent directories.
	WriteFile(ctx context.Context, rel string, data []byte) error
	// List returns every regular file under the workspace-relative directory
	// root ("" = the whole workspace), pruning skipDir directories, in the order
	// filepath.WalkDir would visit them.
	List(ctx context.Context, root string) ([]string, error)
	// ReadEach calls fn once per readable file in rels, in order, skipping any
	// file larger than maxBytes and any file that cannot be read (fn receives nil
	// data for these skipped files). It exists so a
	// tree-wide search is one round trip on the sandbox substrate instead of one
	// exec per file. fn returning an error stops the walk and is returned as-is,
	// so a caller can stop early with errStopRead.
	ReadEach(ctx context.Context, rels []string, maxBytes int64, fn func(rel string, data []byte) error) error
}

// fsStat is the subset of os.FileInfo the file tools actually use.
type fsStat struct {
	Size  int64
	IsDir bool
}

// errStopRead is the sentinel a ReadEach callback returns to end the walk
// without reporting a failure (grep's match cap).
var errStopRead = errors.New("stop reading")

// newWorkspaceFS picks the substrate: a nil sandbox means the tool runs
// directly on the host machine (the default), non-nil means the same operations
// are executed inside the sandbox with the workspace bind-mounted.
func newWorkspaceFS(sb agentcore.Sandbox, ws *Workspace) workspaceFS {
	if sb == nil {
		return hostFS{ws: ws}
	}
	return sandboxFS{sb: sb, ws: ws}
}

// walkOrderLess orders two slash-separated relative paths the way
// filepath.WalkDir visits them: segment by segment, each level sorted by name.
// It differs from plain string comparison — "a/b.txt" precedes "a.txt" because
// the first segments compare "a" < "a.txt" — which is exactly why the sandbox
// substrate cannot just sort its find(1) output lexically and expect grep to
// emit the same lines in the same order as the host.
func walkOrderLess(a, b string) bool {
	as, bs := strings.Split(a, "/"), strings.Split(b, "/")
	for i := 0; i < len(as) && i < len(bs); i++ {
		if as[i] != bs[i] {
			return as[i] < bs[i]
		}
	}
	return len(as) < len(bs)
}

// ---------------------------------------------------------------------------
// host substrate
// ---------------------------------------------------------------------------

// hostFS runs the file tools' I/O directly on the host, under the Workspace
// guards that have always protected them: relative paths only, symlinks
// resolved and re-checked against the root, and a refusal to follow a symlink
// out on write.
type hostFS struct{ ws *Workspace }

func (h hostFS) Stat(ctx context.Context, rel string) (fsStat, error) {
	if err := ctx.Err(); err != nil {
		return fsStat{}, err
	}
	resolved, err := h.resolveExisting(rel)
	if err != nil {
		return fsStat{}, err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return fsStat{}, err
	}
	return fsStat{Size: info.Size(), IsDir: info.IsDir()}, nil
}

func (h hostFS) ReadFile(ctx context.Context, rel string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resolved, err := h.resolveExisting(rel)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(resolved)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return data, err
}

func (h hostFS) WriteFile(_ context.Context, rel string, data []byte) error {
	abs, _, err := h.ws.Resolve(rel)
	if err != nil {
		return err
	}
	dir := filepath.Dir(abs)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	resolvedDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return err
	}
	if !inside(h.ws.Root(), resolvedDir) {
		return fmt.Errorf("path escapes workspace")
	}
	if info, err := os.Lstat(abs); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("refusing to follow symlink")
	}
	return os.WriteFile(abs, data, 0o644)
}

func (h hostFS) List(ctx context.Context, root string) ([]string, error) {
	abs, err := searchRoot(h.ws, root)
	if err != nil {
		return nil, err
	}
	var out []string
	walkErr := filepath.WalkDir(abs, func(path string, d fs.DirEntry, err error) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err != nil {
			return nil // skip unreadable entries rather than aborting the search
		}
		if d.IsDir() {
			if _, skip := skipDir[d.Name()]; skip && path != abs {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil // never follow symlinks out of the workspace
		}
		out = append(out, workspaceRel(h.ws, path))
		if len(out) > 100000 {
			return fmt.Errorf("file listing exceeds 100000 entries; narrow path")
		}
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}
	return filterIgnored(ctx, h, out, root)
}

func (h hostFS) ReadEach(ctx context.Context, rels []string, maxBytes int64, fn func(string, []byte) error) error {
	for _, rel := range rels {
		if err := ctx.Err(); err != nil {
			return err
		}
		abs := filepath.Join(h.ws.Root(), filepath.FromSlash(rel))
		info, err := os.Stat(abs)
		if err != nil || info.IsDir() || info.Size() > maxBytes {
			if err := fn(rel, nil); err != nil {
				return err
			}
			continue
		}
		data, err := os.ReadFile(abs)
		if err != nil {
			if err := fn(rel, nil); err != nil {
				return err
			}
			continue
		}
		if err := fn(rel, data); err != nil {
			return err
		}
	}
	return nil
}

func (h hostFS) ReadRange(ctx context.Context, rel string, offset int64, limit int) ([]byte, string, error) {
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	path, err := h.resolveExisting(rel)
	if err != nil {
		return nil, "", err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return nil, "", err
	}
	if !before.Mode().IsRegular() {
		return nil, "", fmt.Errorf("read range requires a regular file")
	}
	hash := sha256.New()
	buf := make([]byte, 64*1024)
	for {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
		n, err := f.Read(buf)
		if n > 0 {
			_, _ = hash.Write(buf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", err
		}
	}
	if _, err = f.Seek(offset, io.SeekStart); err != nil {
		return nil, "", err
	}
	data, err := io.ReadAll(io.LimitReader(f, int64(limit)))
	if err != nil {
		return nil, "", err
	}
	after, err := f.Stat()
	if err != nil {
		return nil, "", err
	}
	if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return nil, "", fmt.Errorf("file changed while reading; read it again")
	}
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	return data, strings.ToUpper(hex.EncodeToString(hash.Sum(nil)[:fileHashHexLen/2])), err
}

// resolveExisting applies the read-path guard: resolve the relative path, follow
// symlinks, and refuse anything that lands outside the workspace root.
func (h hostFS) resolveExisting(rel string) (string, error) {
	abs, _, err := h.ws.Resolve(rel)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	if !inside(h.ws.Root(), resolved) {
		return "", fmt.Errorf("path escapes workspace")
	}
	return resolved, nil
}

// ---------------------------------------------------------------------------
// sandbox substrate
// ---------------------------------------------------------------------------

// fileWorkdir is where the workspace is bind-mounted for the file tools. It
// matches shellWorkdir so a path the model saw from grep names the same file
// run_shell sees.
const fileWorkdir = shellWorkdir

const (
	// fsSnapshotMaxBytes caps how much file content one ReadEach may pull out of
	// the sandbox in a single exec. A tree walk is deliberately batched into one
	// command, so without this cap a grep over a large workspace would buffer the
	// whole tree in the host process. Past it the caller is told to narrow.
	fsSnapshotMaxBytes = 32 * 1024 * 1024
	// fsTimeoutSeconds bounds a file-tool helper. A tree walk over a large
	// workspace is slower than a one-shot shell command, so it gets more room than
	// the backend's 30s default.
	fsTimeoutSeconds = 60
	// fsExitNotFound / fsExitOverflow are the exit codes the helper scripts use to
	// report a missing path and a snapshot over budget, distinguishing them from
	// an ordinary shell failure.
	fsExitNotFound = 2
	fsExitOverflow = 3
)

// sandboxFS runs the file tools' I/O inside the sandbox, with the workspace
// bind-mounted at fileWorkdir. The guarantees are the container's rather than
// the host guards': a path cannot leave the mount because nothing outside it is
// visible, and a symlink inside the workspace resolves against the container's
// filesystem, not the server's.
//
// Every helper is a POSIX /bin/sh script using only utilities busybox provides
// (find, wc, cat, mkdir, dirname), because the default sandbox image is alpine.
// Paths ride stdin wherever the list can be long, and file content rides stdin
// or stdout as raw bytes — never argv or env, so nothing a tool reads or writes
// is visible to `ps` inside the container.
type sandboxFS struct {
	sb agentcore.Sandbox
	ws *Workspace
}

func (s sandboxFS) Stat(ctx context.Context, rel string) (fsStat, error) {
	if _, _, err := s.ws.Resolve(rel); err != nil {
		return fsStat{}, err
	}
	const script = `p="$1"
if [ -d "$p" ]; then printf 'dir 0\n'
elif [ -f "$p" ]; then printf 'file %s\n' "$(wc -c < "$p")"
else exit 2
fi`
	res, err := s.exec(ctx, agentcore.SandboxExec{Argv: []string{"/bin/sh", "-c", script, "sh", rel}})
	if err != nil {
		return fsStat{}, err
	}
	if res.ExitCode == fsExitNotFound {
		return fsStat{}, fmt.Errorf("%s: no such file or directory", rel)
	}
	if res.ExitCode != 0 {
		return fsStat{}, sandboxCmdErr("stat", res)
	}
	kind, sizeStr, ok := strings.Cut(strings.TrimSpace(res.Stdout), " ")
	if !ok {
		return fsStat{}, fmt.Errorf("%s: unreadable stat output", rel)
	}
	size, _ := strconv.ParseInt(strings.TrimSpace(sizeStr), 10, 64)
	return fsStat{Size: size, IsDir: kind == "dir"}, nil
}

func (s sandboxFS) ReadFile(ctx context.Context, rel string) ([]byte, error) {
	if _, _, err := s.ws.Resolve(rel); err != nil {
		return nil, err
	}
	var out []byte
	found := false
	err := s.ReadEach(ctx, []string{rel}, maxReadWholeFileBytes, func(_ string, data []byte) error {
		if data != nil {
			out, found = data, true
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("%s: no such file or directory", rel)
	}
	return out, nil
}

func (s sandboxFS) WriteFile(ctx context.Context, rel string, data []byte) error {
	if _, _, err := s.ws.Resolve(rel); err != nil {
		return err
	}
	// Content rides stdin, so it never appears in argv or env — the same property
	// the sandboxed HTTP path needs for a resolved credential.
	const script = `p="$1"
mkdir -p "$(dirname "$p")" || exit 1
cat > "$p"`
	res, err := s.exec(ctx, agentcore.SandboxExec{
		Argv:  []string{"/bin/sh", "-c", script, "sh", rel},
		Stdin: string(data),
		// The write path is the one helper that needs WritableFS. It is not about
		// the container's root filesystem (throwaway either way) but about which
		// UID the backend runs as: the locked profile runs as nobody, which cannot
		// write into a workspace directory the host process owns. Isolation is
		// otherwise unchanged — all capabilities dropped, no host env, no network,
		// nothing mounted but the workspace.
		Constraints: agentcore.SandboxLimits{WritableFS: true, TimeoutSeconds: fsTimeoutSeconds},
	})
	if err != nil {
		return err
	}
	if res.ExitCode != 0 {
		return sandboxCmdErr("write", res)
	}
	return nil
}

func (s sandboxFS) List(ctx context.Context, root string) ([]string, error) {
	root = strings.TrimSpace(root)
	if root == "." {
		root = ""
	}
	if root != "" {
		if _, _, err := s.ws.Resolve(root); err != nil {
			return nil, err
		}
	} else {
		root = "."
	}
	// find never prunes the search root itself (its name is always "."), which
	// reproduces the host walk's `skip && path != root` rule: a search explicitly
	// scoped to node_modules still lists it.
	script := `cd "$1" 2>/dev/null || exit 2
find . -type d \( ` + findPruneExpr() + ` \) -prune -o -type f -print`
	res, err := s.exec(ctx, agentcore.SandboxExec{Argv: []string{"/bin/sh", "-c", script, "sh", root}})
	if err != nil {
		return nil, err
	}
	if res.ExitCode == fsExitNotFound {
		return nil, fmt.Errorf("%s: no such directory", root)
	}
	if res.ExitCode != 0 {
		return nil, sandboxCmdErr("list", res)
	}

	prefix := ""
	if root != "." {
		prefix = strings.Trim(filepath.ToSlash(filepath.Clean(root)), "/") + "/"
	}
	var out []string
	for _, line := range strings.Split(res.Stdout, "\n") {
		line = strings.TrimPrefix(strings.TrimRight(line, "\r"), "./")
		if line == "" {
			continue
		}
		out = append(out, prefix+line)
	}
	// find(1) walks in directory order; the host walks in sorted order. Sort with
	// the walk comparator so both substrates hand grep/glob the same sequence.
	sort.Slice(out, func(i, j int) bool { return walkOrderLess(out[i], out[j]) })
	if len(out) > 100000 {
		return nil, fmt.Errorf("file listing exceeds 100000 entries; narrow path")
	}
	return filterIgnored(ctx, s, out, root)
}

func (s sandboxFS) ReadRange(ctx context.Context, rel string, offset int64, limit int) ([]byte, string, error) {
	if _, _, err := s.ws.Resolve(rel); err != nil {
		return nil, "", err
	}
	const script = `p="$1"
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$p" | cut -c1-16
else shasum -a 256 "$p" | cut -c1-16; fi
dd if="$p" bs=1 skip="$2" count="$3" 2>/dev/null`
	res, err := s.exec(ctx, agentcore.SandboxExec{Argv: []string{"/bin/sh", "-c", script, "sh", rel, strconv.FormatInt(offset, 10), strconv.Itoa(limit)}})
	if err != nil {
		return nil, "", err
	}
	if res.ExitCode != 0 {
		return nil, "", sandboxCmdErr("read range", res)
	}
	hash, data, ok := strings.Cut(res.Stdout, "\n")
	if !ok {
		return nil, "", fmt.Errorf("invalid range response")
	}
	if _, err := normalizeFileHash(hash); err != nil {
		return nil, "", err
	}
	return []byte(data), strings.ToUpper(hash), nil
}

func (s sandboxFS) ReadEach(ctx context.Context, rels []string, maxBytes int64, fn func(string, []byte) error) error {
	return s.readEachWithBudget(ctx, rels, maxBytes, fsSnapshotMaxBytes, fn)
}

// readEachWithBudget is ReadEach with the snapshot budget as a parameter, so a
// test can drive the overflow path without materializing 32MB.
func (s sandboxFS) readEachWithBudget(ctx context.Context, rels []string, maxBytes, totalBudget int64, fn func(string, []byte) error) error {
	if len(rels) == 0 {
		return nil
	}
	// The path list rides stdin (no argv length limit, no quoting hazard) and the
	// reply is a length-framed stream: "F <bytes> <path>\n" then exactly that many
	// raw bytes. Framing by byte count rather than a delimiter keeps binary and
	// newline-bearing content intact; a path containing a newline simply fails the
	// -f test and is skipped rather than desyncing the stream. The `tr -d ' '`
	// matters: BSD wc pads its count with leading spaces, busybox wc does not, and
	// the frame header has to be identical either way.
	script := `max_file=$1
max_total=$2
total=0
while IFS= read -r p; do
  if [ ! -f "$p" ] || [ ! -r "$p" ]; then printf 'S %s\n' "$p"; continue; fi
  n=$(wc -c < "$p" 2>/dev/null | tr -d ' ')
  if [ -z "$n" ] || [ "$n" -gt "$max_file" ]; then printf 'S %s\n' "$p"; continue; fi
  total=$((total + n))
  if [ "$total" -gt "$max_total" ]; then exit 3; fi
  printf 'F %s %s\n' "$n" "$p"
  cat "$p"
done`
	res, err := s.exec(ctx, agentcore.SandboxExec{
		Argv: []string{"/bin/sh", "-c", script, "sh",
			strconv.FormatInt(maxBytes, 10), strconv.FormatInt(totalBudget, 10)},
		Stdin: strings.Join(rels, "\n") + "\n",
	})
	if err != nil {
		return err
	}
	if res.ExitCode == fsExitOverflow {
		return fmt.Errorf("more than %dMB of file content — narrow the search with path or glob",
			totalBudget/(1024*1024))
	}
	if res.ExitCode != 0 {
		return sandboxCmdErr("read", res)
	}
	return parseFramedFiles(res.Stdout, fn)
}

// parseFramedFiles decodes the "F <bytes> <path>\n<bytes...>" stream ReadEach's
// helper emits, handing each file to fn. A malformed frame ends the parse rather
// than risking a mis-sliced payload.
func parseFramedFiles(stream string, fn func(string, []byte) error) error {
	for len(stream) > 0 {
		nl := strings.IndexByte(stream, '\n')
		if nl < 0 {
			return nil
		}
		header, rest := stream[:nl], stream[nl+1:]
		if strings.HasPrefix(header, "S ") {
			if err := fn(strings.TrimPrefix(header[2:], "./"), nil); err != nil {
				return err
			}
			stream = rest
			continue
		}
		if !strings.HasPrefix(header, "F ") {
			return nil
		}
		sizeStr, path, ok := strings.Cut(strings.TrimLeft(header[2:], " "), " ")
		if !ok {
			return nil
		}
		size, err := strconv.Atoi(sizeStr)
		if err != nil || size < 0 || size > len(rest) {
			return nil
		}
		if err := fn(strings.TrimPrefix(path, "./"), []byte(rest[:size])); err != nil {
			return err
		}
		stream = rest[size:]
	}
	return nil
}

// findPruneExpr builds the `-name X -o -name Y` clause that prunes skipDir
// directories, sorted so the generated script is stable across runs.
func findPruneExpr() string {
	names := make([]string, 0, len(skipDir))
	for n := range skipDir {
		names = append(names, "-name "+n)
	}
	sort.Strings(names)
	return strings.Join(names, " -o ")
}

// exec runs one helper script in the sandbox with the workspace bind-mounted
// read-write at fileWorkdir, which is also the working directory — so every path
// the helpers handle stays relative and inside the mount. Unless the caller set
// its own, the envelope is fail-closed (no network, read-only root outside the
// mount, default resource caps): a read or a search needs neither egress nor a
// writable system.
func (s sandboxFS) exec(ctx context.Context, req agentcore.SandboxExec) (agentcore.SandboxResult, error) {
	req.Workdir = fileWorkdir
	req.Mounts = []agentcore.SandboxMount{{
		Source:   s.ws.Root(),
		Target:   fileWorkdir,
		ReadOnly: false,
	}}
	if req.Constraints.TimeoutSeconds == 0 {
		req.Constraints.TimeoutSeconds = fsTimeoutSeconds
	}
	return s.sb.Exec(ctx, req)
}

// sandboxCmdErr renders a failed helper into an error that keeps the sandbox's
// own stderr, so a permission problem on the bind mount is diagnosable instead
// of surfacing as a bare non-zero exit.
func sandboxCmdErr(op string, res agentcore.SandboxResult) error {
	if res.Killed {
		return fmt.Errorf("sandbox %s killed: %s", op, res.KillReason)
	}
	if msg := strings.TrimSpace(res.Stderr); msg != "" {
		return fmt.Errorf("sandbox %s failed (exit %d): %s", op, res.ExitCode, msg)
	}
	return fmt.Errorf("sandbox %s failed (exit %d)", op, res.ExitCode)
}
