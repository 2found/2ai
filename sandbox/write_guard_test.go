package sandbox

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

// write_file on both substrates: an existing file is replaced only against the
// snapshot the caller read, atomically, keeping its mode.
func TestWriteFileOverwriteIsHashGuardedAndAtomic(t *testing.T) {
	skipOnWindows(t)
	for _, sub := range []struct {
		name string
		sb   agentcore.Sandbox
	}{{"host", nil}, {"sandbox", NewHostSandbox()}} {
		t.Run(sub.name, func(t *testing.T) {
			ws := mustWorkspace(t)
			write := NewWriteFileTool(sub.sb, ws)
			run := func(args string) (string, error) { return write.Run(context.Background(), args) }
			abs := filepath.Join(ws.Root(), "conf.txt")
			mustWrite(t, ws, "conf.txt", "original")
			if err := os.Chmod(abs, 0o640); err != nil {
				t.Fatal(err)
			}
			unchanged := func(why string) {
				t.Helper()
				if b, _ := os.ReadFile(abs); string(b) != "original" {
					t.Fatalf("%s: file changed to %q", why, b)
				}
			}

			if _, err := run(`{"path":"conf.txt","content":"clobbered"}`); err == nil || !strings.Contains(err.Error(), "expected_hash") {
				t.Fatalf("unread overwrite: err = %v", err)
			}
			unchanged("unread overwrite")
			stale := fileContentHash([]byte("something else"))
			if _, err := run(`{"path":"conf.txt","content":"clobbered","expected_hash":"` + stale + `"}`); err == nil || !strings.Contains(err.Error(), "stale snapshot") {
				t.Fatalf("stale hash: err = %v", err)
			}
			unchanged("stale hash")
			if _, err := run(`{"path":"conf.txt","content":"clobbered","expected_hash":"nothex"}`); err == nil {
				t.Fatal("malformed hash accepted")
			}
			unchanged("malformed hash")

			current := fileContentHash([]byte("original"))
			if _, err := run(`{"path":"conf.txt","content":"replaced","expected_hash":"` + current + `"}`); err != nil {
				t.Fatalf("guarded overwrite: %v", err)
			}
			info, err := os.Stat(abs)
			if err != nil {
				t.Fatal(err)
			}
			if b, _ := os.ReadFile(abs); string(b) != "replaced" {
				t.Fatalf("content = %q", b)
			}
			if info.Mode().Perm() != 0o640 {
				t.Fatalf("mode = %v, want 0640 preserved", info.Mode().Perm())
			}
			entries, _ := os.ReadDir(ws.Root())
			for _, e := range entries {
				if e.Name() != "conf.txt" {
					t.Fatalf("write left %q behind", e.Name())
				}
			}

			// A hash for a file that does not exist is a stale snapshot too.
			if _, err := run(`{"path":"gone.txt","content":"x","expected_hash":"` + current + `"}`); err == nil {
				t.Fatal("expected_hash for a missing file accepted")
			}
			if _, err := os.Stat(filepath.Join(ws.Root(), "gone.txt")); !os.IsNotExist(err) {
				t.Fatalf("gone.txt created: %v", err)
			}
			if err := os.Mkdir(filepath.Join(ws.Root(), "dir"), 0o755); err != nil {
				t.Fatal(err)
			}
			if _, err := run(`{"path":"dir","content":"x"}`); err == nil {
				t.Fatal("writing over a directory accepted")
			}
		})
	}
}

// A symlink is never written through, even one that points inside the
// workspace: replacing it would follow (or silently drop) the link.
func TestWriteFileRefusesSymlinks(t *testing.T) {
	skipOnWindows(t)
	ws := mustWorkspace(t)
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, ws, "real.txt", "real")
	for name, target := range map[string]string{"out.txt": outside, "in.txt": filepath.Join(ws.Root(), "real.txt")} {
		if err := os.Symlink(target, filepath.Join(ws.Root(), name)); err != nil {
			t.Skipf("symlink: %v", err)
		}
		hash := fileContentHash([]byte("real"))
		for _, sb := range []agentcore.Sandbox{nil, NewHostSandbox()} {
			_, err := NewWriteFileTool(sb, ws).Run(context.Background(), fmt.Sprintf(`{"path":%q,"content":"pwned","expected_hash":%q}`, name, hash))
			if err == nil {
				t.Errorf("write through symlink %s accepted (sandbox=%v)", name, sb != nil)
			}
		}
	}
	if b, _ := os.ReadFile(outside); string(b) != "secret" {
		t.Fatalf("outside file changed: %q", b)
	}
	if b, _ := os.ReadFile(filepath.Join(ws.Root(), "real.txt")); string(b) != "real" {
		t.Fatalf("link target changed: %q", b)
	}
}

// os.Root is the confinement, not the earlier path check: even when a
// directory is swapped for an escaping symlink after Resolve approved the path,
// creating parents and writing stay inside the workspace.
func TestAtomicWriteConfinedByRoot(t *testing.T) {
	skipOnWindows(t)
	ws := mustWorkspace(t)
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(ws.Root(), "out")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	root, err := ws.openRoot()
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	for _, name := range []string{"out/new/deep.txt", "out/x.txt"} {
		if err := atomicWrite(root, filepath.FromSlash(name), []byte("x")); err == nil {
			t.Errorf("atomicWrite(%s) escaped the root", name)
		}
	}
	entries, _ := os.ReadDir(outside)
	if len(entries) != 0 {
		t.Fatalf("created %d entries outside the workspace", len(entries))
	}
}

// ReadEach (grep's reader) must not follow a symlink out of the workspace.
func TestHostReadEachDoesNotFollowEscapingSymlinks(t *testing.T) {
	skipOnWindows(t)
	ws := mustWorkspace(t)
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(ws.Root(), "leak.txt")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	mustWrite(t, ws, "ok.txt", "fine")
	got := map[string]string{}
	err := hostFS{ws}.ReadEach(context.Background(), []string{"leak.txt", "ok.txt"}, 1<<20, func(rel string, data []byte) error {
		got[rel] = string(data)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["leak.txt"] != "" || got["ok.txt"] != "fine" {
		t.Fatalf("ReadEach = %q", got)
	}
}
