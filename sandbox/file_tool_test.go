package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileToolsReadWriteWithinWorkspace(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	write := NewWriteFileTool(nil, ws)
	out, err := write.Run(context.Background(), `{"path":"notes/a.txt","content":"hello"}`)
	if err != nil {
		t.Fatalf("write Run: %v", err)
	}
	if !strings.Contains(out, "bytes_written: 5") {
		t.Fatalf("write output = %q", out)
	}
	if !strings.Contains(out, "content_hash: "+fileContentHash([]byte("hello"))) {
		t.Fatalf("write output missing content hash: %q", out)
	}
	read := NewReadFileTool(nil, ws)
	out, err = read.Run(context.Background(), `{"path":"notes/a.txt"}`)
	if err != nil {
		t.Fatalf("read Run: %v", err)
	}
	if !strings.Contains(out, "path: notes/a.txt") ||
		!strings.Contains(out, "content_hash: "+fileContentHash([]byte("hello"))) ||
		!strings.Contains(out, "hello") {
		t.Fatalf("read output = %q", out)
	}
}

func TestFileToolsRejectEscapes(t *testing.T) {
	ws, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	read := NewReadFileTool(nil, ws)
	if _, err := read.Run(context.Background(), `{"path":"../secret"}`); err == nil {
		t.Fatal("expected read escape to fail")
	}
	write := NewWriteFileTool(nil, ws)
	if _, err := write.Run(context.Background(), `{"path":"/tmp/secret","content":"x"}`); err == nil {
		t.Fatal("expected absolute write to fail")
	}
}

func TestReadFileRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatalf("write outside: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "link.txt")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	ws, err := NewWorkspace(root)
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	if _, err := NewReadFileTool(nil, ws).Run(context.Background(), `{"path":"link.txt"}`); err == nil {
		t.Fatal("expected symlink escape to fail")
	}
}
