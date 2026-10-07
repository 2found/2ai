package sandbox

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Workspace guards file tools to one host directory. It accepts only relative
// paths, cleans them, follows the root symlink once, and rejects traversal before
// any filesystem operation happens.
type Workspace struct {
	root         string
	mu           sync.Mutex
	partialReads map[string]string
}

func (w *Workspace) recordRead(path, hash string, partial bool) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !partial {
		delete(w.partialReads, path)
		return nil
	}
	if w.partialReads == nil {
		w.partialReads = map[string]string{}
	}
	if _, exists := w.partialReads[path]; !exists && len(w.partialReads) >= 1024 {
		return fmt.Errorf("partial read tracking is full; complete an earlier read before opening more partial files")
	}
	w.partialReads[path] = hash
	return nil
}
func (w *Workspace) partialRead(path string) string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.partialReads[path]
}

func NewWorkspace(root string) (*Workspace, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("workspace root is empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("workspace root: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(abs, 0o755); err != nil {
				return nil, fmt.Errorf("workspace root: %w", err)
			}
			// Resolve again rather than keeping abs. Root() is the path the shell
			// and browser tools hand Docker as a bind-mount source, so the same
			// workspace resolving to one string the run that created it and
			// another every run after (macOS /var -> /private/var, any symlinked
			// home) would mount two different-looking sources for one directory.
			if resolved, err = filepath.EvalSymlinks(abs); err != nil {
				return nil, fmt.Errorf("workspace root: %w", err)
			}
		} else {
			return nil, fmt.Errorf("workspace root: %w", err)
		}
	}
	return &Workspace{root: filepath.Clean(resolved)}, nil
}

func (w *Workspace) Root() string {
	if w == nil {
		return ""
	}
	return w.root
}

func (w *Workspace) Resolve(rel string) (string, string, error) {
	if w == nil || w.root == "" {
		return "", "", fmt.Errorf("workspace is not configured")
	}
	rel = strings.TrimSpace(rel)
	if rel == "" {
		return "", "", fmt.Errorf("path is empty")
	}
	if filepath.IsAbs(rel) {
		return "", "", fmt.Errorf("path must be relative")
	}
	clean := filepath.Clean(rel)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", "", fmt.Errorf("path escapes workspace")
	}
	abs := filepath.Join(w.root, clean)
	if !inside(w.root, abs) {
		return "", "", fmt.Errorf("path escapes workspace")
	}
	return abs, clean, nil
}

func inside(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, "../")
}
