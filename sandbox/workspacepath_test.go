package sandbox

import (
	"path/filepath"
	"strings"
	"testing"
)

// resolvedTempDir is t.TempDir() with symlinks resolved. Workspace.Root() is
// always the resolved path (it is a Docker bind-mount source, so it has to be
// stable), and on macOS t.TempDir() hands back /var/... for /private/var/... —
// so comparing against the raw temp dir would fail on the symlink, not on
// anything this package did.
func resolvedTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp dir: %v", err)
	}
	return dir
}

// The workspace path is built from ids the server holds and one — the
// conversation id — the client supplies. That makes it a path-traversal surface
// wearing the clothes of a naming convention, and the consequence of getting it
// wrong is not a wrong directory: every tool in this package writes into this
// root, so a root placed outside the base is an agent with write access wherever
// it pointed.

func TestWorkspaceLayoutIsTheOwnershipChain(t *testing.T) {
	base := resolvedTempDir(t)
	ws, err := WorkspaceFor(base, WorkspaceScope{
		WorkspaceID:    "ws1",
		ProjectID:      "proj1",
		AgentID:        "agent1",
		ConversationID: "conv1",
	})
	if err != nil {
		t.Fatalf("WorkspaceFor: %v", err)
	}
	want := filepath.Join(base, "ws1", "proj1", "agent1", "conv1")
	if got := ws.Root(); got != want {
		t.Fatalf("workspace root\n got %q\nwant %q", got, want)
	}
	// The directory is real: a tool that resolves a path into a root nothing
	// created fails on first write, which is a confusing way to learn the
	// workspace was never set up.
	if _, _, err := ws.Resolve("notes.md"); err != nil {
		t.Fatalf("Resolve inside a fresh workspace: %v", err)
	}
}

// TestAHostileConversationIDCannotEscape is the one that matters. The
// conversation id is client-held, so it is attacker-controlled in exactly the
// same sense a URL path segment is.
func TestAHostileConversationIDCannotEscape(t *testing.T) {
	base := resolvedTempDir(t)
	for _, id := range []string{
		"../../../../etc",
		"..",
		".",
		"....",
		"a/../../b",
		"/absolute",
		"C:\\windows",
		"conv\x00null",
	} {
		ws, err := WorkspaceFor(base, WorkspaceScope{
			WorkspaceID: "ws", ProjectID: "p", AgentID: "a", ConversationID: id,
		})
		if err != nil {
			// Refusing is a fine answer; escaping is not. Either way, check the
			// next one.
			continue
		}
		root := ws.Root()
		rel, rerr := filepath.Rel(base, root)
		if rerr != nil || rel == ".." || strings.HasPrefix(rel, "..") {
			t.Fatalf("conversation id %q produced a workspace outside the base:\n base %q\n root %q", id, base, root)
		}
		// And it must still be four levels down — an id that collapsed a level
		// would silently share a directory with another conversation.
		if depth := len(strings.Split(filepath.ToSlash(rel), "/")); depth != 4 {
			t.Fatalf("conversation id %q produced a %d-level path %q, want 4", id, depth, rel)
		}
	}
}

// TestAMissingIDStillGetsAStableDirectory: a scheduled run has no conversation.
// It still has to put files somewhere, and somewhere STABLE — a fresh directory
// per run would accumulate forever with nothing to clean it up.
func TestAMissingIDStillGetsAStableDirectory(t *testing.T) {
	base := resolvedTempDir(t)
	scope := WorkspaceScope{WorkspaceID: "ws", ProjectID: "p", AgentID: "a"}

	first, err := WorkspaceFor(base, scope)
	if err != nil {
		t.Fatalf("WorkspaceFor: %v", err)
	}
	second, err := WorkspaceFor(base, scope)
	if err != nil {
		t.Fatalf("WorkspaceFor (again): %v", err)
	}
	if first.Root() != second.Root() {
		t.Fatalf("two runs of the same unnamed scope got different roots:\n %q\n %q", first.Root(), second.Root())
	}
	if !strings.HasSuffix(first.Root(), unnamedSegment) {
		t.Fatalf("an absent conversation id should resolve to %q, got %q", unnamedSegment, first.Root())
	}
}

// TestTwoConversationsDoNotShareARoot is the isolation the layout exists for.
func TestTwoConversationsDoNotShareARoot(t *testing.T) {
	base := resolvedTempDir(t)
	mk := func(conv string) string {
		ws, err := WorkspaceFor(base, WorkspaceScope{
			WorkspaceID: "ws", ProjectID: "p", AgentID: "a", ConversationID: conv,
		})
		if err != nil {
			t.Fatalf("WorkspaceFor(%q): %v", conv, err)
		}
		return ws.Root()
	}
	if a, b := mk("conv-a"), mk("conv-b"); a == b {
		t.Fatalf("two conversations share a workspace root: %q", a)
	}
}

// TestSafeSegmentFoldsRatherThanRejects keeps the sanitizer honest about the
// order it works in: unsafe bytes fold to dashes FIRST, so a segment that only
// becomes a dot-reference after folding is still caught.
func TestSafeSegmentFoldsRatherThanRejects(t *testing.T) {
	// An id that is already safe and short passes through untouched — the common
	// case, and what keeps a directory listing readable.
	for in, want := range map[string]string{
		"":          unnamedSegment,
		"   ":       unnamedSegment,
		".":         unnamedSegment,
		"..":        unnamedSegment,
		"...":       unnamedSegment,
		"ok-id_1.2": "ok-id_1.2",
	} {
		if got := safeSegment(in); got != want {
			t.Fatalf("safeSegment(%q) = %q, want %q", in, got, want)
		}
	}
	// Anything that had to be changed keeps a readable prefix and gains a hash of
	// the original, so the fold stays legible without becoming many-to-one.
	for _, in := range []string{"a/b", "a\\b", "café", strings.Repeat("x", 200)} {
		got := safeSegment(in)
		if len(got) > maxSegmentLen {
			t.Errorf("safeSegment(%q) = %q, longer than %d", in, got, maxSegmentLen)
		}
		if strings.ContainsAny(got, `/\`) {
			t.Errorf("safeSegment(%q) = %q, still contains a separator", in, got)
		}
	}
}

// TestSafeSegmentIsInjective is the other half of the isolation promise: folding
// and truncation are both many-to-one, so without a disambiguator two distinct
// conversation ids would land in one directory and overwrite each other's files.
// ConversationID is client-supplied, which makes the collision reachable rather
// than theoretical.
func TestSafeSegmentIsInjective(t *testing.T) {
	long := strings.Repeat("c", maxSegmentLen)
	for _, pair := range [][2]string{
		{"a/b", "a-b"},             // different ids, same fold
		{"conv:1", "conv-1"},       // the shape a client actually produces
		{long + "-1", long + "-2"}, // agree on the first maxSegmentLen bytes
	} {
		if a, b := safeSegment(pair[0]), safeSegment(pair[1]); a == b {
			t.Errorf("safeSegment(%q) and safeSegment(%q) both = %q", pair[0], pair[1], a)
		}
	}
}

func TestDefaultWorkspaceBaseIsUnderTheUsersHome(t *testing.T) {
	base, err := DefaultWorkspaceBase()
	if err != nil {
		t.Fatalf("DefaultWorkspaceBase: %v", err)
	}
	if !filepath.IsAbs(base) {
		t.Fatalf("workspace base must be absolute, got %q", base)
	}
	if !strings.HasSuffix(filepath.ToSlash(base), "/"+defaultWorkspaceDirName+"/workspaces") {
		t.Fatalf("workspace base %q is not ~/%s/workspaces", base, defaultWorkspaceDirName)
	}
}

// A pinned folder is the user answering "which folder?" with one they already
// have. The tests below fix the two things that answer has to mean: it replaces
// the derived path instead of nesting under it, and it survives the conversation
// that created it — otherwise "select a folder" would just be a slower way of
// getting a scratch directory.

func TestAPinnedFolderReplacesTheDerivedPath(t *testing.T) {
	base := resolvedTempDir(t)
	pinned := filepath.Join(resolvedTempDir(t), "my-project")

	ws, err := WorkspaceFor(base, WorkspaceScope{
		WorkspaceID:    "ws1",
		ProjectID:      "proj1",
		AgentID:        "agent1",
		ConversationID: "conv1",
		Pinned:         pinned,
	})
	if err != nil {
		t.Fatalf("WorkspaceFor: %v", err)
	}
	if ws.Root() != pinned {
		t.Fatalf("pinned workspace root = %q, want %q", ws.Root(), pinned)
	}
	if strings.Contains(ws.Root(), base) {
		t.Fatalf("a pinned folder must not be nested under the derived base: %q", ws.Root())
	}
}

func TestAPinnedFolderIsTheSameAcrossConversations(t *testing.T) {
	base := resolvedTempDir(t)
	pinned := filepath.Join(resolvedTempDir(t), "shared")

	var roots []string
	for _, conv := range []string{"conv-a", "conv-b"} {
		ws, err := WorkspaceFor(base, WorkspaceScope{AgentID: "agent1", ConversationID: conv, Pinned: pinned})
		if err != nil {
			t.Fatalf("WorkspaceFor(%s): %v", conv, err)
		}
		roots = append(roots, ws.Root())
	}
	if roots[0] != roots[1] {
		t.Fatalf("a pinned folder must not change per conversation: %q vs %q", roots[0], roots[1])
	}
}

func TestResolvePinnedWorkspaceRejectsWhatCannotBeAWorkspace(t *testing.T) {
	for _, bad := range []string{"", "   ", "relative/path", "./here", string(filepath.Separator)} {
		if got, err := ResolvePinnedWorkspace(bad); err == nil {
			t.Errorf("ResolvePinnedWorkspace(%q) = %q, want an error", bad, got)
		}
	}
}

func TestResolvePinnedWorkspaceExpandsHome(t *testing.T) {
	got, err := ResolvePinnedWorkspace("~/agent-files")
	if err != nil {
		t.Fatalf("ResolvePinnedWorkspace: %v", err)
	}
	if strings.Contains(got, "~") || !filepath.IsAbs(got) {
		t.Fatalf("~ was not expanded to an absolute path: %q", got)
	}
}

// TestPinningCanBeDisabled covers the hosted deployment: the pinned column is
// still populated (a row written on a self-host, or before this existed), and
// the run must fall back to the derived layout rather than rooting the agent
// wherever that value points.
func TestPinningCanBeDisabled(t *testing.T) {
	base := resolvedTempDir(t)
	pinned := filepath.Join(resolvedTempDir(t), "somebody-elses-folder")

	ws, err := WorkspaceFor(base, WorkspaceScope{
		WorkspaceID: "ws", ProjectID: "p", AgentID: "a", ConversationID: "c",
		Pinned:      pinned,
		PinDisabled: true,
	})
	if err != nil {
		t.Fatalf("WorkspaceFor: %v", err)
	}
	if want := filepath.Join(base, "ws", "p", "a", "c"); ws.Root() != want {
		t.Fatalf("a disabled pin still escaped the base:\n got %q\nwant %q", ws.Root(), want)
	}
}
