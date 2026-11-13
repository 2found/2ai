package sandbox

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// The agent's workspace is ONE directory per conversation, and every tool in
// this package works inside it — file tools, search, shell, computer use,
// browser artifacts, and saved HTTP responses alike.
//
// That single root is the whole point. An agent that can write a script but
// cannot run it, or fetch a file it cannot then read, has two half-capabilities
// instead of one real one. Giving every tool the same root is what makes
// write-then-run, download-then-parse, and screenshot-then-read work without any
// tool knowing about any other.
//
// The layout is
//
//	<base>/<workspaceId>/<projectId>/<agentId>/<conversationId>
//
// which is the ownership chain read left to right: a tenant's workspace, one of
// its projects, one agent in that project, one conversation with that agent.
// Anything wider would let two conversations scribble on each other's files;
// anything narrower would lose the file the user is still talking about between
// turns.

// defaultWorkspaceDirName is the per-user root under $HOME. It mirrors the
// convention of every other tool that keeps per-user state next to the user
// rather than in a system directory, so an operator can find, back up, or delete
// an agent's files without consulting a config file first.
const defaultWorkspaceDirName = ".agentray"

// maxSegmentLen bounds one path component. The ids are UUIDs and slugs today, but
// ConversationID is client-supplied, and a filesystem that rejects a 4KB
// component would surface as a confusing tool error rather than as the bad input
// it is.
const maxSegmentLen = 64

// unnamedSegment stands in for an id the caller does not have. A run with no
// conversation — a schedule firing, a one-shot manual trigger — still needs
// somewhere to put its files, and giving it a stable name means the agent's
// scheduled work accumulates in one predictable place instead of a new directory
// per run that nothing ever cleans up.
const unnamedSegment = "default"

// WorkspaceScope identifies the one conversation a workspace belongs to. Every
// field is optional; a missing one becomes unnamedSegment, so a partially
// identified run still gets a usable, stable directory rather than an error.
type WorkspaceScope struct {
	WorkspaceID    string
	ProjectID      string
	AgentID        string
	ConversationID string

	// Pinned is a folder the user chose, and it replaces the derived path
	// entirely rather than nesting under it.
	//
	// The derived layout is the right default because it keeps conversations from
	// colliding and needs no decision from anyone. But an agent is often pointed
	// at work that already exists — a repository, a directory of documents — and
	// the answer to "which folder?" is then a folder the user already has, not
	// one AgentRay invented. Pinning is that answer: the same directory across
	// every conversation with this agent, which is also what makes the agent's
	// output somewhere the user can find it.
	//
	// It must be an absolute path (a leading ~ is expanded). Unlike the derived
	// segments it is NOT sanitized into a single component — the whole point is
	// that it names a real place — so it is only ever set from an authenticated
	// operator's configuration, never from anything the model can write.
	Pinned string

	// PinDisabled refuses Pinned and falls back to the derived layout.
	//
	// "The user's call" is only the right default when the user setting the path
	// is the person who owns the host. On a hosted, multi-tenant deployment they
	// are not: workspace owner/admin is reachable by self-serve signup, so an
	// unconfined pin would hand a tenant the host filesystem — read via the file
	// tools, and read-write via the bind mount the sandboxed tools take on
	// Workspace.Root(). It is re-checked here rather than only where the value is
	// saved, because a row written before this existed is still in the database.
	PinDisabled bool
}

// DefaultWorkspaceBase is ~/.agentray/workspaces — where agent files live when
// the operator has not said otherwise.
//
// It has a default at all because requiring configuration to get a workspace is
// how a capability quietly ships turned off: with no root configured the file
// tools have nowhere to work, so they are withheld, and the agent a user just
// granted read_file to cannot read a file. A default that works out of the box
// is the difference between "recommended" and "required".
func DefaultWorkspaceBase() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("workspace base: %w", err)
	}
	return filepath.Join(home, defaultWorkspaceDirName, "workspaces"), nil
}

// WorkspaceFor returns the workspace for one conversation, creating it if
// needed. An empty base resolves to DefaultWorkspaceBase, and a pinned folder
// wins over both.
//
// Every id is reduced to a safe single path component before it is joined, so a
// hostile or merely careless ConversationID ("../../etc") cannot place the
// workspace outside the base. The containment is then re-checked on the joined
// path rather than trusted from the sanitizer, because the cost of being wrong
// here is an agent with a writable root anywhere on the host.
func WorkspaceFor(base string, scope WorkspaceScope) (*Workspace, error) {
	if pinned := strings.TrimSpace(scope.Pinned); pinned != "" && !scope.PinDisabled {
		root, err := ResolvePinnedWorkspace(pinned)
		if err != nil {
			return nil, err
		}
		return NewWorkspace(root)
	}
	base = strings.TrimSpace(base)
	if base == "" {
		def, err := DefaultWorkspaceBase()
		if err != nil {
			return nil, err
		}
		base = def
	}
	absBase, err := filepath.Abs(base)
	if err != nil {
		return nil, fmt.Errorf("workspace base: %w", err)
	}
	absBase = filepath.Clean(absBase)

	root := absBase
	for _, seg := range []string{scope.WorkspaceID, scope.ProjectID, scope.AgentID, scope.ConversationID} {
		root = filepath.Join(root, safeSegment(seg))
	}
	// Defence in depth: safeSegment already cannot emit "..", but a workspace
	// rooted outside its base is the one failure that turns a path bug into
	// host-wide write access, so it is verified rather than assumed.
	if !inside(absBase, root) {
		return nil, fmt.Errorf("workspace path escapes %s", absBase)
	}
	return NewWorkspace(root)
}

// ResolvePinnedWorkspace turns a user-chosen folder into the absolute path a
// workspace can be rooted at, or explains why it cannot be one.
//
// It is exported so the setting can be validated where it is *saved* rather than
// only where it is used. A typo'd folder should be a red message under the field
// the user is looking at, not a tool error inside a run three days later that
// reads like the agent broke.
//
// The only path refused outright is the filesystem root. Every other folder is
// the user's call — that is what pinning means — but "/" is never a considered
// choice, and an agent with write tools rooted there is a mistake with no
// undo.
func ResolvePinnedWorkspace(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("workspace folder: empty path")
	}
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("workspace folder: %w", err)
		}
		path = filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(path, "~"), "/"))
	}
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("workspace folder %q must be an absolute path", path)
	}
	path = filepath.Clean(path)
	if path == string(filepath.Separator) || path == filepath.VolumeName(path)+string(filepath.Separator) {
		return "", fmt.Errorf("workspace folder cannot be the filesystem root")
	}
	return path, nil
}

// safeSegment reduces an arbitrary id to one filesystem-safe path component.
//
// It is an allowlist, not a blocklist: every byte outside [A-Za-z0-9._-] becomes
// a dash. A blocklist here would have to anticipate separators, NUL, drive
// letters, and whatever the next filesystem considers special, and would be
// wrong on the one it did not think of. The allowlist is wrong only in being
// occasionally uglier than necessary, which no one reading a directory name will
// mind.
//
// The dot cases are handled after folding, not before, because "..%2f" and its
// relatives fold INTO dots — checking first would pass a segment that only
// becomes "‥" once the unsafe bytes are replaced.
//
// Folding and truncation are both many-to-one, and the layout's whole promise is
// that two conversations cannot share a directory: "a/b" and "a-b" both fold to
// "a-b", and any two ids agreeing on their first maxSegmentLen safe bytes
// truncate to the same thing. So whenever the result is not already the input,
// a hash of the FULL original is appended — the segment stays readable, and
// distinct ids stay distinct.
func safeSegment(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return unnamedSegment
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	out := b.String()
	// "." and ".." are directory references, not names, and a segment of nothing
	// but dots is one of them however long it is.
	if strings.Trim(out, ".") == "" {
		return unnamedSegment
	}
	if out == s && len(out) <= maxSegmentLen {
		return out
	}
	sum := sha256.Sum256([]byte(s))
	suffix := "-" + hex.EncodeToString(sum[:4])
	if len(out) > maxSegmentLen-len(suffix) {
		out = out[:maxSegmentLen-len(suffix)]
	}
	return out + suffix
}
