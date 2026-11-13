package agentcore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

// The durable session log is a tree, not just a line (pi's session format):
// every entry may carry an ID and a ParentID, appending an entry whose parent
// is an *earlier* entry forks a new branch in place, and an EntryLeafMove
// control entry moves the active leaf without appending a node. Reduce/recover
// walk only the active branch, so a rewound session resumes down the new branch
// while the abandoned one stays in the log for inspection — optionally
// distilled into an EntryBranchSummary that rides along as context.
//
// Compatibility: entries without IDs chain implicitly to the entry appended
// before them, so a flat log (or a writer that never branches) is a
// single-branch tree and behaves exactly as before.

// branchSummaryMarker prefixes the system message a reduced session carries for
// an EntryBranchSummary, so the model reads the abandoned branch's distilled
// context as background (parallel to summaryMarker for compaction).
const branchSummaryMarker = "[summary of an abandoned earlier branch of this conversation]"

// newEntryID returns a short random hex id for a session entry (pi's 8-char
// entry id). The store's Seq remains the total order; ids only address nodes.
func newEntryID() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "" // degrade to an implicit-chain entry
	}
	return hex.EncodeToString(b[:])
}

// SessionNode is one resolved node of the session tree: the entry plus its
// effective id/parent. Ids are synthesized ("#<index>") for legacy id-less
// entries, so every node is addressable by Rewind even in logs written before
// ids existed.
type SessionNode struct {
	Entry    SessionEntry
	ID       string
	ParentID string
}

// isSideRecord reports whether a kind is written outside the save-point buffer
// and lives beside the tree rather than in it (see the side-record block in
// session.go). Side records never become the leaf and never parent another
// entry, so a mid-turn write cannot fork the chain or strand a pending entry.
func isSideRecord(k SessionEntryKind) bool {
	switch k {
	case EntryInbox, EntryAssistantFrame, EntryToolProgress, EntryToolOutcome:
		return true
	}
	return false
}

// buildChain resolves every entry's effective id + parent and the active leaf.
// Replay rule: a node's parent is its explicit ParentID when set, else the
// current leaf; every appended node becomes the new leaf (append-is-branch,
// pi's model); an EntryLeafMove moves the leaf without adding a node. Side
// records (inbox, frames, progress, outcomes) are skipped entirely — they are
// not nodes.
func buildChain(log []SessionEntry) (nodes []SessionNode, byID map[string]int, leaf string) {
	byID = make(map[string]int, len(log))
	cur := ""
	for i, e := range log {
		if e.Kind == EntryLeafMove {
			if e.Target != "" {
				cur = e.Target
			}
			continue
		}
		if isSideRecord(e.Kind) {
			continue
		}
		id := e.ID
		if id == "" {
			id = fmt.Sprintf("#%d", i) // stable within the append-only log
		}
		parent := e.ParentID
		if parent == "" {
			parent = cur
		}
		byID[id] = len(nodes)
		nodes = append(nodes, SessionNode{Entry: e, ID: id, ParentID: parent})
		cur = id
	}
	return nodes, byID, cur
}

// SessionTree resolves a log into addressable nodes (log order). Use it to
// render the tree or to pick a Rewind target; ActivePath gives the branch a
// resumed run will actually see.
func SessionTree(log []SessionEntry) []SessionNode {
	nodes, _, _ := buildChain(log)
	return nodes
}

// ActiveLeaf returns the effective id of the entry the next append will chain
// from ("" for an empty log).
func ActiveLeaf(log []SessionEntry) string {
	_, _, leaf := buildChain(log)
	return leaf
}

// pathIndices walks from the node with effective id `from` to the root,
// returning node indices oldest-first. Unknown ids end the walk (treated as
// root), so a dangling explicit ParentID degrades to a shorter path instead of
// an error.
func pathIndices(nodes []SessionNode, byID map[string]int, from string) []int {
	var rev []int
	seen := map[string]bool{}
	for id := from; id != "" && !seen[id]; {
		seen[id] = true
		idx, ok := byID[id]
		if !ok {
			break
		}
		rev = append(rev, idx)
		id = nodes[idx].ParentID
	}
	out := make([]int, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		out = append(out, rev[i])
	}
	return out
}

// ActivePath returns the entries on the active branch, oldest first: the chain
// from the root to the active leaf. A flat (id-less, never-rewound) log returns
// every entry in order — the pre-tree behavior.
func ActivePath(log []SessionEntry) []SessionEntry {
	if path, ok := linearActivePath(log); ok {
		return path
	}
	nodes, byID, leaf := buildChain(log)
	idxs := pathIndices(nodes, byID, leaf)
	out := make([]SessionEntry, 0, len(idxs))
	for _, i := range idxs {
		out = append(out, nodes[i].Entry)
	}
	return out
}

// linearActivePath avoids materializing the full tree for the overwhelmingly
// common case: an append-only session that has never forked or rewound. The
// general buildChain path needs a SessionNode copy, an id->node index, a
// leaf-to-root walk, and a second SessionEntry copy. On a long server session
// that is substantial resume-time allocation for data already in log order.
//
// We still validate parent continuity and effective-id uniqueness here. Any
// malformed or genuinely branched shape falls back to the authoritative tree
// algorithm, preserving its degradation behavior instead of guessing.
func linearActivePath(log []SessionEntry) ([]SessionEntry, bool) {
	out := make([]SessionEntry, 0, len(log))
	seen := make(map[string]struct{}, len(log))
	leaf := ""
	for i, entry := range log {
		if entry.Kind == EntryLeafMove {
			if entry.Target != "" {
				return nil, false
			}
			continue
		}
		if isSideRecord(entry.Kind) {
			continue
		}
		id := entry.ID
		if id == "" {
			id = fmt.Sprintf("#%d", i)
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, false
		}
		seen[id] = struct{}{}
		parent := entry.ParentID
		if parent == "" {
			parent = leaf
		}
		if parent != leaf {
			return nil, false
		}
		out = append(out, entry)
		leaf = id
	}
	return out, true
}

// commonAncestor returns the effective id of the deepest node shared by the
// branches ending at a and b ("" when they share nothing).
func commonAncestor(nodes []SessionNode, byID map[string]int, a, b string) string {
	onA := map[string]bool{}
	for _, i := range pathIndices(nodes, byID, a) {
		onA[nodes[i].ID] = true
	}
	// Walk b leaf → root; the first node also on a's path is the deepest shared.
	seen := map[string]bool{}
	for id := b; id != "" && !seen[id]; {
		seen[id] = true
		idx, ok := byID[id]
		if !ok {
			return ""
		}
		if onA[id] {
			return id
		}
		id = nodes[idx].ParentID
	}
	return ""
}

// spanMessages collects the conversation messages introduced on the branch
// ending at leaf, EXCLUDING everything at or above ancestor — i.e. the work
// that will be abandoned by a rewind to ancestor. Oldest first.
func spanMessages(nodes []SessionNode, byID map[string]int, leaf, ancestor string) []Message {
	shared := map[string]bool{}
	if ancestor != "" {
		for _, i := range pathIndices(nodes, byID, ancestor) {
			shared[nodes[i].ID] = true
		}
	}
	var out []Message
	for _, i := range pathIndices(nodes, byID, leaf) {
		n := nodes[i]
		if shared[n.ID] {
			continue
		}
		if n.Entry.Kind == EntryMessage && n.Entry.Message != nil {
			out = append(out, *n.Entry.Message)
		}
	}
	return out
}

// BranchOptions configure Rewind.
type BranchOptions struct {
	// Summarize, when non-nil, distills the messages of the abandoned span (old
	// leaf back to the common ancestor with the target, exclusive) into an
	// EntryBranchSummary appended to the new branch, so context from the
	// abandoned work survives the switch (pi's branch summarization on /tree).
	// A summarizer error or empty summary degrades to a bare leaf move — a
	// rewind never fails on a flaky summarizer. The returned Usage is the
	// summarization call's own spend; Rewind stamps it onto the branch-summary
	// entry so this real provider call is never invisible spend.
	Summarize func(ctx context.Context, abandoned []Message) (string, Usage, error)
}

// Rewind moves a session's active leaf to targetID (an effective id from
// SessionTree — an entry's own ID, or "#<index>" for legacy id-less entries).
// Subsequent appends chain from there, and ReduceSession/RecoverSession
// reconstruct history along the new branch. It returns the new leaf id — the
// target itself, or the branch-summary node when one was written in front of
// it. The abandoned branch stays in the log untouched.
func Rewind(ctx context.Context, store SessionStore, sessionID, targetID string, opts BranchOptions) (string, error) {
	log, err := store.Log(ctx, sessionID)
	if err != nil {
		return "", err
	}
	nodes, byID, leaf := buildChain(log)
	if _, ok := byID[targetID]; !ok {
		return "", fmt.Errorf("agentcore: rewind target %q is not in session %s", targetID, sessionID)
	}
	newLeaf := targetID
	if opts.Summarize != nil && leaf != "" && leaf != targetID {
		anc := commonAncestor(nodes, byID, leaf, targetID)
		if abandoned := spanMessages(nodes, byID, leaf, anc); len(abandoned) > 0 {
			if summary, su, serr := opts.Summarize(ctx, abandoned); serr == nil && strings.TrimSpace(summary) != "" {
				bs := SessionEntry{
					Kind:      EntryBranchSummary,
					ID:        newEntryID(),
					ParentID:  targetID,
					Summary:   strings.TrimSpace(summary),
					CreatedAt: time.Now(),
				}
				if su != (Usage{}) {
					bs.Usage = &su
				}
				if bs.ID != "" {
					if err := store.Append(ctx, sessionID, bs); err != nil {
						return "", err
					}
					newLeaf = bs.ID
				}
			}
		}
	}
	// The explicit leaf move makes the rewind durable even when no summary node
	// was appended (and keeps the intent legible in the log either way).
	if err := store.Append(ctx, sessionID, SessionEntry{Kind: EntryLeafMove, Target: newLeaf, CreatedAt: time.Now()}); err != nil {
		return "", err
	}
	return newLeaf, nil
}
