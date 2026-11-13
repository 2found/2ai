package agentcore

import (
	"context"
	"fmt"
	"slices"
	"sync"
)

// MemorySessionStore is an in-process, append-only SessionStore.
//
// It makes a run resumable within one process — a compacted transcript can
// still be reduced back, and the log invariant has something to check against —
// but it does NOT survive a restart. Use it for tests, for local development,
// and for single-process runs where a crash means the work is gone anyway.
// Anything that must outlive the process needs a durable store.
//
// Safe for concurrent use: a run appends from the loop while a job or a
// session_query reads.
type MemorySessionStore struct {
	mu  sync.Mutex
	log map[string][]SessionEntry

	leaseMu sync.Mutex
	leases  map[string]*memorySessionLease
}

type memorySessionLease struct {
	sem  chan struct{}
	refs int
}

var (
	_ SessionStore      = (*MemorySessionStore)(nil)
	_ SessionBatchStore = (*MemorySessionStore)(nil)
	_ SessionLeaseStore = (*MemorySessionStore)(nil)
)

// NewMemorySessionStore returns an empty in-process session store.
func NewMemorySessionStore() *MemorySessionStore {
	return &MemorySessionStore{log: map[string][]SessionEntry{}, leases: map[string]*memorySessionLease{}}
}

// AcquireSessionLease serializes resumes of one session inside this process.
// It intentionally has no cross-process claim; server deployments use the
// PostgreSQL implementation, while laptops keep a dependency-free fast path.
func (m *MemorySessionStore) AcquireSessionLease(ctx context.Context, id string) (context.Context, func() error, error) {
	m.leaseMu.Lock()
	if m.leases == nil {
		m.leases = map[string]*memorySessionLease{}
	}
	lease := m.leases[id]
	if lease == nil {
		lease = &memorySessionLease{sem: make(chan struct{}, 1)}
		m.leases[id] = lease
	}
	lease.refs++
	m.leaseMu.Unlock()

	select {
	case lease.sem <- struct{}{}:
	case <-ctx.Done():
		m.dropLeaseRef(id, lease)
		return ctx, nil, ctx.Err()
	}

	var once sync.Once
	release := func() error {
		once.Do(func() {
			<-lease.sem
			m.dropLeaseRef(id, lease)
		})
		return nil
	}
	return ctx, release, nil
}

func (m *MemorySessionStore) dropLeaseRef(id string, lease *memorySessionLease) {
	m.leaseMu.Lock()
	defer m.leaseMu.Unlock()
	lease.refs--
	if lease.refs == 0 && m.leases[id] == lease {
		delete(m.leases, id)
	}
}

// Append records one entry, assigning its sequence number.
func (m *MemorySessionStore) Append(ctx context.Context, id string, e SessionEntry) error {
	return m.AppendBatch(ctx, id, []SessionEntry{e})
}

// AppendBatch records one save point atomically. Holding the store lock across
// the whole slice guarantees that a concurrent side record cannot split the
// batch and that readers observe either the state before it or the complete
// state after it.
func (m *MemorySessionStore) AppendBatch(_ context.Context, id string, entries []SessionEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.log == nil {
		m.log = map[string][]SessionEntry{}
	}
	for _, e := range entries {
		e = cloneSessionEntry(e)
		e.Seq = len(m.log[id])
		m.log[id] = append(m.log[id], e)
	}
	return nil
}

// Log returns a copy of the ordered entry log for a session, so a caller
// iterating it cannot be raced by a concurrent append.
func (m *MemorySessionStore) Log(_ context.Context, id string) ([]SessionEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := cloneSessionEntries(m.log[id])
	return out, nil
}

// LogFrom returns the session's entries with Seq >= sinceSeq, in order. The
// in-memory store gains nothing from a windowed read (the slice is already in
// hand), but implementing the capability is what lets the resume path be
// exercised end to end without a database — the alternative is a windowing rule
// that only ever runs in production.
func (m *MemorySessionStore) LogFrom(_ context.Context, id string, sinceSeq int) ([]SessionEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []SessionEntry
	for _, e := range m.log[id] {
		if e.Seq >= sinceSeq {
			out = append(out, cloneSessionEntry(e))
		}
	}
	return out, nil
}

// cloneSessionEntry takes a complete value snapshot of an entry. SessionEntry
// is logically immutable once appended, but several of its fields are pointers
// or slices; copying only the outer struct lets the writer (or a Log caller)
// mutate the store behind its lock. Durable stores get this isolation from
// serialization, so the in-memory backend must provide the same contract.
func cloneSessionEntry(e SessionEntry) SessionEntry {
	out := e
	out.Tools = slices.Clone(e.Tools)
	out.Retained = cloneSessionMessages(e.Retained)
	out.Question = slices.Clone(e.Question)
	if e.Message != nil {
		message := cloneSessionMessage(*e.Message)
		out.Message = &message
	}
	if e.State != nil {
		out.State = e.State.clone()
	}
	if e.Usage != nil {
		usage := *e.Usage
		out.Usage = &usage
	}
	if e.Outcome != nil {
		outcome := *e.Outcome
		outcome.Message = cloneSessionMessage(e.Outcome.Message)
		outcome.Invocations = slices.Clone(e.Outcome.Invocations)
		outcome.Extra = cloneSessionMessages(e.Outcome.Extra)
		out.Outcome = &outcome
	}
	return out
}

func cloneSessionEntries(entries []SessionEntry) []SessionEntry {
	if entries == nil {
		return nil
	}
	out := make([]SessionEntry, len(entries))
	for i, entry := range entries {
		out[i] = cloneSessionEntry(entry)
	}
	return out
}

func cloneSessionMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i, message := range messages {
		out[i] = cloneSessionMessage(message)
	}
	return out
}

func cloneSessionMessage(message Message) Message {
	out := message
	out.ToolCalls = slices.Clone(message.ToolCalls)
	out.ContentParts = slices.Clone(message.ContentParts)
	out.ReasoningBlocks = slices.Clone(message.ReasoningBlocks)
	if message.Usage != nil {
		usage := *message.Usage
		out.Usage = &usage
	}
	return out
}

// CheckpointSeq reports the newest self-contained checkpoint and whether the log
// has ever branched. Both are scans here; a real store answers them with an
// index.
func (m *MemorySessionStore) CheckpointSeq(_ context.Context, id string) (int, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	seq, branched := 0, false
	settled := map[string]bool{}
	var queuedKeys []string
	for _, e := range m.log[id] {
		if e.Kind == EntryLeafMove {
			branched = true
		}
		if e.Kind == EntryCompaction && e.Final && e.Retained != nil && e.State != nil {
			seq = e.Seq
		}
		switch e.Kind {
		case EntryInboxDone:
			settled[e.Target] = true
		case EntryInbox:
			key := e.ID
			if key == "" {
				key = fmt.Sprintf("#%d", e.Seq)
			}
			queuedKeys = append(queuedKeys, key)
		}
	}
	// An unsettled inbox item may sit before the checkpoint; a window starting
	// there would drop it, so report no checkpoint and force the full read.
	for _, k := range queuedKeys {
		if !settled[k] {
			return 0, branched, nil
		}
	}
	return seq, branched, nil
}

// Sessions returns the ids that have at least one entry, in no particular
// order. Useful for a local resume picker.
func (m *MemorySessionStore) Sessions() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, 0, len(m.log))
	for id := range m.log {
		out = append(out, id)
	}
	return out
}
