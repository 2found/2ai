package agentcore

import (
	"context"
	"math"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

// MemoryKind classifies a long-term memory entry.
type MemoryKind string

const (
	MemoryFact       MemoryKind = "fact"
	MemoryLearning   MemoryKind = "learning"
	MemoryOutcome    MemoryKind = "outcome"
	MemoryPreference MemoryKind = "preference"
	MemoryProcedure  MemoryKind = "procedure"
	MemoryEpisode    MemoryKind = "episode"
	MemoryContext    MemoryKind = "context"
)

// MemoryEntry is one durable, distilled fact the agent carries across runs.
// The host owns indexed retrieval, identity, retention and sanitization policy.
type MemoryEntry struct {
	ID         string     `json:"id"`
	ScopeID    string     `json:"scope_id"`
	Kind       MemoryKind `json:"kind"`
	Content    string     `json:"content"`
	Tags       []string   `json:"tags"`
	Confidence float64    `json:"confidence"`
	SourceRun  string     `json:"source_run_id"`
	CreatedAt  time.Time  `json:"created_at"`
	// Key identifies an explicit fact slot; only an authorized replacement may
	// change its value. An empty key never implies semantic equivalence.
	Key         string         `json:"key,omitempty"`
	ExpiresAt   *time.Time     `json:"expires_at,omitempty"`
	ConfirmedAt time.Time      `json:"confirmed_at,omitempty"`
	SeenCount   uint64         `json:"seen_count,omitempty"`
	Evidence    string         `json:"evidence,omitempty"` // stated, inferred, tool, or unknown
	Sources     []MemorySource `json:"sources,omitempty"`
}

// MemorySource is host-owned provenance, not a claim authored by the model.
type MemorySource struct {
	Conversation string `json:"conversation,omitempty"`
	Run          string `json:"run,omitempty"`
	Rollout      string `json:"rollout,omitempty"`
}

// MemoryTarget is a host-authorized alias. It is not a tenant ID supplied by
// a model. The store enforces Writable on every mutation, including curation.
type MemoryTarget struct {
	Name        string `json:"name"`
	ScopeID     string `json:"-"`
	Description string `json:"description"`
	Writable    bool   `json:"writable"`
}

type MemoryRetention struct {
	Target     string     `json:"target,omitempty"`
	Kind       MemoryKind `json:"kind,omitempty"`
	Content    string     `json:"content"`
	Tags       []string   `json:"tags,omitempty"`
	Key        string     `json:"key,omitempty"`
	TTLSeconds int64      `json:"ttl_seconds,omitempty"`
	// ReplaceID is required to change an existing keyed fact, so a model cannot
	// silently overwrite a contradiction without reading the old evidence.
	ReplaceID string `json:"replace_id,omitempty"`
}

// ScopedMemory is optional; existing single-scope stores remain compatible.
type ScopedMemory interface {
	MemoryTargets(context.Context, string) ([]MemoryTarget, error)
	Retain(context.Context, string, MemoryRetention) (MemoryEntry, error)
}

// MemoryReader supplies full entries for safe curation after clipped recall.
type MemoryReader interface {
	ReadMemory(context.Context, string, string) (MemoryEntry, error)
}

// MemoryEraser removes a memory's retained revisions as well as its live row.
// Conversation transcripts have a separate host-owned retention policy.
type MemoryEraser interface {
	EraseMemory(context.Context, string, string) error
}

func ValidMemoryKind(kind MemoryKind) bool {
	switch kind {
	case MemoryFact, MemoryLearning, MemoryOutcome, MemoryPreference, MemoryProcedure, MemoryEpisode, MemoryContext:
		return true
	}
	return false
}

// Session is a working-memory thread: the message history of one run, persisted
// so a chat or long autonomous run can resume or be inspected.
type Session struct {
	ID        string    `json:"id"`
	ScopeID   string    `json:"scope_id"`
	ParentID  string    `json:"parent_id,omitempty"` // set when forked
	Messages  []Message `json:"messages"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// MemoryStore is the long-term recall seam backed by the consumer. Native
// checkpoints and SessionStore own working history. A nil MemoryStore disables
// recall and learning.
type MemoryStore interface {
	// Recall returns long-term entries relevant to the query for a scope.
	Recall(ctx context.Context, scopeID, query string, limit int) ([]MemoryEntry, error)
	// Remember persists an entry under the host's authorization and sanitization policy.
	Remember(ctx context.Context, entry MemoryEntry) error
}

// MemoryCurator is the OPTIONAL write-back half of the memory seam: a store
// that implements it lets the model revise what it previously remembered —
// update a stale fact, or retract one that turned out wrong. It is separate
// from MemoryStore (rather than new methods on it) so a consumer's existing
// store keeps satisfying the seam and simply does not offer the curation
// tool; the memory plugin discovers the capability by type assertion.
//
// Both methods are soft: a retracted row stays in the store and is filtered
// out of recall, so the history of having held the belief survives the
// retraction. There is no hard delete on this seam.
type MemoryCurator interface {
	// Supersede retracts the entry id within scopeID, optionally naming the
	// entry that replaces it ("" retracts without a successor). It must
	// report not-found when no live entry with that id exists in the scope —
	// a silent no-op would tell the model a memory is gone when it is not.
	Supersede(ctx context.Context, scopeID, id, replacementID string) error
	// Update replaces the content of the entry id within scopeID: the new
	// entry is written and the old one retracted to it, atomically, so a
	// failure never leaves a retracted memory with no successor.
	Update(ctx context.Context, scopeID, id string, entry MemoryEntry) error
}

// Embedder turns text into dense vectors for semantic memory recall (§14.7).
// It is the embedding analogue of LLMProvider: a narrow, product-agnostic seam
// the consumer backs with a real vendor (OpenAIEmbedder) or a test fake. A nil
// Embedder means the consumer falls back to keyword recall — embeddings are an
// additive relevance upgrade, never a hard dependency.
type Embedder = protocol.Embedder

// Cosine is the cosine similarity of two equal-length vectors, in [-1, 1].
// Mismatched lengths or a zero-magnitude vector yield 0 (no signal) rather than
// NaN, so ranking degrades gracefully instead of panicking.
func Cosine(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
