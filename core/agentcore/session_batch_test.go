package agentcore

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestMemorySessionStoreConformance(t *testing.T) {
	runSessionStoreConformance(t, func() SessionStore { return NewMemorySessionStore() })
}

func TestMemorySessionStoreSnapshotsRichContentParts(t *testing.T) {
	store := NewMemorySessionStore()
	ctx := context.Background()
	msg := Message{Role: RoleTool, ContentParts: []ContentPart{{Type: ContentPartImage, Data: "original"}}}
	if err := store.Append(ctx, "rich", SessionEntry{Kind: EntryMessage, Message: &msg}); err != nil {
		t.Fatal(err)
	}
	msg.ContentParts[0].Data = "writer mutation"
	first, err := store.Log(ctx, "rich")
	if err != nil {
		t.Fatal(err)
	}
	if got := first[0].Message.ContentParts[0].Data; got != "original" {
		t.Fatalf("append did not snapshot content parts: %q", got)
	}
	first[0].Message.ContentParts[0].Data = "reader mutation"
	second, err := store.Log(ctx, "rich")
	if err != nil {
		t.Fatal(err)
	}
	if got := second[0].Message.ContentParts[0].Data; got != "original" {
		t.Fatalf("log did not clone content parts: %q", got)
	}
}

func TestMemorySessionStoreSnapshotsReasoningBlocks(t *testing.T) {
	store := NewMemorySessionStore()
	ctx := context.Background()
	msg := Message{Role: RoleAssistant, ReasoningBlocks: []ReasoningBlock{{
		Type: ReasoningBlockThinking, Text: "original", Signature: "sig", ReplayScope: "anthropic:scope",
	}}}
	if err := store.Append(ctx, "reasoning", SessionEntry{Kind: EntryMessage, Message: &msg}); err != nil {
		t.Fatal(err)
	}
	msg.ReasoningBlocks[0].Text = "writer mutation"
	first, err := store.Log(ctx, "reasoning")
	if err != nil {
		t.Fatal(err)
	}
	if got := first[0].Message.ReasoningBlocks[0].Text; got != "original" {
		t.Fatalf("append did not snapshot reasoning blocks: %q", got)
	}
	first[0].Message.ReasoningBlocks[0].Text = "reader mutation"
	second, err := store.Log(ctx, "reasoning")
	if err != nil {
		t.Fatal(err)
	}
	if got := second[0].Message.ReasoningBlocks[0].Text; got != "original" {
		t.Fatalf("log did not clone reasoning blocks: %q", got)
	}
}

// runSessionStoreConformance is the backend contract in executable form. A new
// in-package backend gets the same ordering/concurrency checks by adding one
// call with its factory; optional capabilities are tested when implemented.
func runSessionStoreConformance(t *testing.T, newStore func() SessionStore) {
	t.Helper()

	t.Run("ordered append", func(t *testing.T) {
		store := newStore()
		ctx := context.Background()
		for i := 0; i < 3; i++ {
			if err := store.Append(ctx, "ordered", SessionEntry{Kind: EntryMessage, Turn: i}); err != nil {
				t.Fatalf("append %d: %v", i, err)
			}
		}
		log, err := store.Log(ctx, "ordered")
		if err != nil {
			t.Fatal(err)
		}
		assertConsecutiveSessionLog(t, log, 3)
		for i, entry := range log {
			if entry.Turn != i {
				t.Fatalf("entry %d has turn %d", i, entry.Turn)
			}
		}
	})

	t.Run("concurrent append", func(t *testing.T) {
		store := newStore()
		ctx := context.Background()
		const writers, perWriter = 8, 40
		var wg sync.WaitGroup
		for writer := 0; writer < writers; writer++ {
			wg.Add(1)
			go func(writer int) {
				defer wg.Done()
				for i := 0; i < perWriter; i++ {
					msg := Message{Role: RoleUser, Content: fmt.Sprintf("%d/%d", writer, i)}
					if err := store.Append(ctx, "concurrent", SessionEntry{Kind: EntryMessage, Message: &msg}); err != nil {
						t.Errorf("writer %d append %d: %v", writer, i, err)
						return
					}
				}
			}(writer)
		}
		wg.Wait()
		log, err := store.Log(ctx, "concurrent")
		if err != nil {
			t.Fatal(err)
		}
		assertConsecutiveSessionLog(t, log, writers*perWriter)
	})

	t.Run("batch remains contiguous", func(t *testing.T) {
		store := newStore()
		batches, ok := store.(SessionBatchStore)
		if !ok {
			t.Skip("backend does not implement SessionBatchStore")
		}
		ctx := context.Background()
		const batchCount, batchSize = 12, 5
		var wg sync.WaitGroup
		for batch := 0; batch < batchCount; batch++ {
			wg.Add(1)
			go func(batch int) {
				defer wg.Done()
				entries := make([]SessionEntry, batchSize)
				for i := range entries {
					entries[i] = SessionEntry{Kind: EntryMessage, ID: fmt.Sprintf("batch-%d-%d", batch, i)}
				}
				if err := batches.AppendBatch(ctx, "batches", entries); err != nil {
					t.Errorf("batch %d: %v", batch, err)
				}
			}(batch)
		}
		// Single-entry side writes race the batches. They may land between save
		// points, but never inside one.
		for side := 0; side < batchCount; side++ {
			wg.Add(1)
			go func(side int) {
				defer wg.Done()
				if err := store.Append(ctx, "batches", SessionEntry{Kind: EntryToolProgress, ID: fmt.Sprintf("side-%d", side)}); err != nil {
					t.Errorf("side %d: %v", side, err)
				}
			}(side)
		}
		wg.Wait()
		log, err := store.Log(ctx, "batches")
		if err != nil {
			t.Fatal(err)
		}
		assertConsecutiveSessionLog(t, log, batchCount*(batchSize+1))
		positions := make(map[string]int, len(log))
		for i, entry := range log {
			positions[entry.ID] = i
		}
		for batch := 0; batch < batchCount; batch++ {
			start := positions[fmt.Sprintf("batch-%d-0", batch)]
			for i := 1; i < batchSize; i++ {
				if got := positions[fmt.Sprintf("batch-%d-%d", batch, i)]; got != start+i {
					t.Fatalf("batch %d split: item %d at %d, want %d", batch, i, got, start+i)
				}
			}
		}
	})
}

func assertConsecutiveSessionLog(t *testing.T, log []SessionEntry, want int) {
	t.Helper()
	if len(log) != want {
		t.Fatalf("log length = %d, want %d", len(log), want)
	}
	for i := 1; i < len(log); i++ {
		if log[i].Seq != log[i-1].Seq+1 {
			t.Fatalf("sequence gap at %d: %d after %d", i, log[i].Seq, log[i-1].Seq)
		}
	}
}

type batchContractProbe struct {
	batchErr   error
	appendCall int
	batchCall  int
}

func (p *batchContractProbe) Append(context.Context, string, SessionEntry) error {
	p.appendCall++
	return nil
}
func (p *batchContractProbe) AppendBatch(context.Context, string, []SessionEntry) error {
	p.batchCall++
	return p.batchErr
}
func (*batchContractProbe) Log(context.Context, string) ([]SessionEntry, error) { return nil, nil }

func TestAppendSessionBatchPrefersAtomicCapability(t *testing.T) {
	wantErr := errors.New("transaction rolled back")
	store := &batchContractProbe{batchErr: wantErr}
	n, err := appendSessionBatch(context.Background(), store, "s", []SessionEntry{{}, {}})
	if !errors.Is(err, wantErr) || n != 0 {
		t.Fatalf("appendSessionBatch = (%d, %v), want (0, %v)", n, err, wantErr)
	}
	if store.batchCall != 1 || store.appendCall != 0 {
		t.Fatalf("calls: batch=%d append=%d, want 1/0", store.batchCall, store.appendCall)
	}
}

type prefixFailureStore struct{ calls int }

func (p *prefixFailureStore) Append(context.Context, string, SessionEntry) error {
	p.calls++
	if p.calls == 2 {
		return errors.New("write failed")
	}
	return nil
}
func (*prefixFailureStore) Log(context.Context, string) ([]SessionEntry, error) { return nil, nil }

func TestAppendSessionBatchLegacyStoreReportsDurablePrefix(t *testing.T) {
	store := &prefixFailureStore{}
	n, err := appendSessionBatch(context.Background(), store, "s", []SessionEntry{{}, {}, {}})
	if err == nil || n != 1 {
		t.Fatalf("appendSessionBatch = (%d, %v), want prefix 1 and error", n, err)
	}
	if store.calls != 2 {
		t.Fatalf("Append called %d times, want 2", store.calls)
	}
}
