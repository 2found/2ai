package agentcore

import (
	"context"
	"fmt"
	"testing"
)

func BenchmarkMemorySessionStoreAppendBatch(b *testing.B) {
	ctx := context.Background()
	entries := benchmarkSessionEntries(64)
	b.ReportAllocs()
	b.SetBytes(int64(benchmarkSessionPayloadBytes(entries)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := NewMemorySessionStore()
		if err := store.AppendBatch(ctx, fmt.Sprintf("session-%d", i), entries); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSessionReduce(b *testing.B) {
	for _, size := range []int{100, 1_000, 10_000} {
		b.Run(fmt.Sprintf("entries_%d", size), func(b *testing.B) {
			log := benchmarkSessionEntries(size)
			b.ReportAllocs()
			b.SetBytes(int64(benchmarkSessionPayloadBytes(log)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				state := ReduceSession(log)
				if len(state.Messages) != size {
					b.Fatalf("reduced %d messages, want %d", len(state.Messages), size)
				}
			}
		})
	}
}

func BenchmarkMemorySessionStoreWindowRead(b *testing.B) {
	ctx := context.Background()
	store := NewMemorySessionStore()
	const oldEntries = 10_000
	if err := store.AppendBatch(ctx, "window", benchmarkSessionEntries(oldEntries)); err != nil {
		b.Fatal(err)
	}
	checkpoint := SessionEntry{
		Kind: EntryCompaction, Final: true,
		Retained: []Message{{Role: RoleSystem, Content: "summary"}},
		State:    &CheckpointState{Model: "model", ActiveTools: []string{"read"}},
	}
	if err := store.Append(ctx, "window", checkpoint); err != nil {
		b.Fatal(err)
	}
	if err := store.AppendBatch(ctx, "window", benchmarkSessionEntries(32)); err != nil {
		b.Fatal(err)
	}
	seq, branched, err := store.CheckpointSeq(ctx, "window")
	if err != nil || branched || seq == 0 {
		b.Fatalf("checkpoint = (%d, %v, %v)", seq, branched, err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		window, err := store.LogFrom(ctx, "window", seq)
		if err != nil {
			b.Fatal(err)
		}
		if len(window) != 33 {
			b.Fatalf("window entries = %d, want 33", len(window))
		}
	}
}

func benchmarkSessionEntries(n int) []SessionEntry {
	entries := make([]SessionEntry, n)
	for i := range entries {
		message := Message{Role: RoleUser, Content: fmt.Sprintf("message-%05d: benchmark payload for durable session reduction", i)}
		id := fmt.Sprintf("entry-%08d", i)
		parent := ""
		if i > 0 {
			parent = fmt.Sprintf("entry-%08d", i-1)
		}
		entries[i] = SessionEntry{Seq: i, Kind: EntryMessage, ID: id, ParentID: parent, Turn: i + 1, Message: &message}
	}
	return entries
}

func benchmarkSessionPayloadBytes(entries []SessionEntry) int {
	total := 0
	for _, entry := range entries {
		if entry.Message != nil {
			total += len(entry.Message.Content)
		}
	}
	return total
}
