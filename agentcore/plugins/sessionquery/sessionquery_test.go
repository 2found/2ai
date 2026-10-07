package sessionquery

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

// seedLog builds an in-memory session store holding the given messages.
func seedLog(t *testing.T, sessionID string, msgs ...agentcore.Message) agentcore.SessionStore {
	t.Helper()
	store := agentcore.NewMemorySessionStore()
	for i, m := range msgs {
		m := m
		err := store.Append(context.Background(), sessionID, agentcore.SessionEntry{
			Kind:      agentcore.EntryMessage,
			Turn:      i + 1,
			Message:   &m,
			CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
		})
		if err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	return store
}

// beginQuery starts the plugin for a run the way the loop would. A nil result
// means the plugin DECLINED the run.
func beginQuery(t *testing.T, p Plugin, store agentcore.SessionStore, sessionID string) *queryRun {
	t.Helper()
	ext, err := p.BeginRun(context.Background(), agentcore.RunInfo{
		SessionID: sessionID,
		Session:   store,
		Durable:   store != nil && sessionID != "",
	})
	if err != nil {
		t.Fatalf("BeginRun: %v", err)
	}
	if ext == nil {
		return nil
	}
	return ext.(*queryRun)
}

func queryPolicy(t *testing.T, store agentcore.SessionStore, sessionID string) *queryRun {
	t.Helper()
	p := beginQuery(t, Plugin{}, store, sessionID)
	if p == nil {
		t.Fatal("expected a session-query policy")
	}
	return p
}

func TestSessionQuery_FindsCompactedAwayDetail(t *testing.T) {
	store := seedLog(t, "s1",
		agentcore.Message{Role: agentcore.RoleUser, Content: "how many novels shipped in July?"},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "The July total was 1,284 novels across 12 sources."},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "Unrelated chatter about the weather."},
	)
	p := queryPolicy(t, store, "s1")
	out, err := (&sessionQueryTool{policy: p}).Run(context.Background(), `{"query":"july total"}`)
	if err != nil {
		t.Fatalf("session_query: %v", err)
	}
	if !strings.Contains(out, "1,284") {
		t.Fatalf("expected the recovered figure, got %q", out)
	}
	if strings.Contains(out, "weather") {
		t.Fatalf("unrelated entry matched: %q", out)
	}
}

func TestSessionQuery_AllTokensMustMatch(t *testing.T) {
	store := seedLog(t, "s1",
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "revenue was flat"},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "churn was up"},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "revenue and churn both moved"},
	)
	p := queryPolicy(t, store, "s1")
	res, err := p.provider.Search(context.Background(), SessionQueryRequest{SessionID: "s1", Query: "revenue churn"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Matches) != 1 {
		t.Fatalf("token-AND should match exactly one entry, got %d", len(res.Matches))
	}
	if !strings.Contains(res.Matches[0].Excerpt, "both moved") {
		t.Fatalf("wrong match: %q", res.Matches[0].Excerpt)
	}
}

func TestSessionQuery_IgnoresCaseAndAccents(t *testing.T) {
	// The motivating case: Vietnamese content written with tones must be found
	// by a model that recalls the phrase without them, and vice versa.
	store := seedLog(t, "s1",
		agentcore.Message{Role: agentcore.RoleAssistant, Content: "Doanh thu tháng Bảy tăng 12%."},
	)
	p := queryPolicy(t, store, "s1")
	for _, q := range []string{"doanh thu", "DOANH THU", "thang bay", "tháng bảy"} {
		res, err := p.provider.Search(context.Background(), SessionQueryRequest{SessionID: "s1", Query: q})
		if err != nil {
			t.Fatalf("Search(%q): %v", q, err)
		}
		if len(res.Matches) != 1 {
			t.Fatalf("query %q found %d matches, want 1 — accent/case folding is not working", q, len(res.Matches))
		}
	}
}

func TestSessionQuery_EmptyQueryBrowsesMostRecent(t *testing.T) {
	msgs := make([]agentcore.Message, 0, 20)
	for i := 0; i < 20; i++ {
		msgs = append(msgs, agentcore.Message{Role: agentcore.RoleAssistant, Content: strings.Repeat("entry ", 1) + string(rune('a'+i))})
	}
	store := seedLog(t, "s1", msgs...)
	p := queryPolicy(t, store, "s1")
	res, err := p.provider.Search(context.Background(), SessionQueryRequest{SessionID: "s1", Limit: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Matches) != 5 {
		t.Fatalf("got %d matches, want the 5 most recent", len(res.Matches))
	}
	// Most recent first.
	if res.Matches[0].Seq < res.Matches[4].Seq {
		t.Fatal("browse order should be newest first")
	}
}

func TestSessionQuery_ScopedToTheRunsOwnSession(t *testing.T) {
	store := agentcore.NewMemorySessionStore()
	mine := agentcore.Message{Role: agentcore.RoleAssistant, Content: "my secret figure is 42"}
	theirs := agentcore.Message{Role: agentcore.RoleAssistant, Content: "their secret figure is 99"}
	_ = store.Append(context.Background(), "s1", agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &mine})
	_ = store.Append(context.Background(), "s2", agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &theirs})

	p := queryPolicy(t, store, "s1")
	// The model cannot widen the scope: session id comes from the run, and the
	// tool takes no session parameter at all.
	out, err := (&sessionQueryTool{policy: p}).Run(context.Background(), `{"query":"secret figure","session_id":"s2"}`)
	if err != nil {
		t.Fatalf("session_query: %v", err)
	}
	if strings.Contains(out, "99") {
		t.Fatal("query leaked into another session's log")
	}
	if !strings.Contains(out, "42") {
		t.Fatalf("own-session match missing: %q", out)
	}
}

func TestSessionQuery_LimitIsCapped(t *testing.T) {
	msgs := make([]agentcore.Message, 0, 100)
	for i := 0; i < 100; i++ {
		msgs = append(msgs, agentcore.Message{Role: agentcore.RoleAssistant, Content: "match me"})
	}
	store := seedLog(t, "s1", msgs...)
	p := beginQuery(t, Plugin{MaxLimit: 3}, store, "s1")
	out, err := (&sessionQueryTool{policy: p}).Run(context.Background(), `{"query":"match","limit":999}`)
	if err != nil {
		t.Fatalf("session_query: %v", err)
	}
	if !strings.HasPrefix(out, "3 match(es)") {
		t.Fatalf("MaxLimit not enforced: %q", out[:40])
	}
}

func TestSessionQuery_NoMatchesIsExplicit(t *testing.T) {
	store := seedLog(t, "s1", agentcore.Message{Role: agentcore.RoleAssistant, Content: "nothing relevant"})
	p := queryPolicy(t, store, "s1")
	out, err := (&sessionQueryTool{policy: p}).Run(context.Background(), `{"query":"zzzzz"}`)
	if err != nil {
		t.Fatalf("session_query: %v", err)
	}
	if !strings.Contains(out, "No entries matched") || !strings.Contains(out, "searched 1") {
		t.Fatalf("a miss must distinguish 'no match' from 'empty log': %q", out)
	}

	empty := queryPolicy(t, agentcore.NewMemorySessionStore(), "s1")
	out, _ = (&sessionQueryTool{policy: empty}).Run(context.Background(), `{"query":"x"}`)
	if !strings.Contains(out, "No searchable history") {
		t.Fatalf("empty log should say so: %q", out)
	}
}

func TestSessionQuery_ExcerptIsBounded(t *testing.T) {
	store := seedLog(t, "s1", agentcore.Message{Role: agentcore.RoleAssistant, Content: "needle " + strings.Repeat("x", 50_000)})
	p := queryPolicy(t, store, "s1")
	res, err := p.provider.Search(context.Background(), SessionQueryRequest{SessionID: "s1", Query: "needle", ExcerptBytes: 200})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Matches[0].Excerpt) > 200 {
		t.Fatalf("excerpt is %d bytes — retrieval must not blow the context it was meant to save",
			len(res.Matches[0].Excerpt))
	}
}

func TestSessionQuery_BookkeepingEntriesAreNotNoise(t *testing.T) {
	store := agentcore.NewMemorySessionStore()
	ctx := context.Background()
	m := agentcore.Message{Role: agentcore.RoleAssistant, Content: "the model answer"}
	_ = store.Append(ctx, "s1", agentcore.SessionEntry{Kind: agentcore.EntryMessage, Message: &m})
	_ = store.Append(ctx, "s1", agentcore.SessionEntry{Kind: agentcore.EntryModelChange, Model: "model"})
	_ = store.Append(ctx, "s1", agentcore.SessionEntry{Kind: agentcore.EntryCompaction, Summary: "model summary"})

	p := queryPolicy(t, store, "s1")
	res, _ := p.provider.Search(ctx, SessionQueryRequest{SessionID: "s1", Query: "model"})
	for _, m := range res.Matches {
		if m.Kind != agentcore.EntryMessage {
			t.Fatalf("default scope should be messages only, got a %s entry", m.Kind)
		}
	}
	// But an explicit Kinds filter can reach the compaction summaries.
	res, _ = p.provider.Search(ctx, SessionQueryRequest{SessionID: "s1", Query: "model", Kinds: []agentcore.SessionEntryKind{agentcore.EntryCompaction}})
	if len(res.Matches) != 1 || res.Matches[0].Kind != agentcore.EntryCompaction {
		t.Fatalf("explicit Kinds filter did not reach compaction entries: %+v", res.Matches)
	}
}

func TestSessionQuery_DisabledWithoutProviderOrSession(t *testing.T) {
	// No provider and no durable session: nothing to search, so no tool.
	if p := beginQuery(t, Plugin{}, nil, ""); p != nil {
		t.Fatal("without a store or provider the seam must stay off")
	}
	if p := beginQuery(t, Plugin{}, agentcore.NewMemorySessionStore(), ""); p != nil {
		t.Fatal("without a session id the built-in provider has nothing to read")
	}
}

// erroringStore fails every Log read.
type erroringStore struct{}

func (erroringStore) Append(context.Context, string, agentcore.SessionEntry) error { return nil }
func (erroringStore) Log(context.Context, string) ([]agentcore.SessionEntry, error) {
	return nil, errors.New("store unavailable")
}

func TestSessionQuery_StoreErrorSurfaces(t *testing.T) {
	p := beginQuery(t, Plugin{}, erroringStore{}, "s1")
	if _, err := (&sessionQueryTool{policy: p}).Run(context.Background(), `{"query":"x"}`); err == nil {
		t.Fatal("a store failure must surface, not read as 'no results'")
	}
}

func TestSessionQuery_MalformedArguments(t *testing.T) {
	p := queryPolicy(t, seedLog(t, "s1", agentcore.Message{Role: agentcore.RoleAssistant, Content: "hi"}), "s1")
	tool := &sessionQueryTool{policy: p}
	if _, err := tool.Run(context.Background(), `not json`); err == nil {
		t.Fatal("malformed arguments must error")
	}
	// An empty argument object (or empty string) is a valid browse request.
	if _, err := tool.Run(context.Background(), ``); err != nil {
		t.Fatalf("empty arguments should browse, got %v", err)
	}
	if _, err := tool.Run(context.Background(), `{}`); err != nil {
		t.Fatalf("empty object should browse, got %v", err)
	}
}

func TestFoldText(t *testing.T) {
	cases := map[string]string{
		"Tháng Bảy": "thang bay",
		"CAFÉ":      "cafe",
		"plain":     "plain",
		"Đơn":       "đon", // đ is a distinct letter, not a combining mark
	}
	for in, want := range cases {
		if got := foldText(in); got != want {
			t.Errorf("foldText(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFoldTokens_DedupesAndDropsPunctuation(t *testing.T) {
	got := foldTokens("Doanh-thu, doanh THU!")
	if len(got) != 2 || got[0] != "doanh" || got[1] != "thu" {
		t.Fatalf("foldTokens = %v, want [doanh thu]", got)
	}
	if len(foldTokens("   ")) != 0 {
		t.Fatal("blank query must produce no tokens")
	}
}
