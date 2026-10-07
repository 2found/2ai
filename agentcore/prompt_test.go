package agentcore

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// The recalled-memory block must not re-pay tokens for paraphrases of the same
// fact, but must keep genuinely distinct facts.
func TestDedupRecalledDropsParaphrasesKeepsDistinct(t *testing.T) {
	recalled := []MemoryEntry{
		{Kind: "learning", Content: "Top page by traffic is the homepage ('/'), followed by the novel details page for Kiem Lai ('/truyen/kiem-lai'), user profile ('/profile'), and admin novels manager ('/admin/novels')."},
		{Kind: "learning", Content: "The top page by traffic is the homepage ('/'), followed by the novel page '/truyen/kiem-lai' and user profile '/profile'."},
		{Kind: "learning", Content: "Top pages by traffic this week are the homepage (/), followed by the novel details page for kiem-lai (/truyen/kiem-lai), /profile, and admin pages (/admin/novels)."},
		{Kind: "preference", Content: "The user prefers stacked-area charts for time-series dashboards."},
	}
	got := dedupRecalled(recalled)
	if len(got) < 2 {
		t.Fatalf("dedup collapsed too much: %d kept", len(got))
	}
	// The distinct preference fact must survive.
	foundPref := false
	for _, m := range got {
		if strings.Contains(m.Content, "stacked-area") {
			foundPref = true
		}
	}
	if !foundPref {
		t.Fatal("dedup dropped a distinct fact (stacked-area preference)")
	}
	// At least one of the three traffic paraphrases must be dropped.
	if len(got) >= len(recalled) {
		t.Fatalf("expected paraphrases to be deduped, kept all %d", len(got))
	}
}

// Vietnamese paraphrases that differ only by diacritics / minor wording must be
// recognised as the same fact (accent folding), not kept as separate bullets.
func TestDedupRecalledFoldsVietnameseAccents(t *testing.T) {
	recalled := []MemoryEntry{
		{Kind: "learning", Content: "Trang có nhiều lượt truy cập nhất là trang chủ, sau đó là trang truyện Kiếm Lai và trang hồ sơ người dùng."},
		{Kind: "learning", Content: "Trang co nhieu luot truy cap nhat la trang chu, sau do la trang truyen Kiem Lai va trang ho so nguoi dung."},
		{Kind: "preference", Content: "Người dùng thích biểu đồ vùng xếp chồng cho dữ liệu theo thời gian."},
	}
	got := dedupRecalled(recalled)
	if len(got) != 2 {
		t.Fatalf("expected accent-only paraphrase to collapse to 2 bullets, got %d", len(got))
	}
}

func TestDedupRecalledNoFalsePositives(t *testing.T) {
	recalled := []MemoryEntry{
		{Kind: "learning", Content: "D7 retention is 22 percent for new readers."},
		{Kind: "learning", Content: "The payment conversion funnel drops most at the checkout step."},
		{Kind: "learning", Content: "Mobile accounts for 80 percent of sessions."},
	}
	if got := dedupRecalled(recalled); len(got) != 3 {
		t.Fatalf("distinct facts were wrongly deduped: %d/3 kept", len(got))
	}
}

// The recalled block is assembled into the system prefix and therefore charged
// on every turn of the run, so its size has to be bounded — nothing did that
// before. These pin both clamps, and that they clamp SIZE only: the block's
// heading and caveat wording are untouched.
func TestRecalledBlockClampsPerEntry(t *testing.T) {
	long := strings.Repeat("a", maxRecallEntryBytes*3)
	got := buildSystemPrompt(AgentDefinition{}, []MemoryEntry{{Kind: "fact", Content: long}}, nil)
	if !strings.Contains(got, "# Recalled memory") {
		t.Fatal("the recalled block is missing entirely")
	}
	if strings.Contains(got, long) {
		t.Fatal("a single memory was injected at full length; the per-entry clamp did not apply")
	}
	if !strings.Contains(got, "…[truncated]") {
		t.Error("the clamped entry does not say it was truncated")
	}
	// One over-long memory must not cost more than its own budget plus the
	// bullet's own few bytes of framing.
	if n := len(got); n > maxRecallEntryBytes+len(responseFormattingGuidance)+512 {
		t.Errorf("prompt is %d bytes for one clamped memory — the clamp is not bounding the entry", n)
	}
}

func TestRecalledBlockClampsTotalSize(t *testing.T) {
	// Distinct facts (dedup must not be what removes them), each near the
	// per-entry cap, so together they blow the block budget several times over.
	var recalled []MemoryEntry
	for i := 0; i < 40; i++ {
		recalled = append(recalled, MemoryEntry{
			Kind:    "fact",
			Content: fmt.Sprintf("fact number %d: ", i) + strings.Repeat(string(rune('a'+i%26)), maxRecallEntryBytes/2),
		})
	}
	got := buildSystemPrompt(AgentDefinition{}, recalled, nil)
	bullets := strings.Count(got, "\n- (fact) ")
	if bullets == 0 {
		t.Fatal("the whole block was clamped away; the budget must still admit what fits")
	}
	if bullets == len(recalled) {
		t.Fatal("every memory was injected; the block budget did not apply")
	}
	if n := len(got); n > maxRecallBlockBytes+len(responseFormattingGuidance)+2048 {
		t.Errorf("prompt is %d bytes, want the recalled block held near its %d-byte budget", n, maxRecallBlockBytes)
	}
}

// Clamping runs AFTER dedup, so a paraphrase cannot spend budget a distinct
// fact further down the list needed. Without that ordering the near-duplicates
// below would consume the block and the distinct fact would be cut.
func TestRecalledBlockClampsAfterDedup(t *testing.T) {
	filler := strings.Repeat("the homepage is the top page by traffic and always has been. ", 8)
	var recalled []MemoryEntry
	for i := 0; i < 30; i++ {
		recalled = append(recalled, MemoryEntry{Kind: "learning", Content: filler})
	}
	const distinct = "checkout conversion drops most at the address step"
	recalled = append(recalled, MemoryEntry{Kind: "fact", Content: distinct})

	got := buildSystemPrompt(AgentDefinition{}, recalled, nil)
	if !strings.Contains(got, distinct) {
		t.Fatal("thirty paraphrases of one learning spent the whole block budget and cut the distinct fact")
	}
}

// TestSystemPromptAdvertisesHeadersNotBodies verifies progressive disclosure:
// all enabled skill headers are listed, but no body is inlined and the loader is
// not called up front.
func TestSystemPromptAdvertisesHeadersNotBodies(t *testing.T) {
	var loaderCalls int
	recorder := &nativeRecorder{}
	faux := recordedNativeProvider(recorder, AssistantText("done"))
	agent, err := New(Config{
		NativeProvider: faux,
		Model:          "test",
		Definition: AgentDefinition{
			Soul:   "identity",
			Agents: "mission",
			Skills: []Skill{
				{ID: "query", Name: "query-runner", Description: "use for query reporting", Enabled: true},
				{ID: "email", Name: "emailer", Description: "use for outbound email", Enabled: true},
				{ID: "off", Name: "disabled-skill", Description: "never advertised", Enabled: false},
			},
			SkillLoader: func(_ context.Context, ids []string) (map[string]string, error) {
				loaderCalls++
				return map[string]string{"query": "steps for running safe queries"}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := agent.Prompt(context.Background(), "please help with query reporting"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if loaderCalls != 0 {
		t.Fatalf("loader must not run during perceive, got %d calls", loaderCalls)
	}
	var system string
	for _, m := range recorder.all()[0].Messages {
		if m.Role == RoleSystem {
			system = m.Content
		}
	}
	// All enabled headers present; disabled one absent.
	for _, want := range []string{"id: query", "query-runner", "id: email", "emailer"} {
		if !strings.Contains(system, want) {
			t.Fatalf("system prompt missing header %q: %q", want, system)
		}
	}
	if strings.Contains(system, "disabled-skill") {
		t.Fatalf("disabled skill advertised: %q", system)
	}
	// No bodies inlined.
	if strings.Contains(system, "steps for running safe queries") {
		t.Fatalf("system prompt should not inline skill bodies: %q", system)
	}
}

// TestReadSkillRoundTrip verifies the model can pull exactly one body on demand
// via read_skill, even under the default-deny policy, and the loader is asked for
// only that skill.
func TestReadSkillRoundTrip(t *testing.T) {
	var loadedIDs []string
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", readSkillToolName, `{"id":"query"}`),
		AssistantText("loaded and done"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:    "test",
		Definition: AgentDefinition{
			Skills: []Skill{
				{ID: "query", Name: "query-runner", Description: "use for query reporting", Enabled: true},
				{ID: "email", Name: "emailer", Description: "use for outbound email", Enabled: true},
			},
			SkillLoader: func(_ context.Context, ids []string) (map[string]string, error) {
				loadedIDs = append(loadedIDs, ids...)
				return map[string]string{
					"query": "steps for running safe queries",
					"email": "steps for sending email",
				}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	// Loader asked for exactly the one requested skill.
	if len(loadedIDs) != 1 || loadedIDs[0] != "query" {
		t.Fatalf("loaded IDs = %v, want [query]", loadedIDs)
	}
	// The tool result fed back to the model carries that one body.
	var sawBody bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && m.Name == readSkillToolName && strings.Contains(m.Content, "steps for running safe queries") {
			sawBody = true
		}
		if strings.Contains(m.Content, "steps for sending email") {
			t.Fatalf("unrequested skill body entered context: %q", m.Content)
		}
	}
	if !sawBody {
		t.Fatalf("read_skill body never reached the model: %+v", res.Messages)
	}
	// The call was allowed despite default-deny policy.
	for _, tr := range res.Tools {
		if tr.Tool == readSkillToolName && !tr.Allowed {
			t.Fatalf("read_skill was blocked by policy: %+v", tr)
		}
	}
}

// TestReadSkillServesPreloadedBody verifies a preloaded body is returned without
// a loader configured.
func TestReadSkillServesPreloadedBody(t *testing.T) {
	faux := scriptedNativeProvider(
		AssistantToolCall("c1", readSkillToolName, `{"id":"sql-runner"}`),
		AssistantText("done"),
	)
	agent, err := New(Config{
		NativeProvider: faux,
		Model:    "test",
		Definition: AgentDefinition{
			Skills: []Skill{{Name: "sql-runner", Description: "sql helper", Body: "preloaded body", Enabled: true}},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	res, err := agent.Prompt(context.Background(), "need sql help")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	var sawBody bool
	for _, m := range res.Messages {
		if m.Role == RoleTool && strings.Contains(m.Content, "preloaded body") {
			sawBody = true
		}
	}
	if !sawBody {
		t.Fatalf("preloaded body not returned by read_skill: %+v", res.Messages)
	}
}

// With no hook shaping the request, the request IS the history, so the anchor
// goes on the last message — the whole turn becomes the cached prefix and the
// next turn re-reads it.
func TestMarkCacheAnchorsStampsFinalMessage(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "task"},
		{Role: RoleAssistant, Content: "done"},
	}
	out := markCacheAnchors(msgs, msgs, "session-1")
	if !out[len(out)-1].CacheAnchor {
		t.Fatal("final message must carry the moving anchor")
	}
	for i := 0; i < len(out)-1; i++ {
		if out[i].CacheAnchor {
			t.Fatalf("unexpected anchor on message %d", i)
		}
	}
	// The persisted history must never carry anchors.
	for i, m := range msgs {
		if m.CacheAnchor {
			t.Fatalf("input slice mutated: anchor on message %d", i)
		}
	}
}

// The case the placement policy exists for. A ContextHook shapes the request
// without entering persisted history, so its trailer is regenerated every turn.
// Anchoring on it caches a prefix that cannot be a prefix of the next request,
// and the run pays the cache-write premium forever without a single read back.
func TestMarkCacheAnchorsSkipsAHookInjectedTrailer(t *testing.T) {
	history := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "task"},
		{Role: RoleAssistant, Content: "done"},
	}
	req := append(append([]Message(nil), history...),
		Message{Role: RoleSystem, Content: "[run plan]\n[~] step one"})

	out := markCacheAnchors(req, history, "k")
	if out[len(out)-1].CacheAnchor {
		t.Fatal("the anchor is on the regenerated trailer: every cache entry it writes is " +
			"unreadable on the next turn, because that trailer will have been re-rendered")
	}
	if !out[len(history)-1].CacheAnchor {
		t.Fatalf("the anchor should sit at the end of the append-only history (index %d): that is "+
			"the longest prefix guaranteed to still be a prefix next turn", len(history)-1)
	}
}

// A hook that rewrites an EARLIER message (redaction) breaks the append-only
// guarantee from that point on, so the anchor stops before it. Conservative, but
// what it writes can actually be read.
func TestMarkCacheAnchorsStopsAtARewrittenMessage(t *testing.T) {
	history := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "my password is hunter2"},
		{Role: RoleAssistant, Content: "done"},
	}
	req := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "my password is [redacted]"},
		{Role: RoleAssistant, Content: "done"},
	}
	out := markCacheAnchors(req, history, "k")
	if !out[0].CacheAnchor {
		t.Fatalf("the anchor should stop at the last message the request and the history agree on")
	}
	for i := 1; i < len(out); i++ {
		if out[i].CacheAnchor {
			t.Fatalf("anchored at %d, past the point where the request diverges from history", i)
		}
	}
}

func TestMarkCacheAnchorsStopsAtChangedRichContent(t *testing.T) {
	history := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleTool, ToolCallID: "c1", Content: "plot", ContentParts: []ContentPart{{Type: ContentPartImage, MIMEType: "image/png", Data: "old"}}},
		{Role: RoleAssistant, Content: "done"},
	}
	req := cloneSessionMessages(history)
	req[1].ContentParts[0].Data = "new"
	out := markCacheAnchors(req, history, "k")
	if !out[0].CacheAnchor {
		t.Fatal("cache prefix must stop before a changed image payload")
	}
	for i := 1; i < len(out); i++ {
		if out[i].CacheAnchor {
			t.Fatalf("anchored at %d past changed rich content", i)
		}
	}
}

func TestMarkCacheAnchorsStopsAtChangedReasoningReplayState(t *testing.T) {
	history := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleAssistant, Content: "same answer", ReasoningBlocks: []ReasoningBlock{{
			Type: ReasoningBlockThinking, Signature: "old", ReplayScope: "anthropic:scope",
		}}},
		{Role: RoleUser, Content: "continue"},
	}
	req := cloneSessionMessages(history)
	req[1].ReasoningBlocks[0].Signature = "new"
	out := markCacheAnchors(req, history, "k")
	if !out[0].CacheAnchor {
		t.Fatal("cache prefix must stop before changed signed reasoning state")
	}
	for i := 1; i < len(out); i++ {
		if out[i].CacheAnchor {
			t.Fatalf("anchored at %d past changed reasoning state", i)
		}
	}
}

func TestMarkCacheAnchorsClearsStaleMarks(t *testing.T) {
	// A hook (or a bug) leaving anchors on history must not accumulate into
	// more breakpoints than a provider allows.
	msgs := []Message{
		{Role: RoleUser, Content: "a", CacheAnchor: true},
		{Role: RoleAssistant, Content: "b", CacheAnchor: true},
		{Role: RoleUser, Content: "c"},
	}
	out := markCacheAnchors(msgs, msgs, "k")
	got := 0
	for _, m := range out {
		if m.CacheAnchor {
			got++
		}
	}
	if got != 1 || !out[2].CacheAnchor {
		t.Fatalf("want exactly one anchor on the final message, got %d", got)
	}
}

func TestMarkCacheAnchorsNoopWithoutCacheKey(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: "a"}}
	out := markCacheAnchors(msgs, msgs, "")
	if out[0].CacheAnchor {
		t.Fatal("no cacheKey must mean no anchors")
	}
}

// With nothing known-stable there is no honest place to put a breakpoint, and
// writing one anyway costs the write premium for an entry that cannot be read.
func TestMarkCacheAnchorsWritesNothingWhenNothingIsStable(t *testing.T) {
	req := []Message{{Role: RoleUser, Content: "a"}}
	out := markCacheAnchors(req, nil, "k")
	for i, m := range out {
		if m.CacheAnchor {
			t.Fatalf("anchored at %d with no history to vouch for it", i)
		}
	}
}
