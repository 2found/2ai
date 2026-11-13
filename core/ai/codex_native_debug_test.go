package ai

import (
	"encoding/json"
	"errors"
	"sync"
	"testing"
)

func TestCodexDebugSnapshotResetAndConcurrency(t *testing.T) {
	policy := newCodexTransportPolicy()
	policy.recordRequest("s", true, true, json.RawMessage(`{"store":true,"input":[{}],"previous_response_id":"r"}`))
	policy.recordFailure("s", errors.New("lost"))
	policy.recordFallback("s")
	snapshot := policy.snapshot("s")
	if snapshot == nil || snapshot.ConnectionsReused != 1 || snapshot.StoreTrueRequests != 1 || snapshot.DeltaRequests != 1 || snapshot.SSEFallbacks != 1 || !policy.disabled("s") {
		t.Fatal("missing debug state")
	}
	*snapshot.LastDeltaInputItems = 999
	*snapshot.WebSocketFallbackActive = false
	*snapshot.LastWebSocketError = "changed"
	snapshot.LastPreviousResponseID[1] = 'X'
	actual := policy.snapshot("s")
	if *actual.LastDeltaInputItems != 1 || !*actual.WebSocketFallbackActive || *actual.LastWebSocketError != "lost" || string(actual.LastPreviousResponseID) != `"r"` {
		t.Fatal("snapshot aliases live state")
	}
	policy.recordRequest("s", false, false, json.RawMessage(`{"input":[]}`))
	actual = policy.snapshot("s")
	if actual.LastDeltaInputItems != nil || actual.LastPreviousResponseID != nil {
		t.Fatal("full request retained stale delta fields")
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			policy.recordRequest("other", false, true, json.RawMessage(`{"input":[]}`))
			_ = policy.snapshot("other")
		}()
	}
	wg.Wait()
	if policy.snapshot("other").Requests != 32 {
		t.Fatal("counter updates lost")
	}
	policy.reset("s")
	if policy.snapshot("s") != nil || policy.disabled("s") || policy.snapshot("other") == nil {
		t.Fatal("per-session reset crossed session boundary")
	}
	policy.reset("")
	if policy.snapshot("other") != nil {
		t.Fatal("all-session reset retained counters")
	}
}

func TestCodexDebugPublicResetKeepsSocketAndCloseKeepsFallback(t *testing.T) {
	session := t.Name()
	defer ResetCodexWebSocketDebugStats(session)
	defer CloseCodexResponsesSessions(session)
	cache := codexNativeSockets
	key := codexSocketKey{session, "account"}
	closed := 0
	entry := &codexSocketEntry{socket: &codexSocket{close: func(int, string) { closed++ }}, busy: true}
	cache.mu.Lock()
	cache.insert(key, entry)
	cache.mu.Unlock()
	codexNativePolicy.recordFailure(session, errors.New("lost"))
	ResetCodexWebSocketDebugStats(session)
	if GetCodexWebSocketDebugStats(session) != nil || codexNativePolicy.disabled(session) || closed != 0 {
		t.Fatal("debug reset closed a connection or retained fallback")
	}
	codexNativePolicy.recordFailure(session, errors.New("lost again"))
	CloseCodexResponsesSessions(session)
	if closed != 1 || !codexNativePolicy.disabled(session) || GetCodexWebSocketDebugStats(session) == nil {
		t.Fatal("closing socket reset debug/fallback state")
	}
}
