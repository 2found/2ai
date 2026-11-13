package ai

import "encoding/json"

// CodexWebSocketDebugStats is a snapshot; optional fields retain Pi's distinction
// between a value that has never been recorded and an explicit false/zero.
type CodexWebSocketDebugStats struct {
	Requests                int             `json:"requests"`
	ConnectionsCreated      int             `json:"connectionsCreated"`
	ConnectionsReused       int             `json:"connectionsReused"`
	CachedContextRequests   int             `json:"cachedContextRequests"`
	StoreTrueRequests       int             `json:"storeTrueRequests"`
	FullContextRequests     int             `json:"fullContextRequests"`
	DeltaRequests           int             `json:"deltaRequests"`
	LastInputItems          int             `json:"lastInputItems"`
	LastDeltaInputItems     *int            `json:"lastDeltaInputItems,omitempty"`
	LastPreviousResponseID  json.RawMessage `json:"lastPreviousResponseId,omitempty"`
	WebSocketFailures       int             `json:"websocketFailures"`
	SSEFallbacks            int             `json:"sseFallbacks"`
	WebSocketFallbackActive *bool           `json:"websocketFallbackActive,omitempty"`
	LastWebSocketError      *string         `json:"lastWebSocketError,omitempty"`
}

func (p *codexTransportPolicy) statsLocked(session string) *CodexWebSocketDebugStats {
	stats := p.stats[session]
	if stats == nil {
		stats = &CodexWebSocketDebugStats{}
		p.stats[session] = stats
	}
	return stats
}
func (p *codexTransportPolicy) recordRequest(session string, reused, cached bool, body json.RawMessage) {
	if session == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	stats := p.statsLocked(session)
	stats.Requests++
	if reused {
		stats.ConnectionsReused++
	} else {
		stats.ConnectionsCreated++
	}
	if cached {
		stats.CachedContextRequests++
	}
	fields, _ := samplingObject(body)
	if string(fields["store"]) == "true" {
		stats.StoreTrueRequests++
	}
	var input []json.RawMessage
	_ = json.Unmarshal(fields["input"], &input)
	stats.LastInputItems = len(input)
	if samplingTruthy(fields["previous_response_id"]) {
		stats.DeltaRequests++
		count := len(input)
		stats.LastDeltaInputItems = &count
		stats.LastPreviousResponseID = append(json.RawMessage(nil), fields["previous_response_id"]...)
	} else {
		stats.FullContextRequests++
		stats.LastDeltaInputItems = nil
		stats.LastPreviousResponseID = nil
	}
}
func (p *codexTransportPolicy) recordFallback(session string) {
	if session == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	stats := p.statsLocked(session)
	stats.SSEFallbacks++
	active := p.fallback[session]
	stats.WebSocketFallbackActive = &active
}
func (p *codexTransportPolicy) snapshot(session string) *CodexWebSocketDebugStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	stats := p.stats[session]
	if stats == nil {
		return nil
	}
	// Deep-copy optional fields so a caller cannot modify live counters.
	snapshot := *stats
	if stats.LastDeltaInputItems != nil {
		v := *stats.LastDeltaInputItems
		snapshot.LastDeltaInputItems = &v
	}
	if stats.WebSocketFallbackActive != nil {
		v := *stats.WebSocketFallbackActive
		snapshot.WebSocketFallbackActive = &v
	}
	if stats.LastWebSocketError != nil {
		v := *stats.LastWebSocketError
		snapshot.LastWebSocketError = &v
	}
	snapshot.LastPreviousResponseID = append(json.RawMessage(nil), stats.LastPreviousResponseID...)
	return &snapshot
}

func GetCodexWebSocketDebugStats(session string) *CodexWebSocketDebugStats {
	return codexNativePolicy.snapshot(session)
}

// ResetCodexWebSocketDebugStats clears counters and sticky fallback for one
// session (or all sessions if empty). It does not close cached connections.
func ResetCodexWebSocketDebugStats(session string) { codexNativePolicy.reset(session) }
