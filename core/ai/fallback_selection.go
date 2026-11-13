package ai

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"
)

// FallbackSelection records the selected candidate without credentials or pool
// handles. Provider-row identity disambiguates candidates with equal models.
type FallbackSelection struct {
	Version    int             `json:"version"`
	Generation uint64          `json:"generation"`
	Rung       int             `json:"rung"`
	ProviderID string          `json:"providerId"`
	Model      json.RawMessage `json:"model"`
}

// ParseFallbackSelection validates one committed transition after the previous
// selection. The host must additionally check the selected provider/model against
// its bound candidates before opening a request.
func ParseFallbackSelection(raw string, previous *FallbackSelection) (FallbackSelection, error) {
	var record FallbackSelection
	var fields map[string]json.RawMessage
	invalid := func() (FallbackSelection, error) {
		return record, errors.New("corrupt native ladder selection record")
	}
	if json.Unmarshal([]byte(raw), &record) != nil || json.Unmarshal([]byte(raw), &fields) != nil {
		return invalid()
	}
	for _, name := range []string{"version", "generation", "rung", "providerId", "model"} {
		if len(fields[name]) == 0 || string(fields[name]) == "null" {
			return invalid()
		}
	}
	var model struct{ ID, API, Provider string }
	if record.Version != 1 || record.Rung < 0 || json.Unmarshal(record.Model, &model) != nil || model.ID == "" || model.API == "" || model.Provider == "" {
		return invalid()
	}
	generation, prior := uint64(0), 0
	if previous != nil {
		generation, prior = previous.Generation, previous.Rung
	}
	if generation == ^uint64(0) || record.Generation != generation+1 || record.Rung == prior {
		return invalid()
	}
	return record, nil
}

// FallbackLadder is a host's sticky rung with bounded recovery. Falling back
// keeps later requests on the rung that worked, and after RecoverAfter on it,
// or RecoverAfterSuccesses requests served by it, the next request probes the
// rungs above it once before using it. A failed probe restarts both counts.
// It is safe for concurrent requests.
type FallbackLadder struct {
	RecoverAfter          time.Duration // zero means 5 minutes
	RecoverAfterSuccesses int           // zero means 8
	Now                   func() time.Time

	mu        sync.Mutex
	rung      int
	since     time.Time
	successes int
}

// Rung reports the rung later requests start from.
func (l *FallbackLadder) Rung() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.rung
}

// Resume restores a rung from a checkpoint; its recovery clock starts now.
func (l *FallbackLadder) Resume(rung int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.move(max(rung, 0))
}

// Bind sets the request's Start and Probe and wraps Commit to record which
// rung served it.
func (l *FallbackLadder) Bind(request FallbackRequest) FallbackRequest {
	l.mu.Lock()
	rung := min(l.rung, max(request.Candidates-1, 0))
	request.Start, request.Probe = rung, 0
	if rung > 0 && l.due() {
		request.Start, request.Probe = 0, rung
	}
	l.mu.Unlock()
	commit, probe := request.Commit, request.Probe
	request.Commit = func(ctx context.Context, index int) error {
		if commit != nil {
			if err := commit(ctx, index); err != nil {
				return err
			}
		}
		l.mu.Lock()
		defer l.mu.Unlock()
		switch {
		case index != l.rung || probe > 0 && index >= probe: // moved, or the probe failed
			l.move(index)
		case index > 0:
			l.successes++
		}
		return nil
	}
	return request
}

func (l *FallbackLadder) due() bool {
	after, successes := l.RecoverAfter, l.RecoverAfterSuccesses
	if after <= 0 {
		after = 5 * time.Minute
	}
	if successes <= 0 {
		successes = 8
	}
	return l.successes >= successes || l.now().Sub(l.since) >= after
}

func (l *FallbackLadder) move(rung int) {
	l.rung, l.since, l.successes = rung, l.now(), 0
}

func (l *FallbackLadder) now() time.Time {
	if l.Now != nil {
		return l.Now()
	}
	return time.Now()
}
