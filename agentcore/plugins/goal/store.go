package goal

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

// Store holds the run's live completion condition and the record of how it got
// there.
//
// It exists for the same reason todo.Store does: the state has to outlive the
// transcript. A condition that lived only in the system message would be
// rewritten by compaction like anything else, and one that lived only in the
// gate's own field could not be read by the tool that changes it. So the
// condition is held here, the gate reads it, the tool writes it, and the loop
// drains it into the durable log.
//
// A nil *Store is not valid; the plugin always builds one.
type Store struct {
	mu    sync.Mutex
	state *State
	// goal is the condition currently in force.
	goal string
	// pending is set by a write and cleared by the loop's drain. It is what makes
	// ReviseGoal a drain rather than a poll: without it, every turn after the
	// first revision would write another identical EntryGoal.
	pending bool
	// revisions is the audit trail, oldest first, including the original.
	revisions []Revision
}

// Revision is one change to the run's objective: what it became, why, and when.
// The reason is the part a human reads later — "the goal changed" is not a
// finding, "the goal changed because CLEARING-7742 turned out to span three
// corpora" is.
type Revision struct {
	Goal   string
	Reason string
	At     time.Time
}

// NewStore builds a store holding the run's starting condition.
func NewStore(initial string) *Store {
	s := &Store{goal: strings.TrimSpace(initial)}
	if s.goal != "" {
		s.revisions = append(s.revisions, Revision{Goal: s.goal, Reason: "initial", At: time.Now()})
	}
	return s
}

// Goal returns the condition currently in force.
func (s *Store) Goal() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.goal
}

// Revisions returns the audit trail, oldest first.
func (s *Store) Revisions() []Revision {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Revision(nil), s.revisions...)
}

// Update replaces the condition and marks it for the loop to persist. A revision
// to the identical condition is dropped: it is not a change, and letting it
// through would write an EntryGoal and rebuild the system prompt — invalidating
// the provider's KV cache for the whole prefix — in exchange for nothing.
func (s *Store) Update(goal, reason string) bool {
	changed, _ := s.update(context.Background(), goal, reason)
	return changed
}

func (s *Store) update(ctx context.Context, goal, reason string) (bool, error) {
	goal = strings.TrimSpace(goal)
	if goal == "" {
		return false, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if goal == s.goal {
		return false, nil
	}
	if err := agentcore.RecordGoalRevision(ctx, agentcore.GoalRevision{Previous: s.goal, Goal: goal, Reason: strings.TrimSpace(reason)}); err != nil {
		return false, err
	}
	s.goal = goal
	s.pending = true
	s.revisions = append(s.revisions, Revision{Goal: goal, Reason: strings.TrimSpace(reason), At: time.Now()})
	return true, nil
}

// adopt brings the store to a condition that is ALREADY durable — the one the
// loop recovered from the log — recording it in the trail without marking it
// pending.
//
// The distinction is the whole point of the drain flag. A recovered condition is
// not a revision: nothing changed, the log already holds an EntryGoal for it, and
// the system prompt is being assembled from it right now. Routing it through
// Update would leave pending set, so the resumed run's very first turn would
// drain it — appending a duplicate EntryGoal and emitting "goal updated: <the
// same goal>" to a watching human about a change that never happened, which is
// the one event a run's viewer is least able to afford being wrong about.
func (s *Store) adopt(goal, reason string) {
	goal = strings.TrimSpace(goal)
	if goal == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if goal == s.goal {
		return
	}
	s.goal = goal
	s.revisions = append(s.revisions, Revision{Goal: goal, Reason: strings.TrimSpace(reason), At: time.Now()})
}

// takePending is the drain half of Update, called by the gate on the loop's
// behalf exactly once per revision.
func (s *Store) takePending() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.pending {
		return "", false
	}
	s.pending = false
	return s.goal, true
}
