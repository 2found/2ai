package agentcore

import "context"

// GoalRevision is a consumer-authored change to a run's completion contract.
// The host persists it separately from provider messages.
type GoalRevision struct {
	Previous string `json:"previous"`
	Goal     string `json:"goal"`
	Reason   string `json:"reason"`
}

type goalRevisionRecorderKey struct{}
type goalRevisionRecorder func(context.Context, GoalRevision) error

// WithGoalRevisionRecorder installs the run owner's commit boundary. The
// consumer must call RecordGoalRevision before publishing the new condition,
// while holding the same lock that serializes its updates.
func WithGoalRevisionRecorder(ctx context.Context, record func(context.Context, GoalRevision) error) context.Context {
	return context.WithValue(ctx, goalRevisionRecorderKey{}, goalRevisionRecorder(record))
}

// RecordGoalRevision commits a change through the current run owner. Legacy
// runs without a recorder retain their existing extension-drain persistence.
func RecordGoalRevision(ctx context.Context, revision GoalRevision) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if record, _ := ctx.Value(goalRevisionRecorderKey{}).(goalRevisionRecorder); record != nil {
		return record(ctx, revision)
	}
	return nil
}
