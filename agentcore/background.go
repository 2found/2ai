package agentcore

import (
	"context"
	"encoding/json"
	"errors"
)

// BackgroundLauncher is the run-owned scheduling capability. Its receipt is
// opaque to tools; the installed scheduler supplies status/wait/cancel tools.
// Keeping it here lets contribution plugins cooperate without importing peers.
type BackgroundLauncher func(tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error)
type backgroundKey struct{}

type runProgressKey struct{}

// WithRunProgress binds a host's run-lived observer. Background tools must use
// this instead of retaining a tool-lived streaming emitter. The host serializes
// callbacks and fences delivery when the run ends.
func WithRunProgress(ctx context.Context, progress func(string)) context.Context {
	return context.WithValue(ctx, runProgressKey{}, progress)
}

// ReportProgress publishes a display note, never transcript content or a tool
// result. It is inert without an observer or after cancellation.
func ReportProgress(ctx context.Context, note string) {
	if ctx.Err() != nil || note == "" {
		return
	}
	progress, _ := ctx.Value(runProgressKey{}).(func(string))
	if progress != nil {
		progress(note)
	}
}

func WithBackgroundLauncher(ctx context.Context, launch BackgroundLauncher) context.Context {
	return context.WithValue(ctx, backgroundKey{}, launch)
}
func LaunchBackground(ctx context.Context, tool, label string, run func(context.Context) (string, error)) (json.RawMessage, error) {
	launch, _ := ctx.Value(backgroundKey{}).(BackgroundLauncher)
	if launch == nil {
		return nil, errors.New("background work is not enabled for this run")
	}
	return launch(tool, label, run)
}

// BackgroundJob is the handle a scheduler binds to the context of one
// background job's work function. Work reaches it through PublishJobOutput and
// HandleJobMessages, never directly, so a tool does not depend on the plugin
// that scheduled it.
type BackgroundJob struct {
	// Output appends text to the job's live output. It must be cheap, safe for
	// concurrent use and bounded by the scheduler.
	Output func(text string)
	// OnMessage registers the handler for messages sent to the running job.
	OnMessage func(handle func(message string) error)
}

type backgroundJobKey struct{}

// WithBackgroundJob binds a job handle to the context its work runs under.
// Schedulers call it; tools never do.
func WithBackgroundJob(ctx context.Context, job BackgroundJob) context.Context {
	return context.WithValue(ctx, backgroundJobKey{}, job)
}

// PublishJobOutput appends text to the live output of the background job whose
// work function ctx belongs to, so the model can read how a running job is
// doing. It is inert outside a job; it still works after the job was
// cancelled, so the output a command flushes while it dies is kept. The
// scheduler bounds what it keeps; a caller may publish as often as it likes.
func PublishJobOutput(ctx context.Context, text string) {
	if text == "" {
		return
	}
	if job, _ := ctx.Value(backgroundJobKey{}).(BackgroundJob); job.Output != nil {
		job.Output(text)
	}
}

// InBackgroundJob reports whether ctx belongs to a background job's work, so a
// caller can skip building output nobody will read.
func InBackgroundJob(ctx context.Context) bool {
	job, _ := ctx.Value(backgroundJobKey{}).(BackgroundJob)
	return job.Output != nil
}

// HandleJobMessages makes the running background job accept messages from the
// model (the job_send tool). handle is called on the sender's goroutine and
// must only queue: it returns an error to refuse. A job that registers no
// handler refuses every message. Inert outside a job.
func HandleJobMessages(ctx context.Context, handle func(message string) error) {
	if job, _ := ctx.Value(backgroundJobKey{}).(BackgroundJob); job.OnMessage != nil {
		job.OnMessage(handle)
	}
}

// BackgroundValues returns a context that is cancelled with job but reads every
// value from values, except the job handle, which it keeps from job. A work
// function that runs its tool under the values of the call that started it
// (so the call's policy, telemetry and session still apply) uses it instead of
// a wrapper that hides job's values, which would silence PublishJobOutput.
func BackgroundValues(job, values context.Context) context.Context {
	return jobValues{Context: job, values: values}
}

type jobValues struct {
	context.Context
	values context.Context
}

func (c jobValues) Value(key any) any {
	if key == (backgroundJobKey{}) {
		return c.Context.Value(key)
	}
	return c.values.Value(key)
}
