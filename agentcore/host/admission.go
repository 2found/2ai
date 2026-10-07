package host

import (
	"context"
	"fmt"
	"sync"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

// Admit runs AI fallback while waiting only for the selected prepared request.
// run calls accept after committing selection and before publishing visible
// events. Its returned request is the last prepared candidate, used when all
// candidates fail without committing. AI owns replay/retry/fallback decisions.
func Admit(ctx context.Context, run func(context.Context, *ai.AssistantMessageEventStream, func(engine.Request)) (engine.Request, ai.AttemptOutcome, error)) (*engine.RequestAdmission, error) {
	type admission struct {
		request engine.Request
		err     error
	}
	ready := make(chan admission, 1)
	done := make(chan struct{})
	var once sync.Once
	var final ai.AttemptOutcome
	var failure error
	notify := func(request engine.Request, err error) { once.Do(func() { ready <- admission{request, err} }) }
	wait := func(ctx context.Context) error {
		select {
		case <-done:
			return failure
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	out := ai.NewAssistantMessageEventStreamWithHooks(ai.AssistantStreamHooks{
		Result: func(ctx context.Context) (*ai.Message, error) {
			if err := wait(ctx); err != nil {
				return nil, err
			}
			return final.Message(), nil
		}, AfterEnd: wait,
	})
	go func() {
		var prepared engine.Request
		defer func() {
			if value := recover(); value != nil {
				failure = fmt.Errorf("native request admission panic: %v", value)
			}
			out.End()
			close(done)
			notify(prepared, failure)
		}()
		prepared, final, failure = run(ctx, out, func(request engine.Request) { notify(request, nil) })
	}()
	selected := <-ready
	if selected.err != nil {
		return nil, selected.err
	}
	return &engine.RequestAdmission{Request: selected.request, Stream: out}, nil
}
