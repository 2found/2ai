package engine

import (
	"context"

	"github.com/2found/2ai/ai"
)

// AgentEventStream is Pi's unbounded event queue with an independent final
// message result. Reading events is optional; Result does not consume them.
// A terminal agent_end settles Result before the producer itself returns.
// Result and the terminal event retain the same live *MessageList.
//
// Wait observes producer completion, including Go errors and panics. Like Pi's
// wrapper, a rejected low-level loop does not invent agent_end, close the event
// queue, or settle Result. Use Wait to observe such contract failures and a
// cancellable context when waiting for events/results that may never arrive.
// Provider error/aborted messages are normal terminal results, not failures.
type AgentEventStream struct {
	*ai.EventStream[Event, *MessageList]
	finished chan struct{}
	failure  error
}

func (s *AgentEventStream) Wait(ctx context.Context) error {
	select {
	case <-s.finished:
		return s.failure
	default:
	}
	select {
	case <-s.finished:
		return s.failure
	case <-ctx.Done():
		return ctx.Err()
	}
}

// AgentLoop starts the low-level loop asynchronously. It queues agent_start
// before returning, as the original wrapper does before its first await.
// Cancelling a reader does not abort the run; ctx controls the provider/tools.
func AgentLoop(ctx context.Context, prompts *MessageList, initial Context, config Config, provider StreamFn) *AgentEventStream {
	return startAgentStream(func(emit EventSink) (*MessageList, error) { return Run(ctx, prompts, initial, config, emit, provider) })
}

// AgentLoopContinue validates the existing tail synchronously, then continues
// without re-emitting old messages. Custom roles are validated by ConvertToLLM
// at the request boundary, exactly as in Continue.
func AgentLoopContinue(ctx context.Context, initial Context, config Config, provider StreamFn) (*AgentEventStream, error) {
	if err := validateContinuation(initial); err != nil {
		return nil, err
	}
	return startAgentStream(func(emit EventSink) (*MessageList, error) { return Continue(ctx, initial, config, emit, provider) }), nil
}

func startAgentStream(run func(EventSink) (*MessageList, error)) *AgentEventStream {
	stream := &AgentEventStream{
		EventStream: ai.NewEventStream(func(event Event) bool { return event.Type == "agent_end" }, func(event Event) *MessageList { return event.Messages }),
		finished:    make(chan struct{}),
	}
	admitted := make(chan struct{})
	go func() {
		first := true
		defer func() {
			if value := recover(); value != nil {
				stream.failure = failureError(value)
			}
			if first {
				close(admitted)
			}
			close(stream.finished)
		}()
		messages, err := run(func(event Event) error {
			stream.Push(event)
			if first {
				first = false
				close(admitted)
			}
			return nil
		})
		stream.failure = err
		if err == nil {
			stream.End(messages)
		}
	}()
	<-admitted
	return stream
}
