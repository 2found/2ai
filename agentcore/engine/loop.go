package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"

	"github.com/2found/2ai/ai"
	"github.com/2found/2ai/jsonjs"
)

// Run starts a prompt invocation. Its returned list is shared with
// Turn.NewMessages and the agent_end event, and excludes pre-existing history.
// The prompt remains live during initial event iteration unless declaring tools
// creates a replacement list. Context/result membership is copied before events.
// Hooks and sink failures propagate without inventing agent_end.
func Run(ctx context.Context, prompts *MessageList, initial Context, config Config, emit EventSink, stream StreamFn) (*MessageList, error) {
	initialMessages, err := declareToolChanges(initial, prompts, config.now)
	if err != nil {
		return nil, err
	}
	newMessages := NewList(initialMessages.Values()...)
	current := initial
	current.Messages = NewList(append(initial.Messages.Values(), initialMessages.Values()...)...)
	if err := emit(Event{Type: "agent_start"}); err != nil {
		return nil, err
	}
	if err := emit(Event{Type: "turn_start"}); err != nil {
		return nil, err
	}
	for index := 0; index < initialMessages.Len(); index++ {
		message := initialMessages.Get(index)
		if err := emitMessage(message, emit); err != nil {
			return nil, err
		}
	}
	if stream == nil && config.AdmitRequest == nil {
		var err error
		stream, err = GetDefaultStreamFn()
		if err != nil {
			return nil, err
		}
	}
	return runLoop(ctx, &current, newMessages, config, emit, stream)
}

// Continue reuses the transcript without emitting the existing prompt again.
func Continue(ctx context.Context, initial Context, config Config, emit EventSink, stream StreamFn) (*MessageList, error) {
	if err := validateContinuation(initial); err != nil {
		return nil, err
	}
	if err := emit(Event{Type: "agent_start"}); err != nil {
		return nil, err
	}
	if err := emit(Event{Type: "turn_start"}); err != nil {
		return nil, err
	}
	if stream == nil && config.AdmitRequest == nil {
		var err error
		stream, err = GetDefaultStreamFn()
		if err != nil {
			return nil, err
		}
	}
	return runLoop(ctx, &initial, NewList[*ai.Message](), config, emit, stream)
}

func runLoop(ctx context.Context, current *Context, messages *MessageList, config Config, emit EventSink, stream StreamFn) (*MessageList, error) {
	sink := emit
	emit = func(event Event) error {
		// Streaming events already hold their provider lock. Subsequent tool,
		// turn and terminal events may still reference that provider's result.
		if event.payloadYield != nil || current.assistantStream == nil {
			return sink(event)
		}
		return current.assistantStream.SynchronizeYielding(func(yield ai.PayloadYield) error {
			event.payloadYield, event.payloadAccess = yield, current.assistantStream.Synchronize
			return sink(event)
		})
	}
	var last *Turn
	pending, err := poll(config.GetSteeringMessages)
	if err != nil {
		return nil, err
	}
	explicit := false
	finish := func() (*MessageList, error) { return messages, emit(Event{Type: "agent_end", Messages: messages}) }
	for {
		moreTools := true
		for moreTools || pending.Len() > 0 {
			var prepared *MessageList
			if last != nil {
				if config.PrepareNextTurn != nil {
					update, err := config.PrepareNextTurn(last)
					if err != nil {
						return nil, err
					}
					if update != nil {
						current = applyUpdate(current, &config, update)
						prepared = update.Messages
					}
				}
				if pending.Len() == 0 {
					pending, err = poll(config.GetSteeringMessages)
					if err != nil {
						return nil, err
					}
				}
				if err := emit(Event{Type: "turn_start"}); err != nil {
					return nil, err
				}
			}
			selected, err := declareToolChanges(*current, NewList(append(prepared.Values(), pending.Values()...)...), config.now)
			if err != nil {
				return nil, err
			}
			for index := 0; index < selected.Len(); index++ {
				message := selected.Get(index)
				if err := emitMessage(message, emit); err != nil {
					return nil, err
				}
				current.Messages.Append(message)
				messages.Append(message)
			}
			pending = nil
			var admitted *RequestAdmission
			if config.AdmitRequest != nil {
				admitted, err = config.AdmitRequest(ctx, Request{Context: current, Model: config.Model, ThinkingLevel: config.thinkingLevel()}, requestOptions(config))
				if err != nil {
					return nil, err
				}
				if admitted == nil || admitted.Stream == nil || admitted.Request.Context == nil || len(admitted.Request.Model) == 0 {
					return nil, errors.New("engine: invalid prepared request admission")
				}
				current = applyUpdate(current, &config, &TurnUpdate{Context: admitted.Request.Context, ThinkingLevel: &admitted.Request.ThinkingLevel})
				// Admission has already selected the request and opened its stream.
				// Preserve that model exactly, including an explicit JSON null.
				config.Model = admitted.Request.Model
			} else if config.PrepareRequest != nil {
				update, err := config.PrepareRequest(ctx, Request{Context: current, Model: config.Model, ThinkingLevel: config.thinkingLevel()})
				if err != nil {
					return nil, err
				}
				if update != nil {
					current = applyUpdate(current, &config, update)
				}
			}
			var message *ai.Message
			if admitted != nil {
				message, err = consumeAssistant(ctx, current, config, emit, admitted.Stream)
			} else {
				message, err = streamAssistant(ctx, current, config, emit, stream)
			}
			if err != nil {
				return nil, err
			}
			messages.Append(message)
			toolResults := NewList[*ai.Message]()
			var stopReason string
			var calls []*ai.ContentBlock
			current.readAssistant(func() {
				stopReason = message.StopReason
				if stopReason != "error" && stopReason != "aborted" {
					calls, err = toolCalls(message)
				}
			})
			if err != nil {
				return nil, err
			}
			if stopReason == "error" || stopReason == "aborted" {
				last = &Turn{Message: message, ToolResults: toolResults, Context: current, NewMessages: messages}
				if config.FinishTurn != nil {
					if _, err := config.FinishTurn(ctx, last); err != nil {
						return nil, err
					}
				}
				if err := emit(Event{Type: "turn_end", Message: message, ToolResults: NewList[*ai.Message]()}); err != nil {
					return nil, err
				}
				return finish()
			}
			moreTools = false
			if len(calls) > 0 {
				batch, err := executeBatch(ctx, current, message, calls, stopReason == "length", config, emit)
				if err != nil {
					return nil, err
				}
				toolResults.Append(batch.messages...)
				moreTools = !batch.terminate
				current.Messages.Append(toolResults.Values()...)
				messages.Append(toolResults.Values()...)
			}
			last = &Turn{Message: message, ToolResults: toolResults, Context: current, NewMessages: messages}
			decision := ""
			if config.FinishTurn != nil {
				decision, err = config.FinishTurn(ctx, last)
				if err != nil {
					return nil, err
				}
			}
			if err := emit(Event{Type: "turn_end", Message: message, ToolResults: toolResults}); err != nil {
				return nil, err
			}
			if decision == "end" {
				return finish()
			}
			explicit = decision == "continue"
			pending, err = poll(config.GetSteeringMessages)
			if err != nil {
				return nil, err
			}
			if moreTools || pending.Len() > 0 {
				explicit = false
			}
		}
		followUp, err := poll(config.GetFollowUpMessages)
		if err != nil {
			return nil, err
		}
		if followUp.Len() > 0 {
			explicit = false
			pending = followUp
			continue
		}
		if explicit {
			explicit = false
			continue
		}
		return finish()
	}
}

func applyUpdate(current *Context, config *Config, update *TurnUpdate) *Context {
	if update.Context != nil {
		current = update.Context
		if current.Messages == nil {
			current.Messages = NewList[*ai.Message]()
		}
	}
	if update.Model != nil && !bytes.Equal(bytes.TrimSpace(update.Model), []byte("null")) {
		config.Model = update.Model
	}
	if update.ThinkingLevel != nil {
		config.setThinkingLevel(*update.ThinkingLevel)
	}
	return current
}

func poll(fn func() (*MessageList, error)) (*MessageList, error) {
	if fn == nil {
		return nil, nil
	}
	return fn()
}

func (c Config) hasReasoning() bool {
	return c.reasoningDefined || c.Reasoning != ""
}

func (c Config) thinkingLevel() string {
	if !c.hasReasoning() {
		return "off"
	}
	return c.Reasoning
}

func (c *Config) setThinkingLevel(level string) {
	c.Reasoning = level
	c.reasoningDefined = level != "off"
	if !c.reasoningDefined {
		c.Reasoning = ""
	}
}

func emitMessage(message *ai.Message, emit EventSink) error {
	if err := emit(Event{Type: "message_start", Message: message}); err != nil {
		return err
	}
	return emit(Event{Type: "message_end", Message: message})
}

func declareToolChanges(current Context, pending *MessageList, now func() int64) (*MessageList, error) {
	systemIndex := -1
	for i := pending.Len() - 1; i >= 0; i-- {
		message := pending.Get(i)
		if message == nil {
			return nil, jsonjs.PropertyReadError(pending.Has(i), "pendingMessages[i].role")
		}
		if message.Role == "system" {
			systemIndex = i
			break
		}
	}
	baseline := pending
	if systemIndex >= 0 {
		baseline = pending.Clone()
		message := ai.WithToolChanges(*pending.Get(systemIndex), ai.ToolStateChanges{})
		baseline.Set(systemIndex, &message)
	}
	// Replay the history before projecting tools, matching argument evaluation
	// order in getToolStateChanges(getCurrentTools(...), tools.map(...)).
	history, err := replayMessageValues(current.Messages)
	if err != nil {
		return nil, err
	}
	baselineValues, err := replayMessageValues(baseline)
	if err != nil {
		return nil, err
	}
	previousTools := ai.GetCurrentTools(append(history, baselineValues...))
	tools, err := declareTools(current.Tools)
	if err != nil {
		return nil, err
	}
	changes := ai.GetToolStateChanges(previousTools, tools)
	unchanged := len(changes.ToolsAdded) == 0 && len(changes.ToolsRemoved) == 0
	if systemIndex >= 0 {
		if unchanged && len(pending.Get(systemIndex).ToolsAdded) == 0 && len(pending.Get(systemIndex).ToolsRemoved) == 0 {
			return pending, nil
		}
		message := ai.WithToolChanges(*pending.Get(systemIndex), changes)
		baseline.Set(systemIndex, &message)
		return baseline, nil
	}
	if unchanged {
		return pending, nil
	}
	update := ai.WithToolChanges(ai.Message{Role: "system", Content: ai.TextContent(""), Timestamp: now()}, changes)
	index := pending.Len()
	for i, message := range pending.Values() {
		if message.Role != "system" {
			index = i
			break
		}
	}
	return NewList(slices.Insert(pending.Clone().Values(), index, &update)...), nil
}

func streamAssistant(ctx context.Context, current *Context, config Config, emit EventSink, stream StreamFn) (*ai.Message, error) {
	messages := current.Messages
	var err error
	if config.TransformContext != nil {
		messages, err = config.TransformContext(ctx, messages)
		if err != nil {
			return nil, err
		}
	}
	if config.ConvertToLLM == nil {
		return nil, errors.New("engine: convertToLLM is required")
	}
	llmMessages, err := config.ConvertToLLM(messages)
	if err != nil {
		return nil, err
	}
	options := requestOptions(config)
	if config.GetAPIKey != nil {
		var model struct {
			Provider string `json:"provider"`
		}
		if err := json.Unmarshal(config.Model, &model); err != nil {
			return nil, err
		}
		key, err := config.GetAPIKey(model.Provider)
		if err != nil {
			return nil, err
		}
		if key != "" {
			options["apiKey"] = key
		}
	}
	response, err := stream(ctx, config.Model, ai.NormalizeContext(ai.Context{Messages: MessageValues(llmMessages.Values())}), options)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, errors.New("engine: streamFn returned a nil stream")
	}
	return consumeAssistant(ctx, current, config, emit, response)
}

func requestOptions(config Config) map[string]any {
	options := make(map[string]any, len(config.Options)+2)
	for key, value := range config.Options {
		options[key] = value
	}
	// Pi spreads the loop config into provider options. Keep its serializable
	// model/execution fields too, even though model is also a positional arg.
	options["model"] = config.Model
	if config.ToolExecution != "" {
		options["toolExecution"] = config.ToolExecution
	} else {
		delete(options, "toolExecution")
	}
	if !config.hasReasoning() {
		delete(options, "reasoning")
	} else {
		options["reasoning"] = config.Reasoning
	}
	return options
}

func consumeAssistant(ctx context.Context, current *Context, config Config, emit EventSink, response *ai.AssistantMessageEventStream) (*ai.Message, error) {
	current.assistantStream = response
	addedPartial, hasPartial := false, false
	emitPayload := func(event Event, yield ai.PayloadYield) error {
		event.payloadYield, event.payloadAccess = yield, response.Synchronize
		return emit(event)
	}
	settle := func() (*ai.Message, error) {
		// Result may await provider work or invoke a host hook, so obtain it
		// before entering the payload critical section. Preserve its identity.
		message, err := response.Result(context.WithoutCancel(ctx))
		if err != nil {
			return nil, err
		}
		if message == nil {
			return nil, errors.New("engine: stream has no final assistant message")
		}
		err = response.SynchronizeYielding(func(yield ai.PayloadYield) error {
			level := config.thinkingLevel()
			message.ThinkingLevel = &level
			if addedPartial {
				current.Messages.setLast(message)
			} else {
				current.Messages.Append(message)
				copy := *message
				if err := emitPayload(Event{Type: "message_start", Message: &copy}, yield); err != nil {
					return err
				}
			}
			return emitPayload(Event{Type: "message_end", Message: message}, yield)
		})
		if err != nil {
			return nil, err
		}
		return message, nil
	}
	for {
		// Abort belongs to the provider. Drain its final aborted/error message;
		// cancelling this reader would truncate Pi's lifecycle.
		event, ok, err := response.Next(context.WithoutCancel(ctx))
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		if event.Type == "done" || event.Type == "error" {
			return settle()
		}
		err = response.SynchronizeYielding(func(yield ai.PayloadYield) error {
			switch event.Type {
			case "start":
				if event.Partial == nil {
					return errors.New("engine: start event has no partial message")
				}
				current.Messages.Append(event.Partial)
				addedPartial, hasPartial = true, true
				copy := *event.Partial
				return emitPayload(Event{Type: "message_start", Message: &copy}, yield)
			case "text_start", "text_delta", "text_end", "thinking_start", "thinking_delta", "thinking_end", "toolcall_start", "toolcall_delta", "toolcall_end":
				if hasPartial {
					if event.Partial == nil {
						return errors.New("engine: update event has no partial message")
					}
					current.Messages.setLast(event.Partial)
					copy := *event.Partial
					return emitPayload(Event{Type: "message_update", Message: &copy, AssistantMessageEvent: &event}, yield)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return settle()
}

func validateContinuation(initial Context) error {
	messages := initial.Messages.indexedSnapshot()
	if messages.Len() == 0 {
		return errors.New("Cannot continue: no messages in context")
	}
	tail := messages.Len() - 1
	message := messages.Get(tail)
	if message == nil {
		return jsonjs.PropertyReadError(messages.Has(tail), "context.messages[context.messages.length - 1].role")
	}
	if message.Role == "assistant" {
		return errors.New("Cannot continue from message role: assistant")
	}
	return nil
}
