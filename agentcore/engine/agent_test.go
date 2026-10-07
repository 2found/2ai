package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

type agentAction struct {
	Op       string            `json:"op"`
	Value    string            `json:"value"`
	Input    json.RawMessage   `json:"input"`
	Images   []ai.ContentBlock `json:"images"`
	Message  ai.Message        `json:"message"`
	Messages []ai.Message      `json:"messages"`
	Tools    []toolSpec        `json:"tools"`
	Model    json.RawMessage   `json:"model"`
}

// These older oracles explicitly spread pendingToolCalls into an array.
// Keep that observation adapter separate from direct engine state JSON.
func agentFixtureState(state engine.State) map[string]json.RawMessage {
	raw, err := json.Marshal(state)
	if err != nil {
		panic(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		panic(err)
	}
	fields["pendingToolCalls"], err = json.Marshal(state.PendingToolCalls.Values())
	if err != nil {
		panic(err)
	}
	return fields
}

type agentInput struct {
	MessageMutationRole string  `json:"messageMutationRole"`
	MessageMutationAt   string  `json:"messageMutationAt"`
	ToolEndContent      *string `json:"toolEndContent"`
	ToolEndTerminate    bool    `json:"toolEndTerminate"`
	Name                string  `json:"name"`
	Initial             struct {
		SystemPrompt  string          `json:"systemPrompt"`
		Model         json.RawMessage `json:"model"`
		ThinkingLevel string          `json:"thinkingLevel"`
		Tools         []toolSpec      `json:"tools"`
		Messages      []ai.Message    `json:"messages"`
	} `json:"initial"`
	Options struct {
		SteeringMode    string          `json:"steeringMode"`
		FollowUpMode    string          `json:"followUpMode"`
		SessionID       *string         `json:"sessionId"`
		ThinkingBudgets json.RawMessage `json:"thinkingBudgets"`
		Transport       string          `json:"transport"`
		MaxRetryDelayMS *int            `json:"maxRetryDelayMs"`
	} `json:"options"`
	Actions   []agentAction `json:"actions"`
	Responses []struct {
		Message ai.Message `json:"message"`
		Failure string     `json:"failure"`
		Partial bool       `json:"partial"`
	} `json:"responses"`
	Reactions []struct {
		Event   string        `json:"event"`
		Role    string        `json:"role"`
		Always  bool          `json:"always"`
		Actions []agentAction `json:"actions"`
	} `json:"reactions"`
	ConvertCustom  bool                 `json:"convertCustom"`
	ConvertFailure string               `json:"convertFailure"`
	Decisions      []string             `json:"decisions"`
	LegacyPrepare  bool                 `json:"legacyPrepare"`
	ContextPrepare bool                 `json:"contextPrepare"`
	NextUpdates    []*engine.TurnUpdate `json:"nextUpdates"`
	RequestUpdates []*engine.TurnUpdate `json:"requestUpdates"`
}

func TestPiAgentOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-agent.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Cases          []struct {
			Input    agentInput      `json:"input"`
			Expected json.RawMessage `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 52 {
		t.Fatal("unexpected Agent oracle revision or coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			actual := runAgentFixture(t, tc.Input)
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if difference := firstJSONDifference(got, want, ""); difference != "" {
				t.Fatal(difference)
			}
		})
	}
}

func firstJSONDifference(got, want any, path string) string {
	if reflect.DeepEqual(got, want) {
		return ""
	}
	switch g := got.(type) {
	case map[string]any:
		w, ok := want.(map[string]any)
		if !ok {
			break
		}
		keys := []string{}
		for key := range g {
			keys = append(keys, key)
		}
		for key := range w {
			if _, exists := g[key]; !exists {
				keys = append(keys, key)
			}
		}
		slices.Sort(keys)
		for _, key := range keys {
			gv, gp := g[key]
			wv, wp := w[key]
			if gp != wp {
				return fmt.Sprintf("%s/%s presence: Go=%v Pi=%v", path, key, gp, wp)
			}
			if difference := firstJSONDifference(gv, wv, path+"/"+key); difference != "" {
				return difference
			}
		}
	case []any:
		w, ok := want.([]any)
		if !ok {
			break
		}
		if len(g) != len(w) {
			return fmt.Sprintf("%s length: Go=%d Pi=%d", path, len(g), len(w))
		}
		for i := range g {
			if difference := firstJSONDifference(g[i], w[i], fmt.Sprintf("%s/%d", path, i)); difference != "" {
				return difference
			}
		}
	}
	return fmt.Sprintf("%s\nGo: %#v\nPi: %#v", path, got, want)
}

func runAgentFixture(t *testing.T, input agentInput) []byte {
	t.Helper()
	var mu sync.Mutex
	events, requests, checks, hooks := []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}
	add := func(target *[]json.RawMessage, value any) {
		raw, err := json.Marshal(value)
		if err != nil {
			panic(err)
		}
		mu.Lock()
		*target = append(*target, raw)
		mu.Unlock()
	}
	makeTools := func(specs []toolSpec) []*engine.Tool {
		result := []*engine.Tool{}
		for _, spec := range specs {
			result = append(result, &engine.Tool{Tool: ai.Tool{Name: spec.Name, Description: spec.Description, Parameters: spec.Parameters}, Label: spec.Label,
				Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
					return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: spec.Name}), Details: argumentRef(`{}`)}, nil
				},
			})
		}
		return result
	}
	var agent *engine.Agent
	responseIndex, finishIndex, prepareIndex, requestIndex := 0, 0, 0, 0
	options := engine.AgentOptions{InitialState: engine.InitialState{SystemPrompt: input.Initial.SystemPrompt, Model: input.Initial.Model,
		ThinkingLevel: input.Initial.ThinkingLevel, Tools: makeTools(input.Initial.Tools), Messages: engine.MessagePointers(input.Initial.Messages)},
		AgentConfig: engine.AgentConfig{Config: engine.Config{Now: func() int64 { return 1700000000123 }}, SteeringMode: input.Options.SteeringMode,
			FollowUpMode: input.Options.FollowUpMode, SessionID: input.Options.SessionID, ThinkingBudgets: input.Options.ThinkingBudgets,
			Transport: input.Options.Transport, MaxRetryDelayMS: input.Options.MaxRetryDelayMS},
	}
	options.StreamFn = func(ctx context.Context, model json.RawMessage, transcript ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		add(&requests, map[string]any{"model": model, "context": transcript, "options": options, "signalMatches": ctx == agent.Signal(), "aborted": ctx.Err() != nil})
		if responseIndex >= len(input.Responses) {
			return nil, errors.New("oracle script exhausted")
		}
		scripted := input.Responses[responseIndex]
		responseIndex++
		if scripted.Failure != "" {
			return nil, errors.New(scripted.Failure)
		}
		final := scripted.Message
		stream := ai.NewAssistantMessageEventStream()
		if scripted.Partial {
			partial, updated := final, final
			partial.Content = ai.BlockContent()
			stream.Push(ai.AssistantMessageEvent{Type: "start", Partial: &partial})
			stream.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "done", Partial: &updated})
		}
		if final.StopReason == "error" || final.StopReason == "aborted" {
			stream.Push(ai.AssistantMessageEvent{Type: "error", Reason: final.StopReason, Error: &final})
		} else {
			stream.Push(ai.AssistantMessageEvent{Type: "done", Reason: final.StopReason, Message: &final})
		}
		return stream, nil
	}
	if input.ConvertCustom || input.ConvertFailure != "" {
		options.ConvertToLLM = func(messages *engine.MessageList) (*engine.MessageList, error) {
			if input.ConvertFailure != "" {
				return nil, errors.New(input.ConvertFailure)
			}
			result := engine.MessagePointers(engine.MessageValues(messages.Values()))
			for i := range result {
				if result[i].Role == "custom" {
					result[i].Role = "user"
				}
			}
			return engine.NewList(result...), nil
		}
	}
	if input.Decisions != nil {
		options.FinishTurn = func(ctx context.Context, turn *engine.Turn) (string, error) {
			ids := []string{}
			for _, message := range turn.ToolResults.Values() {
				ids = append(ids, message.ToolCallID)
			}
			add(&hooks, map[string]any{"hook": "finish", "tools": ids, "signalMatches": ctx == agent.Signal()})
			defer func() { finishIndex++ }()
			if finishIndex < len(input.Decisions) {
				return input.Decisions[finishIndex], nil
			}
			return "", nil
		}
	}
	if input.LegacyPrepare {
		options.PrepareNextTurn = func(ctx context.Context) (*engine.TurnUpdate, error) {
			add(&hooks, map[string]any{"hook": "legacyPrepare", "signalMatches": ctx == agent.Signal()})
			defer func() { prepareIndex++ }()
			if prepareIndex < len(input.NextUpdates) {
				return input.NextUpdates[prepareIndex], nil
			}
			return nil, nil
		}
	}
	if input.ContextPrepare {
		options.PrepareNextTurnWithContext = func(ctx context.Context, turn *engine.Turn) (*engine.TurnUpdate, error) {
			add(&hooks, map[string]any{"hook": "contextPrepare", "messageRole": turn.Message.Role, "newMessages": turn.NewMessages.Len(), "signalMatches": ctx == agent.Signal()})
			defer func() { prepareIndex++ }()
			if prepareIndex < len(input.NextUpdates) {
				return input.NextUpdates[prepareIndex], nil
			}
			return nil, nil
		}
	}
	if input.RequestUpdates != nil {
		options.PrepareRequest = func(ctx context.Context, _ engine.Request) (*engine.TurnUpdate, error) {
			add(&hooks, map[string]any{"hook": "request", "signalMatches": ctx == agent.Signal()})
			defer func() { requestIndex++ }()
			if requestIndex < len(input.RequestUpdates) {
				return input.RequestUpdates[requestIndex], nil
			}
			return nil, nil
		}
	}
	var err error
	agent, err = engine.NewAgent(options)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := func(op string, err error) {
		value := map[string]any{"op": op, "state": agentFixtureState(agent.State()), "queued": agent.HasQueuedMessages(), "peek": agent.PeekQueuedMessages(),
			"steeringMode": agent.SteeringMode(), "followUpMode": agent.FollowUpMode(), "signalPresent": agent.Signal() != nil}
		if err != nil {
			value["error"] = err.Error()
		}
		add(&checks, value)
	}
	var act func(agentAction, bool) error
	act = func(action agentAction, nested bool) error {
		var err error
		switch action.Op {
		case "prompt":
			var value any
			switch action.Input[0] {
			case '"':
				var text string
				err = json.Unmarshal(action.Input, &text)
				value = text
			case '[':
				var messages []ai.Message
				err = json.Unmarshal(action.Input, &messages)
				value = messages
			default:
				var message ai.Message
				err = json.Unmarshal(action.Input, &message)
				value = message
			}
			if err == nil {
				err = agent.Prompt(context.Background(), value, ai.BlockContent(action.Images...).Blocks.Values()...)
			}
		case "continue":
			err = agent.Continue(context.Background())
		case "reset":
			err = agent.Reset()
		case "abort":
			agent.Abort()
		case "steer":
			agent.Steer(&action.Message)
		case "followUp":
			agent.FollowUp(&action.Message)
		case "steeringMode":
			agent.SetSteeringMode(action.Value)
		case "followUpMode":
			agent.SetFollowUpMode(action.Value)
		case "clearSteering":
			agent.ClearSteeringQueue()
		case "clearFollowUp":
			agent.ClearFollowUpQueue()
		case "clearAll":
			agent.ClearAllQueues()
		case "model":
			agent.SetModel(action.Model)
		case "thinking":
			agent.SetThinkingLevel(action.Value)
		case "tools":
			agent.SetTools(makeTools(action.Tools))
		case "messages":
			agent.SetMessages(engine.MessagePointers(action.Messages))
		case "throw":
			return errors.New(action.Value)
		default:
			panic("unknown action " + action.Op)
		}
		name := action.Op
		if nested {
			name = "listener:" + name
		}
		checkpoint(name, err)
		return nil
	}
	fired := map[int]bool{}
	var retainedMessage *ai.Message
	messageMutated := false
	agent.Subscribe(&engine.Listener{Handle: func(ctx context.Context, event engine.Event) error {
		add(&events, map[string]any{"event": event, "state": agentFixtureState(agent.State()), "signalMatches": ctx == agent.Signal(), "aborted": ctx.Err() != nil})
		if event.Type == "message_end" && event.Message.Role == input.MessageMutationRole && retainedMessage == nil {
			retainedMessage = event.Message
		}
		if !messageMutated && event.Type == input.MessageMutationAt && (event.Type == "agent_end" || (event.Message != nil && event.Message.Role == input.MessageMutationRole)) {
			message := event.Message
			if event.Type == "agent_end" {
				message = retainedMessage
			}
			if message.Role == "user" {
				message.Content = ai.TextContent("callback revision")
			} else {
				message.Content = ai.BlockContent(ai.ContentBlock{Type: "text", Text: "callback revision"})
			}
			message.Timestamp = 1700000000123 + 9
			messageMutated = true
			checkpoint("message mutation", nil)
		}
		if event.Type == "tool_execution_end" && input.ToolEndContent != nil {
			event.Result.Content = ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: *input.ToolEndContent})
			event.Result.Terminate = &input.ToolEndTerminate
		}
		for i, reaction := range input.Reactions {
			if event.Type != reaction.Event || (reaction.Role != "" && (event.Message == nil || event.Message.Role != reaction.Role)) || (fired[i] && !reaction.Always) {
				continue
			}
			fired[i] = true
			for _, action := range reaction.Actions {
				if err := act(action, true); err != nil {
					return err
				}
			}
		}
		return nil
	}})
	checkpoint("initial", nil)
	for _, action := range input.Actions {
		if err := act(action, false); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := json.Marshal(map[string]any{"events": events, "requests": requests, "checks": checks, "hooks": hooks})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
