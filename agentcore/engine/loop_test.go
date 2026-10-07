package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

type toolSpec struct {
	RawParameters      string                     `json:"rawParameters"`
	AfterContextResult *engine.ToolResult         `json:"afterContextResult"`
	Name               string                     `json:"name"`
	Label              string                     `json:"label"`
	Description        string                     `json:"description"`
	Parameters         json.RawMessage            `json:"parameters"`
	ExecutionMode      string                     `json:"executionMode"`
	Prepare            json.RawMessage            `json:"prepare"`
	Result             *engine.ToolResult         `json:"result"`
	Updates            []engine.ToolResult        `json:"updates"`
	LateUpdate         *engine.ToolResult         `json:"lateUpdate"`
	Failure            string                     `json:"failure"`
	Before             *engine.BeforeToolResult   `json:"before"`
	After              *engine.AfterToolResult    `json:"after"`
	AfterFailure       string                     `json:"afterFailure"`
	AfterMutation      map[string]json.RawMessage `json:"afterMutation"`
	AfterDeleteResult  []string                   `json:"afterDeleteResult"`
	BeforeFailure      string                     `json:"beforeFailure"`
	BeforeArgs         map[string]json.RawMessage `json:"beforeArgs"`
	BeforeDeleteArgs   []string                   `json:"beforeDeleteArgs"`
}

type loopInput struct {
	MessageMutationAt   string                     `json:"messageMutationAt"`
	MessageMutationRole string                     `json:"messageMutationRole"`
	MessageMutation     map[string]json.RawMessage `json:"messageMutation"`
	RetainedEndAt       string                     `json:"retainedEndAt"`
	RetainedEndTrigger  string                     `json:"retainedEndTrigger"`
	RetainedEndTarget   string                     `json:"retainedEndTarget"`
	RetainedEndMutation map[string]json.RawMessage `json:"retainedEndMutation"`
	RetainedEndDelete   []string                   `json:"retainedEndDelete"`
	EndMutation         map[string]json.RawMessage `json:"endMutation"`
	EndDeleteResult     []string                   `json:"endDeleteResult"`
	EndReplacement      *engine.ToolResult         `json:"endReplacement"`
	EndIsError          *bool                      `json:"endIsError"`
	RetainAfterResults  bool                       `json:"retainAfterResults"`
	Name                string                     `json:"name"`
	Prompts             []*ai.Message              `json:"prompts"`
	Messages            []*ai.Message              `json:"messages"`
	Tools               []toolSpec                 `json:"tools"`
	Responses           []ai.Message               `json:"responses"`
	Resume              bool                       `json:"resume"`
	ConvertCustom       bool                       `json:"convertCustom"`
	TransformPrefix     *ai.Message                `json:"transformPrefix"`
	StreamPartial       bool                       `json:"streamPartial"`
	OmitTerminal        bool                       `json:"omitTerminal"`
	Decisions           []string                   `json:"decisions"`
	Steering            [][]*ai.Message            `json:"steering"`
	FollowUps           [][]*ai.Message            `json:"followUps"`
	NextUpdates         []*engine.TurnUpdate       `json:"nextUpdates"`
	RequestUpdates      []*engine.TurnUpdate       `json:"requestUpdates"`
	Reasoning           string                     `json:"reasoning"`
	APIKeys             []string                   `json:"apiKeys"`
	APIKey              *string                    `json:"apiKey"`
	RecordHooks         bool                       `json:"recordHooks"`
	Mode                string                     `json:"mode"`
	WaitForSecond       bool                       `json:"waitForSecond"`
	CancelBefore        string                     `json:"cancelBefore"`
	OmitStream          bool                       `json:"omitStream"`
	FailEvent           string                     `json:"failEvent"`
}

func TestPiLoopOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-loop.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Cases          []struct {
			Input    loopInput       `json:"input"`
			Expected json.RawMessage `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 143 {
		t.Fatal("unexpected oracle revision or coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			actual := runFixture(t, tc.Input)
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(tc.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				// Report only differing top-level fields; the fixture preserves every
				// event and message for inspection without truncating useful failures.
				g, w := got.(map[string]any), want.(map[string]any)
				for _, key := range []string{"events", "requests", "hooks", "executed", "messages", "error", "afterResults"} {
					if !reflect.DeepEqual(g[key], w[key]) {
						gb, _ := json.Marshal(g[key])
						wb, _ := json.Marshal(w[key])
						t.Errorf("%s\nGo: %s\nPi: %s", key, gb, wb)
					}
				}
			}
		})
	}
}

func runFixture(t *testing.T, input loopInput) []byte {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var mu sync.Mutex
	events, requests, hooks := []json.RawMessage{}, []json.RawMessage{}, []json.RawMessage{}
	executed := []string{}
	afterResults := []*engine.ToolResult{}
	late := []func(){}
	cloneResult := func(value *engine.ToolResult) *engine.ToolResult {
		raw, err := json.Marshal(value)
		if err != nil {
			panic(err)
		}
		var result *engine.ToolResult
		if err := json.Unmarshal(raw, &result); err != nil {
			panic(err)
		}
		return result
	}
	add := func(target *[]json.RawMessage, value any) {
		raw, err := json.Marshal(value)
		if err != nil {
			panic(err)
		}
		mu.Lock()
		*target = append(*target, raw)
		mu.Unlock()
	}
	secondDone := make(chan struct{})
	tools := make([]*engine.Tool, 0, len(input.Tools))
	for _, spec := range input.Tools {
		if spec.RawParameters != "" {
			spec.Parameters = json.RawMessage(spec.RawParameters)
		}
		tool := &engine.Tool{Tool: ai.Tool{Name: spec.Name, Description: spec.Description, Parameters: spec.Parameters}, ExecutionMode: spec.ExecutionMode}
		if spec.Prepare != nil {
			tool.PrepareArguments = func(json.RawMessage) (json.RawMessage, error) { return slices.Clone(spec.Prepare), nil }
		}
		tool.Execute = func(_ context.Context, id string, args any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
			mu.Lock()
			executed = append(executed, id)
			mu.Unlock()
			for _, partial := range spec.Updates {
				update(cloneResult(&partial))
			}
			if spec.LateUpdate != nil {
				mu.Lock()
				late = append(late, func() { update(cloneResult(spec.LateUpdate)) })
				mu.Unlock()
			}
			if input.WaitForSecond && id == "one" {
				<-secondDone
			}
			if spec.Failure != "" {
				return &engine.ToolResult{}, errors.New(spec.Failure)
			}
			if spec.Result != nil {
				return cloneResult(spec.Result), nil
			}
			return &engine.ToolResult{Content: ai.NewBlockList(&ai.ContentBlock{Type: "text", Text: argumentJSON(t, args)}), Details: argumentRef(`{}`)}, nil
		}
		tools = append(tools, tool)
	}
	model := json.RawMessage(`{"id":"test-model","provider":"test-provider","api":"test-api","input":["text"]}`)
	config := engine.Config{Model: model, Reasoning: input.Reasoning, Options: map[string]any{}, ToolExecution: input.Mode, Now: func() int64 { return 1700000000123 }}
	if input.APIKey != nil {
		config.Options["apiKey"] = *input.APIKey
	}
	config.ConvertToLLM = func(messages *engine.MessageList) (*engine.MessageList, error) {
		roles := []string{}
		result := engine.MessagePointers(engine.MessageValues(messages.Values()))
		for i, message := range messages.Values() {
			roles = append(roles, message.Role)
			if input.ConvertCustom && message.Role == "custom" {
				result[i].Role = "user"
			}
		}
		add(&hooks, map[string]any{"hook": "convert", "roles": roles})
		return engine.NewList(result...), nil
	}
	if input.TransformPrefix != nil {
		config.TransformContext = func(_ context.Context, messages *engine.MessageList) (*engine.MessageList, error) {
			add(&hooks, map[string]any{"hook": "transform"})
			return engine.NewList(append([]*ai.Message{input.TransformPrefix}, messages.Values()...)...), nil
		}
	}
	steering, follow, finish, next, request, key := 0, 0, 0, 0, 0, 0
	config.GetSteeringMessages = func() (*engine.MessageList, error) {
		add(&hooks, map[string]any{"hook": "steering"})
		defer func() { steering++ }()
		if steering < len(input.Steering) {
			return engine.NewList(input.Steering[steering]...), nil
		}
		return nil, nil
	}
	config.GetFollowUpMessages = func() (*engine.MessageList, error) {
		add(&hooks, map[string]any{"hook": "followUp"})
		defer func() { follow++ }()
		if follow < len(input.FollowUps) {
			return engine.NewList(input.FollowUps[follow]...), nil
		}
		return nil, nil
	}
	config.FinishTurn = func(_ context.Context, turn *engine.Turn) (string, error) {
		ids := []string{}
		for _, result := range turn.ToolResults.Values() {
			ids = append(ids, result.ToolCallID)
		}
		add(&hooks, map[string]any{"hook": "finish", "stopReason": turn.Message.StopReason, "tools": ids})
		defer func() { finish++ }()
		if finish < len(input.Decisions) {
			return input.Decisions[finish], nil
		}
		return "", nil
	}
	config.PrepareNextTurn = func(*engine.Turn) (*engine.TurnUpdate, error) {
		add(&hooks, map[string]any{"hook": "next"})
		defer func() { next++ }()
		if next < len(input.NextUpdates) {
			return input.NextUpdates[next], nil
		}
		return nil, nil
	}
	config.PrepareRequest = func(context.Context, engine.Request) (*engine.TurnUpdate, error) {
		add(&hooks, map[string]any{"hook": "request"})
		defer func() { request++ }()
		if request < len(input.RequestUpdates) {
			return input.RequestUpdates[request], nil
		}
		return nil, nil
	}
	if input.APIKeys != nil {
		config.GetAPIKey = func(provider string) (string, error) {
			add(&hooks, map[string]any{"hook": "key", "provider": provider})
			defer func() { key++ }()
			if key < len(input.APIKeys) {
				return input.APIKeys[key], nil
			}
			return "", nil
		}
	}
	find := func(name string) toolSpec {
		for _, spec := range input.Tools {
			if spec.Name == name {
				return spec
			}
		}
		return toolSpec{}
	}
	if input.RecordHooks {
		config.Before = func(_ context.Context, call *engine.BeforeToolCall) (*engine.BeforeToolResult, error) {
			add(&hooks, map[string]any{"hook": "before", "id": call.ToolCall.ID, "raw": call.ToolCall.Arguments, "args": call.Args})
			if input.CancelBefore == call.ToolCall.ID {
				cancel()
			}
			spec := find(call.ToolCall.Name)
			if spec.BeforeArgs != nil || spec.BeforeDeleteArgs != nil {
				args := call.Args.(*engine.Object)
				for key, value := range spec.BeforeArgs {
					args.Set(key, argumentRef(string(value)))
				}
				for _, key := range spec.BeforeDeleteArgs {
					args.Delete(key)
				}
			}
			if spec.BeforeFailure != "" {
				return nil, errors.New(spec.BeforeFailure)
			}
			return spec.Before, nil
		}
		config.After = func(_ context.Context, call engine.AfterToolCall) (*engine.AfterToolResult, error) {
			spec := find(call.ToolCall.Name)
			if input.RetainAfterResults {
				mu.Lock()
				afterResults = append(afterResults, call.Result)
				mu.Unlock()
			}
			entry := map[string]any{"hook": "after", "id": call.ToolCall.ID, "isError": call.IsError}
			if spec.BeforeArgs != nil || spec.BeforeDeleteArgs != nil {
				entry["args"] = call.Args
			}
			add(&hooks, entry)
			if spec.AfterMutation != nil || spec.AfterDeleteResult != nil {
				raw, err := json.Marshal(call.Result)
				if err != nil {
					return nil, err
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					return nil, err
				}
				for key, value := range spec.AfterMutation {
					fields[key] = value
				}
				for _, key := range spec.AfterDeleteResult {
					delete(fields, key)
				}
				raw, err = json.Marshal(fields)
				if err != nil {
					return nil, err
				}
				if err := json.Unmarshal(raw, &call.Result); err != nil {
					return nil, err
				}
			}
			if spec.AfterContextResult != nil {
				call.Result = spec.AfterContextResult
			}
			if spec.AfterFailure != "" {
				return nil, errors.New(spec.AfterFailure)
			}
			return spec.After, nil
		}
	}
	responseIndex := 0
	stream := func(_ context.Context, model json.RawMessage, transcript ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		if responseIndex >= len(input.Responses) {
			return nil, errors.New("oracle script exhausted")
		}
		add(&requests, map[string]any{"model": model, "context": transcript, "options": options})
		response := input.Responses[responseIndex]
		responseIndex++
		result := ai.NewAssistantMessageEventStream()
		if input.StreamPartial {
			partial := response
			partial.Content = ai.BlockContent()
			updated := response
			result.Push(ai.AssistantMessageEvent{Type: "start", Partial: &partial})
			result.Push(ai.AssistantMessageEvent{Type: "text_delta", ContentIndex: 0, Delta: "done", Partial: &updated})
		}
		if input.OmitTerminal {
			result.End(&response)
		} else if response.StopReason == "error" || response.StopReason == "aborted" {
			result.Push(ai.AssistantMessageEvent{Type: "error", Reason: response.StopReason, Error: &response})
		} else {
			result.Push(ai.AssistantMessageEvent{Type: "done", Reason: response.StopReason, Message: &response})
		}
		return result, nil
	}
	endResults := map[string]*engine.ToolResult{}
	var endMu sync.Mutex
	messageMutated := false
	sink := func(event engine.Event) error {
		add(&events, event)
		endMu.Lock()
		defer endMu.Unlock()
		if !messageMutated && event.Type == input.MessageMutationAt && event.Message != nil && event.Message.Role == input.MessageMutationRole {
			raw, err := json.Marshal(event.Message)
			if err != nil {
				return err
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				return err
			}
			for key, value := range input.MessageMutation {
				fields[key] = value
			}
			raw, err = json.Marshal(fields)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(raw, event.Message); err != nil {
				return err
			}
			messageMutated = true
		}
		if event.Type == input.FailEvent {
			return errors.New("sink failed")
		}
		if event.Type == "tool_execution_end" {
			if input.EndMutation != nil || input.EndDeleteResult != nil {
				if err := mutateFixtureToolResult(event.Result, input.EndMutation, input.EndDeleteResult); err != nil {
					return err
				}
			}
			if input.EndReplacement != nil {
				event.Result = input.EndReplacement
			}
			if input.EndIsError != nil {
				event.IsError = *input.EndIsError
			}
			endResults[event.ToolCallID] = event.Result
		}
		id := event.ToolCallID
		if event.Message != nil {
			id = event.Message.ToolCallID
		}
		if event.Type == input.RetainedEndAt && id == input.RetainedEndTrigger {
			retained := endResults[input.RetainedEndTarget]
			if retained == nil {
				return errors.New("missing retained tool result")
			}
			if err := mutateFixtureToolResult(retained, input.RetainedEndMutation, input.RetainedEndDelete); err != nil {
				return err
			}
		}
		if event.Type == "tool_execution_end" && event.ToolCallID == "two" && input.WaitForSecond {
			close(secondDone)
		}
		return nil
	}
	var messages *engine.MessageList
	var err error
	current := engine.Context{Messages: engine.NewList(input.Messages...), Tools: engine.NewList(tools...)}
	if input.OmitStream {
		stream = nil
	}
	if input.Resume {
		messages, err = engine.Continue(ctx, current, config, sink, stream)
	} else {
		messages, err = engine.Run(ctx, engine.NewList(input.Prompts...), current, config, sink, stream)
	}
	for _, update := range late {
		update()
	}
	slices.Sort(executed)
	result := map[string]any{"events": events, "requests": requests, "hooks": hooks, "executed": executed}
	if input.RetainAfterResults {
		result["afterResults"] = afterResults
	}
	if err != nil {
		result["error"] = err.Error()
	} else {
		result["messages"] = messages
	}
	raw, marshalErr := json.Marshal(result)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	return raw
}

func mutateFixtureToolResult(result *engine.ToolResult, mutations map[string]json.RawMessage, deleted []string) error {
	raw, err := json.Marshal(result)
	if err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	for key, value := range mutations {
		fields[key] = value
	}
	for _, key := range deleted {
		delete(fields, key)
	}
	raw, err = json.Marshal(fields)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, result)
}
