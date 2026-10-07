package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore/engine"
	"github.com/2found/2ai/ai"
)

func TestPiFailureValues(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-failures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Model          json.RawMessage
		Cases          []struct {
			Input struct {
				Name, Kind, NativeType string
				Value                  any
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 34 {
		t.Fatal("unexpected failure oracle revision or coverage")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Input.Name, func(t *testing.T) {
			failure := test.Input.Value
			switch test.Input.Kind {
			case "error":
				failure = errors.New(failure.(string))
			case "number":
				failure = ai.ParseJSNumber(failure.(string))
			case "self-cycle":
				value := []any{float64(1), nil, float64(2)}
				value[1] = value
				failure = value
			case "mutual-cycle":
				a := []any{float64(1), nil}
				b := []any{float64(2), a}
				a[1] = b
				failure = a
			case "empty-cycle":
				value := make([]any, 1)
				value[0] = value
				failure = value
			case "shared-array":
				child := []any{float64(1), float64(2)}
				failure = []any{child, child}
			}
			switch test.Input.NativeType {
			case "strings":
				failure = []string{"a", "b"}
			case "bools":
				failure = []bool{true, false}
			case "integers":
				failure = []int64{1, 2, 3}
			case "nested":
				failure = [][]int{{1, 2}, {3}}
			case "array":
				failure = [2]string{"a", "b"}
			case "bytes":
				failure = []byte{65, 66}
			case "nil-slice":
				failure = []string(nil)
			case "map":
				failure = map[string]int{"code": 7}
			case "float32":
				failure = float32(failure.(float64))
			case "int64":
				value, err := strconv.ParseInt(test.Input.Value.(string), 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				failure = value
			case "uint64":
				value, err := strconv.ParseUint(test.Input.Value.(string), 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				failure = value
			case "named-cycle":
				type array []any
				value := array{float64(1), nil, float64(2)}
				value[1] = value
				failure = value
			}
			agent, err := engine.NewAgent(engine.AgentOptions{InitialState: engine.InitialState{Model: fixture.Model}, AgentConfig: engine.AgentConfig{
				Config: engine.Config{Now: func() int64 { return 1700000000123 }},
				StreamFn: func(context.Context, json.RawMessage, ai.TranscriptContext, map[string]any) (*ai.AssistantMessageEventStream, error) {
					panic(failure)
				},
			}})
			if err != nil {
				t.Fatal(err)
			}
			events := []json.RawMessage{}
			agent.Subscribe(&engine.Listener{Handle: func(_ context.Context, event engine.Event) error {
				raw, err := json.Marshal(event)
				events = append(events, raw)
				return err
			}})
			if err := agent.Prompt(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			outcomes := map[string]engine.ToolOutcome{}
			for _, stage := range []string{"prepare", "before", "execute", "after"} {
				tool := &engine.Tool{Tool: ai.Tool{Name: "tool", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(context.Context, string, any, func(*engine.ToolResult)) (*engine.ToolResult, error) {
					if stage == "execute" {
						panic(failure)
					}
					return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)}, nil
				}}
				hooks := engine.ToolHooks{}
				if stage == "prepare" {
					tool.PrepareArguments = func(json.RawMessage) (json.RawMessage, error) { panic(failure) }
				}
				if stage == "before" {
					hooks.Before = func(context.Context, *engine.BeforeToolCall) (*engine.BeforeToolResult, error) { panic(failure) }
				}
				if stage == "after" {
					hooks.After = func(context.Context, engine.AfterToolCall) (*engine.AfterToolResult, error) { panic(failure) }
				}
				call := ai.ContentBlock{Type: "toolCall", ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}
				outcome, err := engine.RunToolCall(context.Background(), &call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{Role: "assistant", Content: ai.BlockContent(call)}, &engine.Context{}, hooks, nil)
				if err != nil {
					t.Fatal(err)
				}
				outcomes[stage] = outcome
			}
			actual, err := json.Marshal(map[string]any{"events": events, "state": agentFixtureState(agent.State()), "tools": outcomes})
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(test.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if difference := firstJSONDifference(got, want, ""); difference != "" {
				t.Fatal(difference)
			}
		})
	}
}

func TestPiToolUpdateFailures(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-failures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Updates        []struct {
			Mode     string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Updates) < 3 {
		t.Fatal("unexpected update oracle revision or coverage")
	}
	for _, test := range fixture.Updates {
		t.Run(test.Mode, func(t *testing.T) {
			failure := errors.New("update failed")
			continued, caught, after := false, false, false
			tool := &engine.Tool{Tool: ai.Tool{Name: "tool", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				invoke := func() { update(&engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)}) }
				if test.Mode == "caught" {
					func() { defer func() { caught = recover() == failure }(); invoke() }()
				} else {
					invoke()
				}
				continued = true
				return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)}, nil
			}}
			call := ai.ContentBlock{Type: "toolCall", ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}
			outcome, err := engine.RunToolCall(context.Background(), &call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{Role: "assistant", Content: ai.BlockContent(call)}, &engine.Context{}, engine.ToolHooks{After: func(context.Context, engine.AfterToolCall) (*engine.AfterToolResult, error) {
				after = true
				return nil, nil
			}}, func(*engine.ToolResult) error {
				if test.Mode == "async" {
					return failure
				}
				panic(failure)
			})
			result := map[string]any{"continued": continued, "caught": caught, "after": after}
			if err != nil {
				result["error"], result["sameError"] = err.Error(), err == failure
			} else {
				result["outcome"] = outcome
			}
			actual, err := json.Marshal(result)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(test.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if difference := firstJSONDifference(got, want, ""); difference != "" {
				t.Fatal(difference)
			}
		})
	}
}

func TestPiPendingToolUpdates(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-failures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		PendingUpdates []struct {
			Mode     string
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.PendingUpdates) != 7 {
		t.Fatal("missing pending-update oracle cases")
	}
	for _, test := range fixture.PendingUpdates {
		t.Run(test.Mode, func(t *testing.T) {
			firstError, secondError, toolError := errors.New("first update failed"), errors.New("second update failed"), errors.New("tool failed")
			entered, release, firstDone, executed := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
			late := strings.HasPrefix(test.Mode, "late-")
			secondEntered, rejectSecond, secondDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			defer unblock()
			var after atomic.Bool
			tool := &engine.Tool{Tool: ai.Tool{Name: "tool", Parameters: json.RawMessage(`{"type":"object"}`)}, Execute: func(_ context.Context, _ string, _ any, update func(*engine.ToolResult)) (*engine.ToolResult, error) {
				defer close(executed)
				go func() { defer close(firstDone); update(&engine.ToolResult{Details: argumentRef(`{"id":"first"}`)}) }()
				<-entered
				if late {
					go func() {
						defer close(secondDone)
						update(&engine.ToolResult{Details: argumentRef(`{"id":"second"}`)})
					}()
					<-secondEntered
				} else if test.Mode == "update-error" || test.Mode == "tool-and-update-error" || test.Mode == "already-rejected-order" {
					update(&engine.ToolResult{Details: argumentRef(`{"id":"second"}`)})
				}
				if test.Mode == "already-rejected-order" {
					unblock()
					<-firstDone
				}
				if test.Mode == "tool-error" || test.Mode == "tool-and-update-error" || test.Mode == "late-tool-and-update-error" {
					return &engine.ToolResult{}, toolError
				}
				return &engine.ToolResult{Content: ai.NewBlockList(), Details: argumentRef(`{}`)}, nil
			}}
			type completion struct {
				outcome engine.ToolOutcome
				err     error
			}
			done := make(chan completion, 1)
			go func() {
				call := ai.ContentBlock{Type: "toolCall", ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}
				outcome, err := engine.RunToolCall(context.Background(), &call, engine.NewList([]*engine.Tool{tool}...), &ai.Message{Role: "assistant", Content: ai.BlockContent(call)}, &engine.Context{}, engine.ToolHooks{After: func(context.Context, engine.AfterToolCall) (*engine.AfterToolResult, error) {
					after.Store(true)
					return nil, nil
				}}, func(result *engine.ToolResult) error {
					if argumentJSON(t, result.Details) == `{"id":"first"}` {
						close(entered)
						<-release
						if test.Mode == "already-rejected-order" {
							return firstError
						}
						return nil
					}
					if late {
						close(secondEntered)
						<-rejectSecond
					}
					return secondError
				})
				done <- completion{outcome, err}
			}()
			<-executed
			wait := time.Second
			if test.Mode == "success" || test.Mode == "tool-error" {
				wait = 50 * time.Millisecond
			}
			var completed completion
			settled := false
			var beforeRejection map[string]any
			if late {
				select {
				case completed = <-done:
					settled = true
				default:
				}
				beforeRejection = map[string]any{"settled": settled, "after": after.Load()}
				close(rejectSecond)
			}
			if !settled {
				select {
				case completed = <-done:
					settled = true
				case <-time.After(wait):
				}
			}
			beforeRelease := map[string]any{"settled": settled, "after": after.Load()}
			unblock()
			if !settled {
				select {
				case completed = <-done:
				case <-time.After(time.Second):
					t.Fatal("call did not settle after releasing updates")
				}
			}
			<-firstDone
			if late {
				<-secondDone
			}
			result := map[string]any{"beforeRelease": beforeRelease, "after": after.Load()}
			if late {
				result["beforeRejection"] = beforeRejection
			}
			if completed.err != nil {
				result["error"] = completed.err.Error()
				if completed.err != firstError && completed.err != secondError {
					t.Fatal("lost update error identity")
				}
			} else {
				result["outcome"] = completed.outcome
			}
			actual, err := json.Marshal(result)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(actual, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(test.Expected, &want); err != nil {
				t.Fatal(err)
			}
			if difference := firstJSONDifference(got, want, ""); difference != "" {
				t.Fatal(difference)
			}
		})
	}
}
