package subagent_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/agentcore/plugins/jobs"
	"github.com/2found/2ai/agentcore/plugins/subagent"
	"github.com/2found/2ai/ai"
)

// gateTool blocks until released: it parks a child between two of its steps so
// a message can arrive at a known moment.
type gateTool struct {
	echoTool
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func newGate() *gateTool {
	return &gateTool{echoTool: echoTool{name: "gate"}, entered: make(chan struct{}), release: make(chan struct{})}
}

// waitEntered waits for the child to reach the gate, or gives up.
func (g *gateTool) waitEntered() {
	select {
	case <-g.entered:
	case <-time.After(10 * time.Second):
	}
}

func (g *gateTool) Run(ctx context.Context, args string) (string, error) {
	g.once.Do(func() { close(g.entered) })
	select {
	case <-g.release:
		return "gate open", nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

const steerMarker = "child-task: tidy the repo"

// steerScript wires a parent and its async child. The parent spawns the child,
// then runs parentTurns in order; the child calls the gate, then answers. seen
// receives the messages of each child request, as text.
type steerScript struct {
	mu        sync.Mutex
	childSeen [][]string
	parent    atomic.Int32
	child     atomic.Int32
	childTool [][]string
	// parentResults are the tool results the parent read at its latest request.
	parentResults []string
}

func (s *steerScript) provider(parent func(turn int32) ai.Message, childAnswer string) *ai.FallbackProvider {
	return nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		isChild := false
		var texts []string
		for _, m := range view.Messages() {
			if m.Role == "user" && m.Content.Text != nil {
				texts = append(texts, *m.Content.Text)
				if strings.Contains(*m.Content.Text, steerMarker) {
					isChild = true
				}
			}
		}
		if !isChild {
			var results []string
			for _, m := range view.Messages() {
				if m.Role == "toolResult" {
					raw, _ := json.Marshal(m)
					results = append(results, string(raw))
				}
			}
			s.mu.Lock()
			s.parentResults = results
			s.mu.Unlock()
			return ai.ScriptedStream(parent(s.parent.Add(1)))(ctx, model, view, options)
		}
		var tools []string
		for _, tool := range ai.GetCurrentTools(view.Messages()) {
			tools = append(tools, tool.Name)
		}
		s.mu.Lock()
		s.childSeen = append(s.childSeen, texts)
		s.childTool = append(s.childTool, tools)
		s.mu.Unlock()
		if s.child.Add(1) == 1 {
			return ai.ScriptedStream(nativeCall("c1", "gate", `{"text":"x"}`))(ctx, model, view, options)
		}
		return ai.ScriptedStream(nativeAnswer(childAnswer))(ctx, model, view, options)
	})
}

func countIn(texts []string, needle string) int {
	n := 0
	for _, t := range texts {
		n += strings.Count(t, needle)
	}
	return n
}

func steerAgent(t *testing.T, p *ai.FallbackProvider, gate *gateTool, fork subagent.ForkRunner) *agentcore.Agent {
	t.Helper()
	a, err := agentcore.New(agentcore.Config{
		NativeProvider: p, Model: "faux-1",
		Tools:  agentcore.NewToolSet(gate),
		Policy: agentcore.NewAllowList("gate", subagent.ToolSpawnSubagent),
		Extensions: []agentcore.ExtensionFactory{
			&subagent.Plugin{AllowAsync: true, RunFork: fork},
			jobs.Plugin{AllowSend: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// A message the parent sends reaches the running child at its next step as a
// user message, exactly once, and is persisted in the child's own transcript so
// a resumed child still has it. The child keeps its depth cap: it is not offered
// spawn_subagent because it can be steered.
func TestJobSendReachesTheRunningChildOnce(t *testing.T) {
	gate := newGate()
	script := &steerScript{}
	spawn := `{"task":"` + steerMarker + `","async":true}`
	provider := script.provider(func(turn int32) ai.Message {
		switch turn {
		case 1:
			return nativeCall("p1", subagent.ToolSpawnSubagent, spawn)
		case 2:
			gate.waitEntered() // the child is mid-flight
			return nativeCall("p2", "job_status", `{"id":"job_1"}`)
		case 3:
			return nativeCall("p3", "job_send", `{"id":"job_1","message":"also skip the vendor directory"}`)
		case 4:
			close(gate.release)
			return nativeCall("p4", "job_wait", `{"id":"job_1","timeout_seconds":20}`)
		}
		return nativeAnswer("parent done")
	}, "child done")

	var mu sync.Mutex
	var childState json.RawMessage
	fork := func(ctx context.Context, child *agentcore.Agent, req subagent.ForkRequest, sink agentcore.StreamSink) (agentcore.RunResult, error) {
		res, err := child.RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: req.Prompt}}, Task: req.Task, Sink: sink})
		mu.Lock()
		childState = res.NativeState
		mu.Unlock()
		return res, err
	}
	a := steerAgent(t, provider, gate, fork)
	res, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "parent work"}}})
	if err != nil || res.StopReason != "stop" || res.Final != "parent done" {
		t.Fatalf("run = %+v %v", res, err)
	}
	script.mu.Lock()
	seen, tools, results := script.childSeen, script.childTool, strings.Join(script.parentResults, "\n")
	script.mu.Unlock()
	// While it ran, the parent could see which step the child was on.
	if !strings.Contains(results, "running gate") || !strings.Contains(results, "sub-agent] started") {
		t.Fatalf("job_status of the running child showed no progress: %s", results)
	}
	if len(seen) != 2 {
		t.Fatalf("child made %d requests, want 2", len(seen))
	}
	if countIn(seen[0], "skip the vendor") != 0 {
		t.Fatal("the message reached the child before it was sent")
	}
	if got := countIn(seen[1], "Message from the parent agent"); got != 1 || countIn(seen[1], "also skip the vendor directory") != 1 {
		t.Fatalf("child's second request holds the message %d times: %q", got, seen[1])
	}
	for _, names := range tools {
		if strings.Contains(strings.Join(names, ","), subagent.ToolSpawnSubagent) {
			t.Fatalf("steering gave the child a spawn tool: %v", names)
		}
	}
	if !strings.Contains(res.Messages[len(res.Messages)-1].Content, "parent done") {
		t.Fatalf("messages = %+v", res.Messages)
	}

	// Persisted: a child resumed from its checkpoint still reads the message, once.
	mu.Lock()
	state := childState
	mu.Unlock()
	if !strings.Contains(string(state), "also skip the vendor directory") {
		t.Fatal("the steering message is not in the child's checkpoint")
	}
	var resumedSeen []string
	resumeProvider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		for _, m := range view.Messages() {
			if m.Role == "user" && m.Content.Text != nil {
				resumedSeen = append(resumedSeen, *m.Content.Text)
			}
		}
		return ai.ScriptedStream(nativeAnswer("resumed"))(ctx, model, view, options)
	})
	again := steerAgent(t, resumeProvider, newGate(), nil)
	if _, err := again.RunNative(context.Background(), agentcore.NativeRun{State: state, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "carry on"}}}); err != nil {
		t.Fatal(err)
	}
	if countIn(resumedSeen, "also skip the vendor directory") != 1 {
		t.Fatalf("resumed child sees the message %d times", countIn(resumedSeen, "also skip the vendor directory"))
	}
}

// A message queued while the child is parked inside a tool call is read at the
// step that follows, and shapes the child's answer.
func TestMessageSentWhileTheChildIsInAToolShapesItsAnswer(t *testing.T) {
	gate := newGate()
	script := &steerScript{}
	spawn := `{"task":"` + steerMarker + `","async":true}`
	provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		var userTexts []string
		for _, m := range view.Messages() {
			if m.Role == "user" && m.Content.Text != nil {
				userTexts = append(userTexts, *m.Content.Text)
			}
		}
		if !strings.Contains(strings.Join(userTexts, "\n"), steerMarker) {
			switch script.parent.Add(1) {
			case 1:
				return ai.ScriptedStream(nativeCall("p1", subagent.ToolSpawnSubagent, spawn))(ctx, model, view, options)
			case 2:
				gate.waitEntered()
				return ai.ScriptedStream(nativeCall("p2", "job_send", `{"id":"job_1","message":"last-minute correction"}`))(ctx, model, view, options)
			case 3:
				return ai.ScriptedStream(nativeCall("p3", "job_wait", `{"id":"job_1","timeout_seconds":20}`))(ctx, model, view, options)
			}
			return ai.ScriptedStream(nativeAnswer("parent done"))(ctx, model, view, options)
		}
		switch script.child.Add(1) {
		case 1:
			return ai.ScriptedStream(nativeCall("c1", "gate", `{"text":"x"}`))(ctx, model, view, options)
		}
		if strings.Contains(strings.Join(userTexts, "\n"), "last-minute correction") {
			return ai.ScriptedStream(nativeAnswer("child read the correction"))(ctx, model, view, options)
		}
		return ai.ScriptedStream(nativeAnswer("child answered without it"))(ctx, model, view, options)
	})
	a := steerAgent(t, provider, gate, nil)
	// Open the gate right after the message is queued, so the child's next step
	// is the one that must see it.
	go func() {
		gate.waitEntered()
		time.Sleep(200 * time.Millisecond)
		close(gate.release)
	}()
	res, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "parent work"}}})
	if err != nil || res.Final != "parent done" {
		t.Fatalf("run = %+v %v", res, err)
	}
	var found bool
	for _, m := range res.Messages {
		if strings.Contains(m.Content, "child read the correction") {
			found = true
		}
	}
	if !found {
		t.Fatalf("the parent never received the child's corrected answer: %+v", res.Messages)
	}
}

// job_send is refused, with a reason the model can act on, for a job that is
// finished, unknown or not a sub-agent. Nothing is queued anywhere.
func TestJobSendRefusals(t *testing.T) {
	gate := newGate()
	close(gate.release)
	script := &steerScript{}
	spawn := `{"task":"` + steerMarker + `","async":true}`
	var results []string
	var mu sync.Mutex
	provider := nativeProvider(func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, options map[string]any) (*ai.AssistantMessageEventStream, error) {
		var userTexts []string
		var toolResults []string
		for _, m := range view.Messages() {
			if m.Role == "user" && m.Content.Text != nil {
				userTexts = append(userTexts, *m.Content.Text)
			}
			if m.Role == "toolResult" {
				raw, _ := json.Marshal(m)
				toolResults = append(toolResults, string(raw))
			}
		}
		if strings.Contains(strings.Join(userTexts, "\n"), steerMarker) {
			if script.child.Add(1) == 1 {
				return ai.ScriptedStream(nativeCall("c1", "gate", `{"text":"x"}`))(ctx, model, view, options)
			}
			return ai.ScriptedStream(nativeAnswer("child done"))(ctx, model, view, options)
		}
		turn := script.parent.Add(1)
		mu.Lock()
		results = toolResults
		mu.Unlock()
		switch turn {
		case 1:
			return ai.ScriptedStream(nativeCall("p1", subagent.ToolSpawnSubagent, spawn))(ctx, model, view, options)
		case 2:
			return ai.ScriptedStream(nativeCall("p2", "job_wait", `{"id":"job_1","timeout_seconds":20}`))(ctx, model, view, options)
		case 3:
			return ai.ScriptedStream(nativeCall("p3", "job_send", `{"id":"job_1","message":"too late"}`))(ctx, model, view, options)
		case 4:
			return ai.ScriptedStream(nativeCall("p4", "job_send", `{"id":"job_404","message":"hello"}`))(ctx, model, view, options)
		case 5:
			return ai.ScriptedStream(nativeCall("p5", "job_send", `{"id":"job_1","message":""}`))(ctx, model, view, options)
		}
		return ai.ScriptedStream(nativeAnswer("parent done"))(ctx, model, view, options)
	})
	a := steerAgent(t, provider, gate, nil)
	if _, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "parent work"}}}); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	all := strings.Join(results, "\n")
	mu.Unlock()
	for _, want := range []string{"already finished", "job not found", "message is required"} {
		if !strings.Contains(all, want) {
			t.Errorf("tool results lack %q:\n%s", want, all)
		}
	}
}

// Without AllowSend the model is not offered job_send at all.
func TestJobSendIsOffOnlyByDefault(t *testing.T) {
	script := newNativeScript(nativeAnswer("hi"))
	a, err := agentcore.New(agentcore.Config{NativeProvider: script.Provider, Model: "faux-1", Policy: agentcore.NewAllowList("x"), Extensions: []agentcore.ExtensionFactory{jobs.Plugin{}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.RunNative(context.Background(), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}}); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tool := range script.Recorded[0].Tools {
		names = append(names, tool.Name)
	}
	if got := strings.Join(names, ","); got != "job_list,job_status,job_wait,job_cancel" {
		t.Fatalf("tools = %s", got)
	}
}
