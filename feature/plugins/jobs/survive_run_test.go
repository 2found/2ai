package jobs_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/2found/2ai/core/agentcore"
	"github.com/2found/2ai/core/ai"
	"github.com/2found/2ai/feature/plugins/jobs"
)

func call(id, name, args string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "toolUse", Content: ai.BlockContent(ai.ContentBlock{Type: "toolCall", ID: id, Name: name, Arguments: json.RawMessage(args)})}
}

func answer(text string) ai.Message {
	return ai.Message{Role: "assistant", Model: "test", StopReason: "stop", Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: text})}
}

func provider(replies ...ai.Message) (*ai.FallbackProvider, func() []string) {
	var mu sync.Mutex
	var results []string
	script := ai.ScriptedStream(replies...)
	return &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{Model: json.RawMessage(`{"id":"test"}`), Stream: func(ctx context.Context, model json.RawMessage, view ai.TranscriptContext, opts map[string]any) (*ai.AssistantMessageEventStream, error) {
		mu.Lock()
		results = nil
		for _, m := range view.Messages() {
			raw, _ := json.Marshal(m)
			results = append(results, string(raw))
		}
		mu.Unlock()
		return script(ctx, model, view, opts)
	}}}}, func() []string { mu.Lock(); defer mu.Unlock(); return append([]string(nil), results...) }
}

// devServer is a host tool that goes asynchronous: it starts a "server" that
// runs until told to stop, and publishes a line as it comes up.
type devServer struct {
	started chan struct{}
	stop    chan struct{}
	ctxErr  chan error
}

func (d *devServer) Name() string { return "dev_server" }
func (d *devServer) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{Name: "dev_server", Description: "start the dev server", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}
}
func (d *devServer) Run(ctx context.Context, _ string) (string, error) {
	receipt, err := agentcore.LaunchBackground(ctx, "dev_server", "npm run dev", func(jobCtx context.Context) (string, error) {
		agentcore.PublishJobOutput(jobCtx, "ready on http://localhost:5173\n")
		close(d.started)
		select {
		case <-d.stop:
			return "server stopped", nil
		case <-jobCtx.Done():
			d.ctxErr <- jobCtx.Err()
			return "", jobCtx.Err()
		}
	})
	return string(receipt), err
}

func newAgent(t *testing.T, p *ai.FallbackProvider, tool agentcore.Tool, plugin jobs.Plugin) *agentcore.Agent {
	t.Helper()
	a, err := agentcore.New(agentcore.Config{NativeProvider: p, Model: "faux-1", Tools: agentcore.NewToolSet(tool),
		Policy: agentcore.NewAllowList("dev_server"), Extensions: []agentcore.ExtensionFactory{plugin}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// The whole story, through the real loop: the model starts a server, keeps it,
// and answers; the run ends and the server does not; the next turn of the same
// conversation lists it, reads its output and cancels it; a stranger's turn
// sees nothing; the host learns when a kept job ends while nobody is talking.
func TestKeptJobSurvivesToTheNextTurn(t *testing.T) {
	ended := make(chan jobs.Ended, 4)
	hold := jobs.NewHold(func(e jobs.Ended) { ended <- e })
	plugin := jobs.Plugin{Survive: &jobs.Survival{Hold: hold}}
	server := &devServer{started: make(chan struct{}), stop: make(chan struct{}), ctxErr: make(chan error, 1)}

	first, _ := provider(call("c1", "dev_server", `{}`), call("c2", "job_keep", `{"id":"job_1"}`), answer("the server is up on :5173"))
	ctx := jobs.WithConversation(context.Background(), "conv-42")
	res, err := newAgent(t, first, server, plugin).RunNative(ctx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "start the dev server"}}})
	if err != nil || res.StopReason != "stop" || res.Final != "the server is up on :5173" {
		t.Fatalf("first run = %+v %v", res, err)
	}
	select {
	case err := <-server.ctxErr:
		t.Fatalf("the kept job was cancelled with its run: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	// The same conversation, a later turn, resuming from the checkpoint.
	second, seen := provider(call("c3", "job_list", `{}`), call("c4", "job_status", `{"id":"job_1"}`), call("c5", "job_cancel", `{"id":"job_1"}`), call("c6", "job_wait", `{"id":"job_1","timeout_seconds":5}`), answer("stopped it"))
	res2, err := newAgent(t, second, server, plugin).RunNative(ctx, agentcore.NativeRun{State: res.NativeState, Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "what is running?"}}})
	if err != nil || res2.Final != "stopped it" {
		t.Fatalf("second run = %+v %v", res2, err)
	}
	all := strings.Join(seen(), "\n")
	for _, want := range []string{"running (kept)", "ready on http://localhost:5173", "Cancellation requested"} {
		if !strings.Contains(all, want) {
			t.Errorf("the next turn never saw %q:\n%s", want, all)
		}
	}
	select {
	case err := <-server.ctxErr:
		if err == nil {
			t.Fatal("no cancellation error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("job_cancel from the next turn did not stop the server")
	}
	select {
	case e := <-ended:
		// The job ended while this turn was still attached? No: the turn that
		// cancelled it was attached, so the host must not be told.
		t.Fatalf("callback fired for a job an attached run owned: %+v", e)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestKeptJobIsNotVisibleToAnotherConversationThroughTheLoop(t *testing.T) {
	hold := jobs.NewHold(nil)
	plugin := jobs.Plugin{Survive: &jobs.Survival{Hold: hold}}
	server := &devServer{started: make(chan struct{}), stop: make(chan struct{}), ctxErr: make(chan error, 1)}
	first, _ := provider(call("c1", "dev_server", `{}`), call("c2", "job_keep", `{"id":"job_1"}`), answer("up"))
	res, err := newAgent(t, first, server, plugin).RunNative(jobs.WithConversation(context.Background(), "conv-A"), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}})
	if err != nil || res.Final != "up" {
		t.Fatal(res, err)
	}
	other, seen := provider(call("c3", "job_status", `{"id":"job_1"}`), call("c4", "job_cancel", `{"id":"job_1"}`), answer("nothing there"))
	if _, err := newAgent(t, other, server, plugin).RunNative(jobs.WithConversation(context.Background(), "conv-B"), agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "status?"}}}); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(seen(), "\n"); strings.Count(got, "job not found") < 2 || strings.Contains(got, "localhost:5173") {
		t.Fatalf("another conversation reached the job:\n%s", got)
	}
	select {
	case <-server.ctxErr:
		t.Fatal("another conversation cancelled the job")
	case <-time.After(100 * time.Millisecond):
	}
	hold.CancelAll()
}

// With the option off the loop behaves as it always did: the job dies with the
// run and job_keep does not exist.
func TestWithoutTheOptionAJobDiesWithItsRun(t *testing.T) {
	server := &devServer{started: make(chan struct{}), stop: make(chan struct{}), ctxErr: make(chan error, 1)}
	p, seen := provider(call("c1", "dev_server", `{}`), call("c2", "job_keep", `{"id":"job_1"}`), answer("done"))
	// An unkept job also holds the final answer back, so end the run the way a
	// host does when the person walks away.
	runCtx, stop := context.WithCancel(jobs.WithConversation(context.Background(), "conv-off"))
	go func() { <-server.started; time.Sleep(50 * time.Millisecond); stop() }()
	// The aborted run reports its cancellation; what matters is what the job did.
	_, _ = newAgent(t, p, server, jobs.Plugin{}).RunNative(runCtx, agentcore.NativeRun{Input: []agentcore.Message{{Role: agentcore.RoleUser, Content: "go"}}})
	select {
	case <-server.ctxErr:
	case <-time.After(5 * time.Second):
		t.Fatal("the job outlived its run without the option")
	}
	if got := strings.Join(seen(), "\n"); strings.Contains(got, "will keep running") {
		t.Fatalf("job_keep acted without the option:\n%s", got)
	}
}
