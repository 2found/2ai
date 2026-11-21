package jobs

import (
	"context"
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
)

// A disposable job (a dev server) still running at the final answer is
// cancelled then and re-opens nothing; any other pending job still does.
func TestDisposableJobEndsWithTheAnswer(t *testing.T) {
	b := begin(t, Plugin{Disposable: func(tool string) bool { return tool == "bash" }})
	server, _ := b.start(context.Background(), "bash", "python3 -m http.server", block)
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue {
		t.Fatalf("a dev server re-opened the run: %+v", d)
	}
	if r, _ := b.get(server.ID); r.State != cancelled || !r.Announced {
		t.Fatalf("server = %+v, want cancelled and not announced again", r)
	}
	if text := completionNotice(b.drain(), 1024); text != "" {
		t.Fatalf("its end was announced: %q", text)
	}
	b.stopWait = 0
	other, _ := b.start(context.Background(), "spawn_subagent", "research", block)
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); !d.Continue || !strings.Contains(d.Inject[0].Content, "still pending") {
		t.Fatalf("a pending sub-agent did not re-open the run: %+v", d)
	}
	b.CloseRun()
	_ = other
}

// The model that read how a job ended (op cancel, op status, op wait) is
// not told again.
func TestAJobTheModelSawEndIsNotAnnounced(t *testing.T) {
	b := begin(t, Plugin{})
	r, _ := b.start(context.Background(), "bash", "server", block)
	out, err := (jobCancelTool{jobTool{b, ToolName}}).Run(context.Background(), `{"id":"`+r.ID+`"}`)
	if err != nil || !strings.Contains(out, "it has stopped") {
		t.Fatalf("op cancel = %q, %v", out, err)
	}
	if d := b.TurnStopping(context.Background(), agentcore.StopInfo{}); d.Continue {
		t.Fatalf("a job the model cancelled re-opened the run: %+v", d)
	}
	done, _ := b.start(context.Background(), "bash", "quick", func(context.Context) (string, error) { return "ok", nil })
	settled(t, b, done.ID)
	if _, err := (jobStatusTool{jobTool{b, ToolName}}).Run(context.Background(), `{"id":"`+done.ID+`"}`); err != nil {
		t.Fatal(err)
	}
	if len(b.drain()) != 0 {
		t.Fatal("a job read with op status was announced")
	}
}
