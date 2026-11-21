package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// The board gives the model one tool: its op enum lists exactly the ops the
// plugin enables, each op reaches its handler, and an op the run does not
// take is refused with the ones it does.
func TestOneJobToolTakesTheEnabledOps(t *testing.T) {
	b := begin(t, Plugin{})
	tools := b.Tools()
	if len(tools) != 1 || tools[0].Name() != ToolName {
		t.Fatalf("tools = %v", toolNames(b))
	}
	schema := tools[0].Schema()
	raw, _ := json.Marshal(schema.Parameters)
	if !strings.Contains(string(raw), `"enum":["list","status","wait","cancel"]`) || strings.Contains(string(raw), "message") || strings.Contains(schema.Description, "op keep") {
		t.Fatalf("schema = %s\n%s", raw, schema.Description)
	}
	tool := tools[0]
	if out, err := tool.Run(context.Background(), `{"op":"list"}`); err != nil || out != "No background jobs." {
		t.Fatalf("list = %q, %v", out, err)
	}
	r, _ := b.start(context.Background(), "bash", "sleep", block)
	if out, err := tool.Run(context.Background(), fmt.Sprintf(`{"op":"status","id":%q,"output_bytes":null}`, r.ID)); err != nil || !strings.Contains(out, "running") {
		t.Fatalf("status = %q, %v", out, err)
	}
	if out, err := tool.Run(context.Background(), fmt.Sprintf(`{"op":"cancel","id":%q}`, r.ID)); err != nil || !strings.Contains(out, r.ID) {
		t.Fatalf("cancel = %q, %v", out, err)
	}
	for args, want := range map[string]string{
		`{}`:                             "op is required: one of list, status, wait, cancel",
		`{"op":"keep","id":"job_1"}`:     `op "keep" is not one this run takes`,
		`{"op":"send","id":"job_1"}`:     `op "send" is not one this run takes`,
		`{"op":"status"}`:                "id is required",
		`{"op":"status","id":"job_404"}`: "job not found",
	} {
		if _, err := tool.Run(context.Background(), args); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want %q", args, err, want)
		}
	}
	both, _ := run(t, &Survival{Hold: NewHold(nil)}, "conv-1", func(p *Plugin) { p.AllowSend = true })
	schema = both.Tools()[0].Schema()
	raw, _ = json.Marshal(schema.Parameters)
	if !strings.Contains(string(raw), `"enum":["list","status","wait","cancel","send","keep"]`) || !strings.Contains(string(raw), `"message"`) || !strings.Contains(schema.Description, "op keep") || !strings.Contains(schema.Description, "unless kept") {
		t.Fatalf("schema with send and keep = %s\n%s", raw, schema.Description)
	}
}
