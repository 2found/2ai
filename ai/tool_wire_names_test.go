package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

var portableName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_-]{0,63}$`)

// A gateway in front of a Gemini-class backend rewrites a function name that
// starts with a digit and never restores it. The caller must still see the
// tool it declared, on every reply shape, and the gateway must see nothing to
// rewrite.
func TestToolNamesSurviveAGatewayThatRewritesThem(t *testing.T) {
	const tool = "2found-hub_radar_scan"
	for _, shape := range []string{"json", "sse", "stream"} {
		t.Run(shape, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req oaiRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatalf("decode request: %v", err)
				}
				sent := map[string]bool{}
				for _, item := range req.Tools {
					if !portableName.MatchString(item.Function.Name) {
						t.Errorf("tool %q is not portable", item.Function.Name)
					}
					sent[item.Function.Name] = true
				}
				for _, m := range req.Messages {
					for _, call := range m.ToolCalls {
						if !sent[call.Function.Name] {
							t.Errorf("history call %q does not match a declared tool", call.Function.Name)
						}
					}
					if m.Role == "tool" && !sent[m.Name] {
						t.Errorf("tool result %q does not match a declared tool", m.Name)
					}
				}
				choice, _ := json.Marshal(req.ToolChoice)
				if strings.Contains(string(choice), tool) && !strings.Contains(string(choice), `"_`+tool) {
					t.Errorf("named tool choice kept the raw name: %s", choice)
				}
				name := req.Tools[0].Function.Name
				if name[0] >= '0' && name[0] <= '9' {
					name = "_" + name // the gateway's own repair
				}
				call := fmt.Sprintf(`{"index":0,"id":"call_1","type":"function","function":{"name":%q,"arguments":"{}"}}`, name)
				if shape == "json" {
					fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[%s]},"finish_reason":"tool_calls"}]}`, call)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[%s]}}]}\n\n", call)
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer srv.Close()
			p := NewOpenAIProvider("", srv.URL, DefaultCompat())
			p.HTTP, p.StreamHTTP = srv.Client(), srv.Client()
			req := protocol.ChatRequest{
				Model:      "lite",
				Tools:      []protocol.ToolSchema{{Name: tool, Description: "Scan the radar.", Parameters: map[string]any{"type": "object"}}, {Name: "data-pool_query", Parameters: map[string]any{"type": "object"}}},
				ToolChoice: protocol.ToolChoice{Mode: protocol.ToolChoiceNamed, Name: tool},
				Messages: []protocol.Message{
					{Role: protocol.RoleUser, Content: "scan"},
					{Role: protocol.RoleAssistant, ToolCalls: []protocol.ToolCall{{ID: "call_0", Name: tool, Arguments: "{}"}}},
					{Role: protocol.RoleTool, ToolCallID: "call_0", Name: tool, Content: "nothing new"},
				},
			}
			var got []protocol.ToolCall
			if shape == "stream" {
				ch, err := p.Stream(context.Background(), req)
				if err != nil {
					t.Fatal(err)
				}
				for delta := range ch {
					if delta.Err != nil {
						t.Fatal(delta.Err)
					}
					if delta.ToolCall != nil {
						got = append(got, *delta.ToolCall)
					}
				}
			} else {
				resp, err := p.Chat(context.Background(), req)
				if err != nil {
					t.Fatal(err)
				}
				got = resp.Message.ToolCalls
			}
			if len(got) != 1 || got[0].Name != tool {
				t.Fatalf("model call came back as %+v, want %q", got, tool)
			}
		})
	}
}

func TestPortableToolNamesStayDistinctAndReversible(t *testing.T) {
	long := strings.Repeat("a", 80)
	req := protocol.ChatRequest{Tools: []protocol.ToolSchema{{Name: "2a"}, {Name: "_2a"}, {Name: "web.search"}, {Name: "plain_tool"}, {Name: long + "x"}, {Name: long + "y"}}}
	names := newToolWireNames(req)
	seen := map[string]string{}
	for _, tool := range req.Tools {
		wire := names.toWire(tool.Name)
		if !portableName.MatchString(wire) {
			t.Errorf("%q -> %q is not portable", tool.Name, wire)
		}
		if other, ok := seen[wire]; ok {
			t.Errorf("%q and %q share wire name %q", other, tool.Name, wire)
		}
		seen[wire] = tool.Name
		if back := names.fromWire(wire); back != tool.Name {
			t.Errorf("%q -> %q -> %q", tool.Name, wire, back)
		}
	}
	if names.toWire("plain_tool") != "plain_tool" || names.toWire("_2a") != "_2a" {
		t.Error("a portable name was renamed")
	}
}

func TestPortableCompletionsBodyIsUntouchedOrFullyRenamed(t *testing.T) {
	clean := json.RawMessage(`{"model":"m","tools":[{"type":"function","function":{"name":"read","parameters":{"z":1,"a":2}}}],"messages":[{"role":"tool","name":"read","content":"<ok>"}]}`)
	out, _, err := portableCompletionsToolNames(clean)
	if err != nil || string(out) != string(clean) {
		t.Fatalf("portable body changed: %s %v", out, err)
	}
	body := json.RawMessage(`{"model":"m","tool_choice":{"type":"function","function":{"name":"2x"}},"tools":[{"type":"function","function":{"name":"2x","parameters":{}}}],"messages":[{"role":"system","tools":[{"type":"custom","custom":{"name":"9y"}}]},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"2x","arguments":"{}"}}]},{"role":"tool","name":"2x","tool_call_id":"c","content":"ok"}]}`)
	out, names, err := portableCompletionsToolNames(body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), `"2x"`) || strings.Contains(string(out), `"9y"`) || strings.Count(string(out), `"_2x"`) != 4 || !strings.Contains(string(out), `"_9y"`) {
		t.Fatalf("not every site was renamed: %s", out)
	}
	if names.fromWire("_2x") != "2x" || names.fromWire("_9y") != "9y" {
		t.Fatal("renames do not map back")
	}
}
