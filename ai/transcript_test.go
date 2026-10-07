package ai

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func assertPiJSON(t *testing.T, expected json.RawMessage, actual any) {
	t.Helper()
	encoded, err := json.Marshal(actual)
	if err != nil {
		t.Fatal(err)
	}
	var want, got any
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("Pi: %s\nGo: %s", expected, encoded)
	}
}

func TestPiTranscriptOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-transcript.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		UpstreamCommit string `json:"upstreamCommit"`
		Transcripts    []struct {
			Name     string
			Context  Context
			Expected map[string]json.RawMessage
		}
		ToolStates []struct {
			Previous, Current []Tool
			Expected          json.RawMessage
		}
		Transforms []struct {
			Name      string
			Messages  []Message
			Model     Model
			Normalize bool
			Now       int64
			Expected  json.RawMessage
		}
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	manifest, err := os.ReadFile("testdata/upstream.json")
	if err != nil {
		t.Fatal(err)
	}
	var pin struct{ Commit string }
	if err := json.Unmarshal(manifest, &pin); err != nil {
		t.Fatal(err)
	}
	if fixtures.UpstreamCommit != pin.Commit {
		t.Fatal("Pi oracle revision does not match the source pin")
	}
	for _, fixture := range fixtures.Transcripts {
		t.Run(fixture.Name, func(t *testing.T) {
			before, _ := json.Marshal(fixture.Context)
			context := NormalizeContext(fixture.Context)
			messages := context.Messages()
			systemTexts, contentTexts := [][2]string{}, [][2]string{}
			for _, message := range messages {
				if message.Role == "system" {
					systemTexts = append(systemTexts, [2]string{GetSystemMessageText(message), RenderSystemMessageUpdate(message)})
				}
				contentTexts = append(contentTexts, [2]string{ContentText(message.Content), ContentText(message.Content, "|")})
			}
			outputs := map[string]any{
				"normalized": context, "initial": GetInitialSystemMessage(messages), "withoutInitial": WithoutInitialSystemMessage(messages),
				"current": GetCurrentSystemMessage(messages), "prompt": GetCurrentSystemPrompt(messages),
				"collapsed": CollapseSystemMessages(context), "resolvedTrue": ResolveTranscript(context, true), "resolvedFalse": ResolveTranscript(context, false),
				"currentTools": GetCurrentTools(messages), "declaredTools": GetDeclaredTools(messages),
				"redefinitions": HasToolRedefinitions(messages), "nonAdditive": HasNonAdditiveToolChanges(messages),
				"anchoredTools": ResolveTranscriptTools(messages, true), "flatTools": ResolveTranscriptTools(messages, false),
				"systemTexts": systemTexts, "contentTexts": contentTexts,
			}
			for key, actual := range outputs {
				t.Run(key, func(t *testing.T) { assertPiJSON(t, fixture.Expected[key], actual) })
			}
			assertPiJSON(t, before, fixture.Context)
			collapsed := CollapseSystemMessages(context)
			encoded, _ := json.Marshal(collapsed)
			assertPiJSON(t, encoded, CollapseSystemMessages(collapsed))
		})
	}
	for i, fixture := range fixtures.ToolStates {
		t.Run("tool states/"+string(rune('a'+i)), func(t *testing.T) {
			assertPiJSON(t, fixture.Expected, GetToolStateChanges(fixture.Previous, fixture.Current))
		})
	}
	for _, fixture := range fixtures.Transforms {
		t.Run("transform/"+fixture.Name, func(t *testing.T) {
			before, _ := json.Marshal(fixture.Messages)
			var normalize ToolCallIDNormalizer
			if fixture.Normalize {
				normalize = func(id string, _ *Model, _ *Message) string { return "normalized:" + id }
			}
			actual := transformMessagesAt(fixture.Messages, fixture.Model, normalize, func() int64 { return fixture.Now })
			assertPiJSON(t, fixture.Expected, actual)
			assertPiJSON(t, before, fixture.Messages)
		})
	}
}

func TestPiTranscriptDeclarationCopy(t *testing.T) {
	tool := Tool{Name: "a", Parameters: json.RawMessage(`{"type":"object"}`), Extra: map[string]json.RawMessage{"label": json.RawMessage(`"display"`)}}
	declaration := ToToolDeclaration(tool)
	declaration.Parameters[2] = 'X'
	if string(tool.Parameters) != `{"type":"object"}` || declaration.Extra != nil {
		t.Fatal("declaration retains mutable schema or display metadata")
	}
	if DeclarationsEqual(tool, Tool{Name: "a", Parameters: json.RawMessage(`{"type":`)}) {
		t.Fatal("invalid schema compares equal")
	}
}

func TestPiTranscriptJSONDeclarationSemantics(t *testing.T) {
	for _, pair := range [][2]string{
		{`{"n":-0,"v":1.0,"s":"\u0061"}`, `{"n":0,"v":1,"s":"a"}`},
		{`{"b":1,"10":10,"2":2}`, `{"2":2,"10":10,"b":1}`},
		{`{"a":1,"b":2,"a":3}`, `{"a":3,"b":2}`},
	} {
		if !declarationJSONEqual(json.RawMessage(pair[0]), json.RawMessage(pair[1])) {
			t.Errorf("JSON.stringify-equivalent schemas differ: %s / %s", pair[0], pair[1])
		}
	}
	if declarationJSONEqual(json.RawMessage(`{"a":1,"b":2}`), json.RawMessage(`{"b":2,"a":1}`)) {
		t.Fatal("declaration comparison discarded property insertion order")
	}
}

func TestTranscriptWirePresenceAndMutation(t *testing.T) {
	raw := json.RawMessage(`{"role":"assistant","content":[],"model":null,"usage":null,"responseId":null}`)
	var message Message
	if err := json.Unmarshal(raw, &message); err != nil {
		t.Fatal(err)
	}
	assertPiJSON(t, raw, message)
	message.Model, message.Timestamp = "updated", 42
	assertPiJSON(t, json.RawMessage(`{"role":"assistant","content":[],"model":"updated","usage":null,"responseId":null,"timestamp":42}`), message)
	var tool Tool
	if err := json.Unmarshal([]byte(`{"name":"read","parameters":{"type":"object"}}`), &tool); err != nil {
		t.Fatal(err)
	}
	tool.Description = "Read evidence"
	assertPiJSON(t, json.RawMessage(`{"name":"read","description":"Read evidence","parameters":{"type":"object"}}`), tool)
	generated := Message{Role: "assistant", Content: MessageContent{}}
	encoded, err := json.Marshal(generated)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	_ = json.Unmarshal(encoded, &fields)
	for _, name := range []string{"timestamp", "api", "provider", "model", "usage", "stopReason"} {
		if _, ok := fields[name]; !ok {
			t.Fatalf("new message lost required %s: %s", name, encoded)
		}
	}
}

func TestTranscriptUsageWirePresenceAndMutation(t *testing.T) {
	raw := json.RawMessage(`{"role":"assistant","content":[],"usage":{"input":5,"output":null,"providerMeter":{"requests":1},"cost":{"input":0.1,"currency":"USD"}}}`)
	var message Message
	if err := json.Unmarshal(raw, &message); err != nil {
		t.Fatal(err)
	}
	assertPiJSON(t, raw, message)
	// Copies share only immutable encoding metadata; updating counters in a
	// copied record must retain extensions without mutating the original.
	usage := *message.Usage
	usage.Output, usage.Cost.Output = 7, 0.2
	usage.Cost.Input = 0
	assertPiJSON(t, json.RawMessage(`{"input":5,"output":7,"providerMeter":{"requests":1},"cost":{"input":0,"output":0.2,"currency":"USD"}}`), usage)
	assertPiJSON(t, raw, message)
	for _, source := range []string{`{}`, `{"cost":null}`, `{"input":null,"cost":{}}`, `{"reasoning":null,"cacheWrite1h":null}`} {
		var decoded Usage
		if err := json.Unmarshal([]byte(source), &decoded); err != nil {
			t.Fatal(err)
		}
		assertPiJSON(t, json.RawMessage(source), decoded)
	}
	// Native producers still get the complete initial usage object.
	assertPiJSON(t, json.RawMessage(`{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}`), Usage{})
}

func TestToolDeclarationReplacementPreservesSourceMessage(t *testing.T) {
	raw := json.RawMessage(`{"role":"system","content":null,"timestamp":null,"toolsAdded":null,"toolsRemoved":null,"host":{"opaque":1}}`)
	var source Message
	if err := json.Unmarshal(raw, &source); err != nil {
		t.Fatal(err)
	}
	stripped := WithToolChanges(source, ToolStateChanges{})
	assertPiJSON(t, json.RawMessage(`{"role":"system","content":null,"timestamp":null,"host":{"opaque":1}}`), stripped)
	replaced := WithToolChanges(source, ToolStateChanges{ToolsRemoved: []ToolReference{{Name: "read"}}})
	assertPiJSON(t, json.RawMessage(`{"role":"system","content":null,"timestamp":null,"host":{"opaque":1},"toolsRemoved":[{"name":"read"}]}`), replaced)
	assertPiJSON(t, raw, source)
	// Explicit fields supplied by Go callers through Extra follow the same
	// removal semantics, without deleting entries from the caller's map.
	source.Extra["toolsAdded"] = json.RawMessage(`null`)
	source.Extra["toolsRemoved"] = nil
	stripped = WithToolChanges(source, ToolStateChanges{})
	assertPiJSON(t, json.RawMessage(`{"role":"system","content":null,"timestamp":null,"host":{"opaque":1}}`), stripped)
	if _, exists := source.Extra["toolsRemoved"]; !exists {
		t.Fatal("declaration replacement changed the source metadata map")
	}
}
