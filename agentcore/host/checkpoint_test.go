package host

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/2found/2ai/ai/protocol"
)

func TestCompactorCheckpointRestoreAndIsolation(t *testing.T) {
	raw := piRequestJSON(piLongRequest())
	var persisted json.RawMessage
	options := CompactorOptions{
		Revision: "revision-a",
		Policy: CompactionPolicy{Budget: 1500, KeepRecent: 500,
			Summarize: func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
				return "Earlier evidence is retained.", protocol.Usage{}, nil
			}},
		Record: func(_ context.Context, raw json.RawMessage) error {
			persisted = append(json.RawMessage(nil), raw...)
			return nil
		},
	}
	compactor := NewCompactor(options)
	if compactor.Checkpoint() != nil {
		t.Fatal("new compactor invented a checkpoint")
	}
	want := compactor.Transform(context.Background(), raw)
	checkpoint := compactor.Checkpoint()
	if len(checkpoint) == 0 || !SameJSON(checkpoint, persisted) {
		t.Fatal("checkpoint differs from the committed summary")
	}
	options.Policy.Summarize = func(context.Context, json.RawMessage, string) (string, protocol.Usage, error) {
		t.Fatal("restored exact prefix was summarized again")
		return "", protocol.Usage{}, nil
	}
	restored := NewCompactor(options)
	if err := restored.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	// Neither the input to Restore nor the exported checkpoint owns saved bytes.
	checkpoint[0] = '!'
	exported := restored.Checkpoint()
	exported[0] = '!'
	if got := restored.Transform(context.Background(), raw); !SameJSON(got, want) {
		t.Fatal("restored view changed after caller mutated checkpoint bytes")
	}
	before := restored.Checkpoint()
	var foreign map[string]any
	if err := json.Unmarshal(before, &foreign); err != nil {
		t.Fatal(err)
	}
	foreign["revision"] = "revision-b"
	for _, invalid := range []json.RawMessage{[]byte(`null`), []byte(`{"prefix_count":0}`), piRequestJSON(foreign)} {
		if err := restored.Restore(invalid); err == nil {
			t.Fatal("invalid or foreign checkpoint restored")
		}
		if !SameJSON(before, restored.Checkpoint()) {
			t.Fatal("failed restore replaced the useful checkpoint")
		}
	}
}

func TestCompactionPolicyModelWindowDoesNotMutateBase(t *testing.T) {
	policy := CompactionPolicy{Budget: 100000, KeepRecent: 20000}
	small := policy.ForWindow(20000)
	large := policy.ForWindow(200000)
	if small.Budget != 15000 || large.Budget != 100000 || policy.Budget != 100000 {
		t.Fatalf("model switch retained an earlier rung's budget: %#v %#v %#v", policy, small, large)
	}
}
