package jsonjs

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestRawObjectRetainsDistinctUTF16KeysAndOwnsValues(t *testing.T) {
	raw := []byte(`{"\ud800":1,"\ud801":2,"�":3,"\ud800":4,"large":1e400}`)
	var object RawObject
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatal(err)
	}
	high, _ := DecodeJSON([]byte(`"\ud800"`))
	next, _ := DecodeJSON([]byte(`"\ud801"`))
	if len(object) != 4 || string(object[high.(string)]) != "4" || string(object[next.(string)]) != "2" || string(object["�"]) != "3" || string(object["large"]) != "1e400" {
		t.Fatalf("lost property or raw number: %#v", object)
	}
	for i := range raw {
		raw[i] = ' '
	}
	encoded, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	var copy RawObject
	if err := json.Unmarshal(encoded, &copy); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(copy, object) {
		t.Fatalf("round trip changed fields: %s", encoded)
	}
	if string(copy[high.(string)]) != "4" {
		t.Fatal("decode retained caller-owned bytes")
	}
}

func TestRawObjectRejectsMalformedInputWithoutReplacingReceiver(t *testing.T) {
	object := RawObject{"keep": json.RawMessage(`true`)}
	for _, raw := range []string{`{"x":}`, `{"x":1} trailing`, `[]`, `1`, `"text"`} {
		if err := json.Unmarshal([]byte(raw), &object); err == nil {
			t.Fatalf("accepted %q", raw)
		}
		if len(object) != 1 || string(object["keep"]) != "true" {
			t.Fatalf("failed decode changed receiver: %#v", object)
		}
	}
	if _, err := json.Marshal(RawObject{"x": json.RawMessage(`{`)}); err == nil {
		t.Fatal("exported malformed raw value")
	}
	if err := json.Unmarshal([]byte(`null`), &object); err != nil || object != nil {
		t.Fatalf("null: %#v %v", object, err)
	}
	encoded, err := json.Marshal(object)
	if err != nil || string(encoded) != "null" {
		t.Fatalf("nil object: %s %v", encoded, err)
	}
}
