package jscompat_test

import (
	"encoding/json"
	"reflect"
	"testing"
)

// assertPiJSON mirrors the root package's test helper of the same name: it
// compares the JSON value of actual against the recorded upstream value.
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
