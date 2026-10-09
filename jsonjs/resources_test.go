package jsonjs

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func resourceObject() []byte {
	return []byte(`{"name":"fixture","nested":[` + strings.TrimSuffix(strings.Repeat(`{"z":1e999,"2":"\ud800","1":["escaped \\\" brace }",null,true,"`+strings.Repeat("payload ", 64)+`"]},`, 32), ",") + `],"tail":-0}`)
}

func TestRawPropertiesPreserveNestedSourceWithoutMaterializingIt(t *testing.T) {
	raw := resourceObject()
	props, err := DecodeObjectProperties(raw)
	if err != nil || len(props) != 3 {
		t.Fatalf("properties: %d %v", len(props), err)
	}
	if props[0].Name != "name" || string(props[0].Value) != `"fixture"` || props[1].Name != "nested" || !json.Valid(props[1].Value) || props[2].Name != "tail" || string(props[2].Value) != "-0" {
		t.Fatal("raw source or property order changed")
	}
	if !bytes.Contains(props[1].Value, []byte(`1e999`)) || !bytes.Contains(props[1].Value, []byte(`\ud800`)) {
		t.Fatal("nested values were normalized")
	}
	allocations := testing.AllocsPerRun(10, func() {
		if _, err := DecodeObjectProperties(raw); err != nil {
			t.Fatal(err)
		}
	})
	if allocations > 50 {
		t.Fatalf("raw property boundary scan materialized nested values: %.0f allocations", allocations)
	}
}

func BenchmarkDecodeObjectProperties(b *testing.B) {
	raw := resourceObject()
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeObjectProperties(raw); err != nil {
			b.Fatal(err)
		}
	}
}

func TestQuoteStringFastPathKeepsUnicodeAndEscapes(t *testing.T) {
	value := strings.Repeat("Tiếng Việt 🌏 <tag> & \u2028 \u2029\n\t\"\\\x00", 128)
	encoded := QuoteString(value)
	var decoded string
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded != value {
		t.Fatalf("UTF-8/escaping changed: %v", err)
	}
	if bytes.Contains(encoded, []byte(`\u003c`)) || !bytes.Contains(encoded, []byte("\u2028")) {
		t.Fatal("JSON.stringify escaping changed")
	}
	if n := testing.AllocsPerRun(20, func() { _ = QuoteString(value) }); n > 3 {
		t.Fatalf("ordinary UTF-8 took the UTF-16 materialization path: %.0f allocations", n)
	}
	for input, want := range map[string]bool{"0": true, "1": true, "4294967294": true, "4294967295": false, "4294967296": false, "01": false, "+1": false, "-0": false, "name": false, "": false} {
		if _, ok := ArrayIndex(input); ok != want {
			t.Fatalf("array index %q: %v", input, ok)
		}
	}
}

func BenchmarkQuoteStringUTF8(b *testing.B) {
	value := strings.Repeat("Tiếng Việt 🌏\n", 128)
	b.ReportAllocs()
	b.SetBytes(int64(len(value)))
	for i := 0; i < b.N; i++ {
		_ = QuoteString(value)
	}
}
