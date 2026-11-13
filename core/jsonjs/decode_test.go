package jsonjs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

func TestDecodeJSONStringUTF16Oracle(t *testing.T) {
	// Reuse the source-runtime hashes for all 65,536 individual code units
	// and 1,048,576 valid surrogate pairs; no JavaScript runtime is needed.
	raw, err := os.ReadFile("../ai/testdata/pi-json-stringify.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit, BunVersion  string
		SingleUnits, SurrogatePairs []string
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || fixture.BunVersion != "1.3.14" || len(fixture.SingleUnits) != 256 || len(fixture.SurrogatePairs) != 1024 {
		t.Fatal("unexpected UTF-16 oracle coverage")
	}
	check := func(name string, hashes []string, inputs func(int, int) []byte, count int) {
		t.Run(name, func(t *testing.T) {
			for page, expected := range hashes {
				hash := sha256.New()
				for offset := 0; offset < count; offset++ {
					value, err := DecodeJSON(inputs(page, offset))
					if err != nil {
						t.Fatal(err)
					}
					hash.Write(QuoteString(value.(string)))
					hash.Write([]byte{'\n'})
				}
				if actual := hex.EncodeToString(hash.Sum(nil)); actual != expected {
					t.Fatalf("page %x: got %s; want %s", page, actual, expected)
				}
			}
		})
	}
	check("single units", fixture.SingleUnits, func(page, offset int) []byte {
		return fmt.Appendf(nil, `"\u%04x"`, page*256+offset)
	}, 256)
	check("single units with raw surrogates", fixture.SingleUnits, func(page, offset int) []byte {
		unit := page*256 + offset
		if unit >= 0xd800 && unit <= 0xdfff {
			return []byte{'"', 0xe0 | byte(unit>>12), 0x80 | byte(unit>>6&0x3f), 0x80 | byte(unit&0x3f), '"'}
		}
		return fmt.Appendf(nil, `"\u%04x"`, unit)
	}, 256)
	check("surrogate pairs", fixture.SurrogatePairs, func(page, offset int) []byte {
		return fmt.Appendf(nil, `"\u%04x\u%04x"`, 0xd800+page, 0xdc00+offset)
	}, 1024)
}

func TestDecodeJSONRepresentation(t *testing.T) {
	for raw, expected := range map[string]string{
		`"\ud800"`:       "\xed\xa0\x80",
		`"\ud801"`:       "\xed\xa0\x81",
		`"\udfff"`:       "\xed\xbf\xbf",
		`"\ufffd"`:       "\xef\xbf\xbd",
		`"\ud83d\ude00"`: "😀",
		`"\udc00\ud800"`: "\xed\xb0\x80\xed\xa0\x80",
	} {
		value, err := DecodeJSON([]byte(raw))
		if err != nil || value != expected {
			t.Fatalf("%s: %#v, %v", raw, value, err)
		}
	}
	value, err := DecodeJSON([]byte(`{"\ud800":1,"\ud801":2,"\ufffd":3,"numbers":[1e400,-1e400,-1e-400,9007199254740993],"duplicate":1,"duplicate":2,"null":null,"bool":true}`))
	if err != nil {
		t.Fatal(err)
	}
	object := value.(map[string]any)
	if len(object) != 7 || object["\xed\xa0\x80"] != float64(1) || object["\xed\xa0\x81"] != float64(2) || object["�"] != float64(3) || object["duplicate"] != float64(2) || object["null"] != nil || object["bool"] != true {
		t.Fatal("decoded object lost key/value identity:", object)
	}
	numbers := object["numbers"].([]any)
	if !math.IsInf(numbers[0].(float64), 1) || !math.IsInf(numbers[1].(float64), -1) || !math.Signbit(numbers[2].(float64)) || numbers[3] != float64(9007199254740992) {
		t.Fatal("decoded numbers differ:", numbers)
	}
	for _, raw := range []string{"", " ", "NaN", "Infinity", "1 2", `[1,]`, `{"x":}`, `"\uXYZW"`, "\"unterminated"} {
		if _, err := DecodeJSON([]byte(raw)); err == nil {
			t.Fatal("accepted invalid JSON:", raw)
		}
	}
}

func TestDecodeObjectPropertiesRetainsRawValues(t *testing.T) {
	raw := []byte(`{"\ud800":1e400,"\ud801":-0,"\ud800": [1,{"x":-1e999}]}`)
	properties, err := DecodeObjectProperties(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(properties) != 3 {
		t.Fatalf("lost duplicate property: %+v", properties)
	}
	wants := []struct{ key, value string }{
		{`"\ud800"`, "1e400"},
		{`"\ud801"`, "-0"},
		{`"\ud800"`, `[1,{"x":-1e999}]`},
	}
	for i, want := range wants {
		if string(QuoteString(properties[i].Name)) != want.key || string(properties[i].Value) != want.value {
			t.Fatalf("property %d: key=%s value=%s", i, QuoteString(properties[i].Name), properties[i].Value)
		}
	}
	for _, malformed := range []string{"", `{"a":}`, `{"a":1} trailing`} {
		if fields, err := DecodeObjectProperties([]byte(malformed)); err == nil || fields != nil {
			t.Fatalf("accepted malformed object %q: %+v %v", malformed, fields, err)
		}
	}
	for _, other := range []string{"null", "[]", "1", `"text"`} {
		if fields, err := DecodeObjectProperties([]byte(other)); err != nil || len(fields) != 0 {
			t.Fatalf("non-object %q: %+v %v", other, fields, err)
		}
	}
}
