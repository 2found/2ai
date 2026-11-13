package ai

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

func TestPiJSONStringifyUTF16Oracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-json-stringify.json")
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
	t.Run("single units", func(t *testing.T) {
		for page, expected := range fixture.SingleUnits {
			hash := sha256.New()
			for offset := 0; offset < 256; offset++ {
				encoded, err := StringifyJSON(fmt.Appendf(nil, `"\u%04x"`, page*256+offset))
				if err != nil {
					t.Fatal(err)
				}
				hash.Write(encoded)
				hash.Write([]byte{'\n'})
			}
			if actual := hex.EncodeToString(hash.Sum(nil)); actual != expected {
				t.Fatalf("UTF-16 page %02x: got %s; want %s", page, actual, expected)
			}
		}
	})
	t.Run("surrogate pairs", func(t *testing.T) {
		for offset, expected := range fixture.SurrogatePairs {
			high := 0xd800 + offset
			hash := sha256.New()
			for low := 0xdc00; low <= 0xdfff; low++ {
				encoded, err := StringifyJSON(fmt.Appendf(nil, `"\u%04x\u%04x"`, high, low))
				if err != nil {
					t.Fatal(err)
				}
				hash.Write(encoded)
				hash.Write([]byte{'\n'})
			}
			if actual := hex.EncodeToString(hash.Sum(nil)); actual != expected {
				t.Fatalf("surrogate high unit %04x: got %s; want %s", high, actual, expected)
			}
		}
	})
}
