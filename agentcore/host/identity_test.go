package host

import (
	"encoding/json"
	"testing"
)

func TestPiJSONIdentityPreservesNumbersAcrossJSONBSpelling(t *testing.T) {
	for _, pair := range [][2]string{
		{`1e3`, `1000`}, {`1.000`, `1`}, {`4e-8`, `0.00000004`},
		{`1e21`, `1000000000000000000000`}, {`-0.000`, `0`},
		{`9.007199254740993e15`, `9007199254740993`},
		{`1e00000000000000000000000000000003`, `1000`},
		{`1e1000000000000000000000000`, `10e999999999999999999999999`},
	} {
		if !SameJSON([]byte(pair[0]), []byte(pair[1])) {
			t.Errorf("same number has different identity: %s / %s", pair[0], pair[1])
		}
	}
	for _, pair := range [][2]string{
		{`9007199254740992`, `9007199254740993`},
		{`0.123456789012345678901234567890`, `0.123456789012345678901234567891`},
		{`1e309`, `2e309`}, {`null`, `0`}, {`"1000"`, `1000`},
		{`{"extension":[9007199254740992]}`, `{"extension":[9007199254740993]}`},
		{`1 2`, `1`},
	} {
		if SameJSON([]byte(pair[0]), []byte(pair[1])) {
			t.Errorf("different values share an identity: %s / %s", pair[0], pair[1])
		}
	}
	// Native number spellings emitted by JSON.stringify / encoding/json keep
	// their previous digest representation, including very small cost fields.
	for _, number := range []float64{0, 1, -123, 0.1, 0.000001, 0.0000001, 4e-8, 1e20, 1e21, 1.25e30, 1.2345678901234567, 5e-324} {
		raw, _ := json.Marshal(number)
		canonical, err := CanonicalJSON(raw)
		if err != nil || string(canonical) != string(raw) {
			t.Errorf("ordinary native digest spelling changed: %s -> %s (%v)", raw, canonical, err)
		}
	}
}
