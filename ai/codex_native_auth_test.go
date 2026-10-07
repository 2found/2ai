package ai

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestPiCodexAuthOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-auth.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Accounts       []struct {
			Token          string
			Account, Error *string
		}
		URLs []struct {
			Base *string
			URL  string
		}
		Headers []struct {
			Input struct {
				Name                string
				Initial, Additional json.RawMessage
				Session             *string
				Websocket           bool
			}
			Expected map[string]string
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Accounts)+len(fixture.URLs)+len(fixture.Headers) != 44 {
		t.Fatal("unexpected auth oracle revision/coverage")
	}
	for i, tc := range fixture.Accounts {
		t.Run(fmt.Sprintf("account/%d", i), func(t *testing.T) {
			got, err := codexNativeAccountID(tc.Token)
			if tc.Error != nil {
				if err == nil || err.Error() != *tc.Error {
					t.Fatalf("error=%v, want %s", err, *tc.Error)
				}
				return
			}
			if err != nil || tc.Account == nil || got != *tc.Account {
				t.Fatalf("account=%q error=%v want=%v", got, err, tc.Account)
			}
		})
	}
	for i, tc := range fixture.URLs {
		t.Run(fmt.Sprintf("url/%d", i), func(t *testing.T) {
			base := ""
			if tc.Base != nil {
				base = *tc.Base
			}
			if got := codexNativeURL(base); got != tc.URL {
				t.Fatalf("url=%q, want %q", got, tc.URL)
			}
		})
	}
	for _, tc := range fixture.Headers {
		t.Run(fmt.Sprintf("headers/%s/ws=%v", tc.Input.Name, tc.Input.Websocket), func(t *testing.T) {
			headers, err := codexNativeHeaders(tc.Input.Initial, tc.Input.Additional, "account", "token", tc.Input.Session, tc.Input.Websocket)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(headers.Get("User-Agent"), "pi (") {
				t.Fatal("lost platform header")
			}
			headers.Set("User-Agent", "<platform>")
			got := map[string]string{}
			for name, values := range headers {
				got[strings.ToLower(name)] = strings.Join(values, ", ")
			}
			if !reflect.DeepEqual(got, tc.Expected) {
				t.Fatalf("headers=%v, want %v", got, tc.Expected)
			}
		})
	}
}
