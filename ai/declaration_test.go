package ai

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPiDeclarationJSONOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-declarations.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Name, Field, Left, Right string
			Expected                 struct {
				Left, Right    string
				Equal, Changed bool
			}
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 24 {
		t.Fatal("unexpected declaration oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			tool := func(raw string) Tool {
				tool := Tool{Name: "tool", Description: "test", Parameters: json.RawMessage(`{"type":"object"}`)}
				if tc.Field == "parameters" {
					tool.Parameters = json.RawMessage(raw)
				} else {
					tool.ConstrainedSampling = json.RawMessage(raw)
				}
				return tool
			}
			a, b := tool(tc.Left), tool(tc.Right)
			value := func(tool Tool) string {
				declared := ToToolDeclaration(tool)
				if tc.Field == "parameters" {
					return string(declared.Parameters)
				}
				return string(declared.ConstrainedSampling)
			}
			actual := tc.Expected
			actual.Left, actual.Right = value(a), value(b)
			actual.Equal = DeclarationsEqual(a, b)
			actual.Changed = len(GetToolStateChanges([]Tool{a}, []Tool{b}).ToolsAdded) > 0
			if !reflect.DeepEqual(actual, tc.Expected) {
				t.Fatalf("Go: %+v\nPi: %+v", actual, tc.Expected)
			}
		})
	}
}
