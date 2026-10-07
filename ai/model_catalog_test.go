package ai

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/2found/2ai/jsonjs"
)

type catalogFixture struct {
	UpstreamCommit string
	Cases          []struct {
		Name, Kind string
		Input      json.RawMessage
		Output     *string
		Error      *string
	}
	Mutations []struct {
		Kind, Output string
		Same         bool
	}
	StoreCases []struct {
		Entry json.RawMessage
		Steps json.RawMessage
	}
	StoreTrace  json.RawMessage
	NativeCases []struct {
		Mode  string
		Keys  []string
		Error *string
	}
}

func readCatalogFixture(t *testing.T) catalogFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/pi-model-catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture catalogFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 714 || len(fixture.NativeCases) != 9 || len(fixture.Mutations) != 3 || len(fixture.StoreCases) != 9 {
		t.Fatal("unexpected catalog oracle coverage")
	}
	return fixture
}

func catalogDecode(t *testing.T, raw []byte) any {
	t.Helper()
	value, err := jsonjs.DecodeValue(raw)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func catalogStringify(t *testing.T, value any) string {
	t.Helper()
	raw, err := jsonjs.StringifyValue(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func catalogFunction(kind string) func(string, any) (*Object, error) {
	switch kind {
	case "Chat":
		return FlattenChatModelCatalog
	case "Image":
		return FlattenImageModelCatalog
	case "Classifier":
		return FlattenClassifierModelCatalog
	default:
		panic("unknown catalog kind")
	}
}

func checkCatalogError(t *testing.T, err error, expected *string) {
	t.Helper()
	if expected == nil {
		if err != nil {
			t.Fatal(err)
		}
	} else if err == nil || err.Error() != *expected {
		t.Fatalf("Go error %v, Pi %q", err, *expected)
	}
}

func TestPiModelCatalog(t *testing.T) {
	fixture := readCatalogFixture(t)
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			input := catalogDecode(t, tc.Input)
			before := catalogStringify(t, input)
			result, err := catalogFunction(tc.Kind)("ignored", input)
			checkCatalogError(t, err, tc.Error)
			if tc.Output != nil {
				if actual := catalogStringify(t, result); actual != *tc.Output {
					t.Fatalf("Go %s, Pi %s", actual, *tc.Output)
				}
			}
			if catalogStringify(t, input) != before {
				t.Fatal("projection mutated input")
			}
		})
	}
	for _, tc := range fixture.Mutations {
		t.Run(tc.Kind+"/reference", func(t *testing.T) {
			kind := map[string]string{"Chat": "chat", "Image": "image", "Classifier": "classifier"}[tc.Kind]
			meta := NewObject(Property{Name: "x", Value: 1})
			model := NewObject(Property{Name: "type", Value: kind}, Property{Name: "id", Value: "a"}, Property{Name: "meta", Value: meta})
			group := NewObject(Property{Name: "a", Value: model})
			result, err := catalogFunction(tc.Kind)("ignored", NewObject(Property{Name: "one", Value: group}))
			if err != nil {
				t.Fatal(err)
			}
			meta.Set("x", 2)
			group.Set("a", NewObject(Property{Name: "type", Value: kind}, Property{Name: "id", Value: "b"}))
			if same := result.Get("a") == model; same != tc.Same {
				t.Fatal("model identity changed")
			}
			if actual := catalogStringify(t, result); actual != tc.Output {
				t.Fatalf("Go %s, Pi %s", actual, tc.Output)
			}
		})
	}
}

func TestPiModelCatalogNativeValues(t *testing.T) {
	for _, tc := range readCatalogFixture(t).NativeCases {
		t.Run(tc.Mode, func(t *testing.T) {
			group := NewArray()
			model := NewObject(Property{Name: "type", Value: "chat"}, Property{Name: "id", Value: "m"})
			var groups any = NewObject(Property{Name: "api", Value: group})
			switch tc.Mode {
			case "undefined-groups":
				groups = Undefined
			case "undefined-model":
				group.Append(Undefined)
			case "sparse":
				group.SetLength(4)
				group.Set(2, model)
			case "named":
				group.SetProperty("extra", model)
				group.Set(3, NewObject(Property{Name: "type", Value: "chat"}, Property{Name: "id", Value: "indexed"}))
			case "nonfinite":
				for _, id := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1)} {
					group.Append(NewObject(Property{Name: "type", Value: "chat"}, Property{Name: "id", Value: id}))
				}
			case "recursive-id":
				id := NewArray()
				id.Append(id)
				model.Set("id", id)
				group.Append(model)
			case "undefined-id":
				model.Set("id", Undefined)
				group.Append(model)
			case "array-id-join", "array-id-toString":
				id := NewArray("x")
				property := "join"
				if tc.Mode == "array-id-toString" {
					property = "toString"
				}
				id.SetProperty(property, Null)
				model.Set("id", id)
				group.Append(model)
			default:
				t.Fatal("unhandled fixture")
			}
			result, err := FlattenChatModelCatalog("ignored", groups)
			checkCatalogError(t, err, tc.Error)
			if err == nil {
				keys := []string{}
				for _, property := range result.Entries() {
					keys = append(keys, property.Name)
				}
				if catalogStringify(t, keys) != catalogStringify(t, tc.Keys) {
					t.Fatalf("keys %v, Pi %v", keys, tc.Keys)
				}
			}
		})
	}
}
