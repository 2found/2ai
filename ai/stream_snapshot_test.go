package ai

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestAssistantStreamSnapshotDetachesEveryPublicField(t *testing.T) {
	// Include both union branches and mutable extension data. The reflection
	// walk below also exercises new exported fields when this input grows.
	var message Message
	if err := json.Unmarshal([]byte(`{
 "role":"assistant","timestamp":1,"api":"test","provider":"test","model":"test",
 "content":[{"type":"toolCall","id":"call","name":"echo","arguments":{"value":1},"textSignature":"text","thinkingSignature":"thinking","thoughtSignature":"thought","namespace":"tools","redacted":true,"extension":{"value":2}},null],
 "sections":{"first":"policy"},"toolsAdded":[{"name":"echo","parameters":{"type":"object"},"constrainedSampling":{"mode":"grammar"},"extension":{"value":3}}],"toolsRemoved":[{"name":"old"}],
 "responseModel":"response-model","responseId":"response-id","providerThinkingLevel":"high","thinkingLevel":"low","errorMessage":"error","rawStopReason":"raw","endTurn":true,
 "diagnostics":{"value":4},"deferred":{"value":5},"details":{"value":6},"nestedCalls":[{"value":7}],
 "usage":{"input":1,"output":2,"cacheRead":3,"cacheWrite":4,"cacheWrite1h":5,"reasoning":6,"totalTokens":7,"cost":{"input":1,"output":2,"cacheRead":3,"cacheWrite":4,"total":10},"extension":{"value":8}},"extension":{"value":9}
}`), &message); err != nil {
		t.Fatal(err)
	}
	text := &Message{Role: "user", Content: TextContent("original"), Extra: map[string]json.RawMessage{}}
	source := AssistantMessageEvent{Extra: map[string]json.RawMessage{"extension": json.RawMessage(`{"value":10}`)}, Type: "toolcall_end", Partial: &message, Message: text, Error: &message, ToolCall: message.Content.Blocks.Get(0)}
	before, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	stream := NewAssistantMessageEventStream()
	snapshot, err := stream.SnapshotEvent(source)
	if err != nil {
		t.Fatal(err)
	}
	copied, err := json.Marshal(snapshot)
	if err != nil || string(copied) != string(before) {
		t.Fatalf("snapshot changed transcript wire shape: %s, %v", copied, err)
	}
	// Mutate all public references, including map values and byte slices.
	// Private encoding metadata is immutable and intentionally not exposed.
	seen := map[any]bool{}
	var mutate func(reflect.Value)
	mutate = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Interface:
			if !v.IsNil() {
				mutate(v.Elem())
			}
		case reflect.Pointer:
			if v.IsNil() || seen[v.Interface()] {
				return
			}
			seen[v.Interface()] = true
			if object, ok := v.Interface().(*Object); ok {
				object.Set("snapshotEdit", true)
				return
			}
			if array, ok := v.Interface().(*Array); ok {
				array.Append("snapshotEdit")
				return
			}
			mutate(v.Elem())
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if v.Type().Field(i).IsExported() {
					mutate(v.Field(i))
				}
			}
		case reflect.Slice:
			for i := 0; i < v.Len(); i++ {
				mutate(v.Index(i))
			}
		case reflect.Map:
			if v.IsNil() {
				return
			}
			entries := v.MapRange()
			for entries.Next() {
				mutate(entries.Value())
			}
			v.SetMapIndex(reflect.ValueOf("snapshotEdit"), reflect.ValueOf(json.RawMessage(`true`)))
		case reflect.String:
			v.SetString("snapshot")
		case reflect.Bool:
			v.SetBool(!v.Bool())
		case reflect.Int64:
			v.SetInt(v.Int() + 1)
		case reflect.Uint8:
			v.SetUint('x')
		case reflect.Float64:
			v.SetFloat(v.Float() + 1)
		}
	}
	mutate(reflect.ValueOf(&snapshot))
	after, err := json.Marshal(source)
	if err != nil || string(after) != string(before) {
		t.Fatalf("snapshot mutation reached producer-owned data: %s, %v", after, err)
	}
}

func TestAssistantStreamSnapshotDoesNotSerialize(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	message := &Message{Role: "assistant", Content: BlockReferences(&ContentBlock{Type: "toolCall", Arguments: json.RawMessage(`{`)}),
		Usage: &Usage{Input: math.NaN(), Output: math.Inf(1), Reasoning: &negativeZero, Cost: UsageCost{Total: math.Inf(-1)}},
		Extra: map[string]json.RawMessage{"unfinished": json.RawMessage(`[`)}}
	stream := NewAssistantMessageEventStream()
	snapshot, err := stream.SnapshotEvent(AssistantMessageEvent{Type: "start", Partial: message})
	if err != nil {
		t.Fatal("snapshot introduced a serialization failure", err)
	}
	u := snapshot.Partial.Usage
	if !math.IsNaN(u.Input) || !math.IsInf(u.Output, 1) || !math.IsInf(u.Cost.Total, -1) || !math.Signbit(*u.Reasoning) {
		t.Fatal("snapshot coerced live numeric values")
	}
	message.Content.Blocks.Get(0).Arguments[0] = 'x'
	message.Extra["unfinished"][0] = 'x'
	*message.Usage.Reasoning = 1
	if string(snapshot.Partial.Content.Blocks.Get(0).Arguments) != "{" || string(snapshot.Partial.Extra["unfinished"]) != "[" || !math.Signbit(*u.Reasoning) {
		t.Fatal("snapshot retained mutable JSON or optional number fields")
	}
	if _, err := json.Marshal(snapshot); err == nil {
		t.Fatal("explicit JSON export unexpectedly accepted unserializable data")
	}
}

func TestAssistantStreamSnapshotDetachesLiveDetailsGraph(t *testing.T) {
	child := NewObject(Property{Name: "negativeZero", Value: math.Copysign(0, -1)}, Property{Name: "infinity", Value: math.Inf(1)})
	array := NewArray(child)
	array.SetLength(4)
	array.Set(2, child)
	array.Set(3, Undefined)
	array.SetProperty("meta", child)
	array.SetProperty("self", array)
	calls := 0
	array.SetProperty("fn", func() { calls++ })
	native := map[string]any{"child": child, "bytes": []byte{1, 2}}
	native["self"] = native
	root := NewObject(Property{Name: "array", Value: array}, Property{Name: "native", Value: native})
	root.Set("self", root)
	first := &Message{Role: "assistant", Details: root}
	second := &Message{Role: "assistant", Details: root}
	stream := NewAssistantMessageEventStream()
	copied, err := stream.SnapshotEvent(AssistantMessageEvent{Type: "start", Partial: first, Message: second, Error: first})
	if err != nil {
		t.Fatal(err)
	}
	details := copied.Partial.Details.(*Object)
	if details == root || copied.Message.Details != details || copied.Error != copied.Partial || details.Get("self") != details {
		t.Fatal("snapshot lost detachment or shared/cyclic identity")
	}
	clonedArray := details.Get("array").(*Array)
	clonedChild := clonedArray.Get(0).(*Object)
	clonedNative := details.Get("native").(map[string]any)
	if clonedArray == array || clonedChild == child || clonedArray.Get(2) != clonedChild || clonedNative["child"] != clonedChild {
		t.Fatal("nested graph lost detachment or repeated reference")
	}
	if clonedArray.Len() != 4 || clonedArray.Has(1) || !clonedArray.Has(3) || clonedArray.Get(3) != Undefined {
		t.Fatal("snapshot changed sparse array shape")
	}
	namedChild, childFound := clonedArray.GetProperty("meta")
	self, selfFound := clonedArray.GetProperty("self")
	function, functionFound := clonedArray.GetProperty("fn")
	if !childFound || namedChild != clonedChild || !selfFound || self != clonedArray || !functionFound || calls != 0 {
		t.Fatal("snapshot lost named array graph or evaluated a property function")
	}
	if _, ok := function.(func()); !ok {
		t.Fatal("snapshot discarded a named function value")
	}
	clonedArray.DeleteProperty("meta")
	if value, found := array.GetProperty("meta"); !found || value != child {
		t.Fatal("snapshot property deletion reached source")
	}
	if !math.Signbit(clonedChild.Get("negativeZero").(float64)) || !math.IsInf(clonedChild.Get("infinity").(float64), 1) {
		t.Fatal("snapshot serialized live numbers")
	}
	clonedChild.Set("changed", true)
	clonedArray.Append("appended")
	clonedNative["bytes"].([]byte)[0] = 9
	clonedNative["self"].(map[string]any)["changed"] = true
	if _, changed := child.Lookup("changed"); changed || array.Len() != 4 || native["bytes"].([]byte)[0] != 1 {
		t.Fatal("snapshot mutation reached original details")
	}
	if native["changed"] != nil || clonedNative["changed"] != true {
		t.Fatal("native map cycle was shared with the source or broken")
	}
}
