package jsonjs

import "testing"

func TestObjectFieldsCanContinueAfterSerialization(t *testing.T) {
	var fields ObjectFields
	fields.Set([]byte(`"b"`), []byte(`1`))
	fields.Set([]byte(`"2"`), []byte(`2`))
	if got := string(fields.Marshal()); got != `{"2":2,"b":1}` {
		t.Fatal(got)
	}
	fields.Set([]byte(`"b"`), []byte(`3`))
	fields.Set([]byte(`"1"`), []byte(`4`))
	fields.Set([]byte(`"a"`), []byte(`5`))
	if got := string(fields.Marshal()); got != `{"1":4,"2":2,"b":3,"a":5}` {
		t.Fatal("serialization invalidated duplicate-key positions:", got)
	}
}
