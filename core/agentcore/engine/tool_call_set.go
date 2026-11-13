package engine

import "slices"

// ToolCallSet is an immutable, insertion-ordered set of executing tool call IDs.
// Agent retains the same set between tool events and replaces it at every start,
// end, reset and run settlement. Retained sets can be read concurrently.
type ToolCallSet struct {
	values []string
}

func (s *ToolCallSet) Len() int {
	if s == nil {
		return 0
	}
	return len(s.values)
}

func (s *ToolCallSet) Has(id string) bool {
	return s != nil && slices.Contains(s.values, id)
}

// Values returns a detached slice in insertion order.
func (s *ToolCallSet) Values() []string {
	values := []string{}
	if s != nil {
		values = append(values, s.values...)
	}
	return values
}

// MarshalJSON matches JSON.stringify on Pi's Set. Transport adapters that need
// an array of IDs must explicitly serialize Values instead.
func (s *ToolCallSet) MarshalJSON() ([]byte, error) { return []byte("{}"), nil }

func (s *ToolCallSet) with(id string) *ToolCallSet {
	next := &ToolCallSet{values: s.Values()}
	if !next.Has(id) {
		next.values = append(next.values, id)
	}
	return next
}

func (s *ToolCallSet) without(id string) *ToolCallSet {
	next := &ToolCallSet{values: s.Values()}
	if index := slices.Index(next.values, id); index >= 0 {
		next.values = slices.Delete(next.values, index, index+1)
	}
	return next
}
