package agentcore

import (
	"math"
	"testing"
)

// TestCosine pins the three anchors of cosine similarity plus the
// degrade-to-zero guards (mismatched length, zero magnitude) that keep ranking
// from producing NaN.
func TestCosine(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
		{"scaled-identical", []float32{1, 1, 1}, []float32{4, 4, 4}, 1},
		{"length-mismatch", []float32{1, 2}, []float32{1, 2, 3}, 0},
		{"zero-vector", []float32{0, 0}, []float32{1, 1}, 0},
		{"empty", nil, nil, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Cosine(tc.a, tc.b)
			if math.IsNaN(got) {
				t.Fatalf("Cosine returned NaN")
			}
			if math.Abs(got-tc.want) > 1e-9 {
				t.Fatalf("Cosine(%v,%v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestCosineRanksSemanticNeighborFirst verifies the ordering property recall
// depends on: a query vector scores its near-neighbor above an unrelated vector.
func TestCosineRanksSemanticNeighborFirst(t *testing.T) {
	query := []float32{1, 1, 0}
	near := []float32{0.9, 1.1, 0.0}
	far := []float32{-1, 0, 1}
	if Cosine(query, near) <= Cosine(query, far) {
		t.Fatalf("expected near neighbor to outrank far one")
	}
}
