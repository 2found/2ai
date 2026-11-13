package protocol

import "context"

type Embedder interface {
	// Embed returns one vector per input string, index-aligned. All returned
	// vectors share the same dimension.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}
