package host

import (
	"context"
	"encoding/json"
)

// testTransform is the request view under the compactor's own policy.
func (c *Compactor) testTransform(ctx context.Context, raw json.RawMessage) json.RawMessage {
	return c.testTransformWith(ctx, raw, c.config)
}

func (c *Compactor) testTransformWith(ctx context.Context, raw json.RawMessage, policy CompactionPolicy) json.RawMessage {
	_, out, _ := c.transform(ctx, raw, policy, false)
	return out
}
