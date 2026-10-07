package agentcore_test

import (
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/testing/agentcore"
)

func TestSessionStoreConformanceMemory(t *testing.T) {
	store := agentcore.NewMemorySessionStore()
	var next atomic.Uint64
	agentcoretest.RunSessionStoreConformance(t, agentcoretest.SessionStoreHarness{
		Store: store,
		NewSessionID: func(*testing.T) string {
			return fmt.Sprintf("memory-conformance-%d", next.Add(1))
		},
	})
}
