package memory

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRedactSecretsKeepsUsefulFacts(t *testing.T) {
	secrets := []string{"ghp_abcdefghijklmnopqrstuvwxyz1234567890", "Bearer abcdefghijklmnop1234", "password=supersecretvalue", "postgres://user:password@localhost/db", "sk-proj-abcdefghijklmnopqrstuv1234"}
	for _, secret := range secrets {
		clean := RedactSecrets("endpoint /status/ready " + secret)
		if clean == "endpoint /status/ready "+secret || !strings.Contains(clean, "/status/ready") {
			t.Fatal("credential not removed or fact damaged", clean)
		}
	}
	safe := "Long prefers Vietnamese; staging uses /status/ready."
	if RedactSecrets(safe) != safe {
		t.Fatal("ordinary user preference damaged")
	}
}
func TestRecoveryWaitsForCapacityAndDrainsWithoutNewRuns(t *testing.T) {
	w := NewConsolidationWorker(1)
	w.coalesceDelay = 0
	slow, second := newProgressStore(1), newProgressStore(9)
	entered, release := make(chan struct{}), make(chan struct{})
	w.submit(progressCuration("slow", slow, func(ctx context.Context, _ Consolidation) ([]Change, error) {
		close(entered)
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return nil, nil
	}))
	runProgressWorker(t, w)
	awaitProgress(t, entered)
	admitted := make(chan struct{})
	go func() {
		defer close(admitted)
		if err := w.Recover(context.Background(), Plugin{Store: second, Consolidator: func(context.Context, Consolidation) ([]Change, error) { return nil, nil }}, "second"); err != nil {
			t.Error(err)
		}
	}()
	select {
	case <-admitted:
		t.Fatal("recovery bypassed full admission bound")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	awaitProgress(t, admitted)
	for i := 0; i < 3; i++ {
		awaitProgress(t, second.committed)
	}
	pending, consumed, reads := second.counts()
	if pending != 0 || consumed != 9 {
		t.Fatal("recovery did not drain", pending, consumed)
	}
	time.Sleep(25 * time.Millisecond)
	_, _, later := second.counts()
	if later != reads {
		t.Fatal("idle recovery polled storage")
	}
}
