package sandbox

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestEvalSessionRegistryEvictsIdleAndCapacitySessions(t *testing.T) {
	r := NewEvalSessionRegistry(1, 20*time.Millisecond)
	t.Cleanup(r.Close)
	start := func() (*evalProcess, error) { return nil, nil }

	first, err := r.acquire("first", false, start)
	if err != nil {
		t.Fatal(err)
	}
	r.release("first", first)
	second, err := r.acquire("second", false, start)
	if err != nil {
		t.Fatal(err)
	}
	if !evalKernelClosed(first) {
		t.Fatal("capacity eviction did not close the least-recent idle kernel")
	}
	r.release("second", second)

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		_, present := r.sessions["second"]
		r.mu.Unlock()
		if !present {
			if !evalKernelClosed(second) {
				t.Fatal("idle eviction removed but did not close kernel")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("idle kernel was not evicted")
}

func TestEvalSessionRegistryDoesNotEvictActiveSession(t *testing.T) {
	r := NewEvalSessionRegistry(1, time.Hour)
	t.Cleanup(r.Close)
	start := func() (*evalProcess, error) { return nil, nil }
	first, _ := r.acquire("first", false, start)
	second, err := r.acquire("second", false, start)
	if err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	count := len(r.sessions)
	r.mu.Unlock()
	if count != 2 || evalKernelClosed(first) {
		t.Fatalf("active session was evicted: count=%d closed=%v", count, evalKernelClosed(first))
	}
	r.release("first", first)
	r.release("second", second)
}

func evalKernelClosed(kernel *evalKernel) bool {
	kernel.executionMu.Lock()
	defer kernel.executionMu.Unlock()
	return kernel.closed
}

func TestEvalSessionRegistryResetHasNoAcquireGap(t *testing.T) {
	r := NewEvalSessionRegistry(4, time.Hour)
	t.Cleanup(r.Close)
	start := func() (*evalProcess, error) { return nil, nil }
	old, err := r.acquire("same", false, start)
	if err != nil {
		t.Fatal(err)
	}
	r.release("same", old)
	// Simulate a cell still owning the execution lock while reset begins.
	old.executionMu.Lock()
	unlocked := false
	defer func() {
		if !unlocked {
			old.executionMu.Unlock()
		}
	}()

	var resetKernel, ordinaryKernel *evalKernel
	var resetErr, ordinaryErr error
	resetDone := make(chan struct{})
	go func() {
		resetKernel, resetErr = r.acquire("same", true, start)
		close(resetDone)
	}()
	deadline := time.Now().Add(time.Second)
	for {
		r.mu.Lock()
		_, present := r.sessions["same"]
		r.mu.Unlock()
		if !present {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("reset did not remove old session")
		}
		time.Sleep(time.Millisecond)
	}

	ordinaryDone := make(chan struct{})
	go func() {
		ordinaryKernel, ordinaryErr = r.acquire("same", false, start)
		close(ordinaryDone)
	}()
	select {
	case <-ordinaryDone:
		t.Fatal("ordinary acquire slipped into reset gap")
	case <-time.After(20 * time.Millisecond):
	}
	old.executionMu.Unlock()
	unlocked = true
	<-resetDone
	<-ordinaryDone
	if resetErr != nil || ordinaryErr != nil {
		t.Fatalf("reset err=%v ordinary err=%v", resetErr, ordinaryErr)
	}
	if resetKernel != ordinaryKernel {
		t.Fatal("ordinary acquire did not join the reset kernel")
	}
	r.release("same", resetKernel)
	r.release("same", ordinaryKernel)
}

func TestEvalSessionRegistryRejectsAcquireAfterClose(t *testing.T) {
	r := NewEvalSessionRegistry(1, time.Hour)
	r.Close()
	var starts atomic.Int32
	if _, err := r.acquire("closed", false, func() (*evalProcess, error) {
		starts.Add(1)
		return nil, nil
	}); err == nil {
		t.Fatal("closed registry accepted a new kernel")
	}
	if starts.Load() != 0 {
		t.Fatal("closed registry started a process")
	}
}

func TestEvalSessionRegistryCloseWinsAgainstKernelStartup(t *testing.T) {
	r := NewEvalSessionRegistry(1, time.Hour)
	started := make(chan struct{})
	unblock := make(chan struct{})
	errCh := make(chan error, 1)
	go func() {
		_, err := r.acquire("starting", false, func() (*evalProcess, error) {
			close(started)
			<-unblock
			return nil, nil
		})
		errCh <- err
	}()
	<-started
	r.Close()
	close(unblock)
	if err := <-errCh; err == nil {
		t.Fatal("kernel startup survived registry shutdown")
	}
}
