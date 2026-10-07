package sandbox

import (
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/2found/2ai/agentcore"
)

func TestLSPSessionRegistryEvictsIdleAndCapacityClients(t *testing.T) {
	r := NewLSPSessionRegistry(1, 20*time.Millisecond)
	t.Cleanup(r.Close)
	start := func() (*lspClient, error) { return &lspClient{}, nil }

	first, err := r.acquire("first", start)
	if err != nil {
		t.Fatal(err)
	}
	r.release("first", first)
	second, err := r.acquire("second", start)
	if err != nil {
		t.Fatal(err)
	}
	if !lspClientClosed(first) {
		t.Fatal("capacity eviction did not close least-recent idle client")
	}
	r.release("second", second)

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		_, present := r.clients["second"]
		r.mu.Unlock()
		if !present {
			if !lspClientClosed(second) {
				t.Fatal("idle eviction removed but did not close client")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("idle client was not evicted")
}

func TestLSPSessionRegistryDoesNotEvictActiveClient(t *testing.T) {
	r := NewLSPSessionRegistry(1, time.Hour)
	t.Cleanup(r.Close)
	start := func() (*lspClient, error) { return &lspClient{}, nil }
	first, _ := r.acquire("first", start)
	second, err := r.acquire("second", start)
	if err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	count := len(r.clients)
	r.mu.Unlock()
	if count != 2 || lspClientClosed(first) {
		t.Fatalf("active client was evicted: count=%d closed=%v", count, lspClientClosed(first))
	}
	r.release("first", first)
	r.release("second", second)
}

func lspClientClosed(client *lspClient) bool {
	client.executionMu.Lock()
	defer client.executionMu.Unlock()
	return client.closed
}

func TestLSPSessionRegistryInvalidationFinishesBeforeSameKeyRestart(t *testing.T) {
	r := NewLSPSessionRegistry(2, time.Hour)
	t.Cleanup(r.Close)
	closing := make(chan struct{})
	unblock := make(chan struct{})
	old, err := r.acquire("same", func() (*lspClient, error) {
		return &lspClient{cleanup: func() {
			close(closing)
			<-unblock
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	r.release("same", old)
	invalidated := make(chan struct{})
	go func() {
		r.invalidate("same", old)
		close(invalidated)
	}()
	<-closing

	var starts atomic.Int32
	acquired := make(chan *lspClient, 1)
	go func() {
		client, _ := r.acquire("same", func() (*lspClient, error) {
			starts.Add(1)
			return &lspClient{}, nil
		})
		acquired <- client
	}()
	select {
	case <-acquired:
		t.Fatal("same-key replacement started before invalidation completed")
	case <-time.After(20 * time.Millisecond):
	}
	if starts.Load() != 0 {
		t.Fatal("replacement start overlapped old client close")
	}
	close(unblock)
	<-invalidated
	client := <-acquired
	if starts.Load() != 1 {
		t.Fatalf("replacement starts = %d", starts.Load())
	}
	r.release("same", client)
}

func TestLSPSessionRegistryRejectsAcquireAfterClose(t *testing.T) {
	r := NewLSPSessionRegistry(1, time.Hour)
	r.Close()
	var starts atomic.Int32
	if _, err := r.acquire("closed", func() (*lspClient, error) {
		starts.Add(1)
		return &lspClient{}, nil
	}); err == nil {
		t.Fatal("closed registry accepted a new client")
	}
	if starts.Load() != 0 {
		t.Fatal("closed registry started a process")
	}
}

func TestLSPSessionRegistryCloseWinsAgainstClientStartup(t *testing.T) {
	r := NewLSPSessionRegistry(1, time.Hour)
	started := make(chan struct{})
	unblock := make(chan struct{})
	cleaned := make(chan struct{})
	errCh := make(chan error, 1)
	go func() {
		_, err := r.acquire("starting", func() (*lspClient, error) {
			close(started)
			<-unblock
			return &lspClient{cleanup: func() { close(cleaned) }}, nil
		})
		errCh <- err
	}()
	<-started
	r.Close()
	close(unblock)
	if err := <-errCh; err == nil {
		t.Fatal("client startup survived registry shutdown")
	}
	select {
	case <-cleaned:
	case <-time.After(time.Second):
		t.Fatal("client created during shutdown was not closed")
	}
}

type controlledLSPProcess struct {
	exited   chan struct{}
	killed   chan struct{}
	exitOnce atomic.Bool
}

func newControlledLSPProcess() *controlledLSPProcess {
	return &controlledLSPProcess{exited: make(chan struct{}), killed: make(chan struct{})}
}

func (p *controlledLSPProcess) Stdin() io.WriteCloser { return nopWriteCloser{Writer: io.Discard} }
func (p *controlledLSPProcess) Stdout() io.ReadCloser { return io.NopCloser(&emptyReader{}) }
func (p *controlledLSPProcess) Stderr() io.ReadCloser { return io.NopCloser(&emptyReader{}) }
func (p *controlledLSPProcess) Wait() (agentcore.SandboxResult, error) {
	<-p.exited
	return agentcore.SandboxResult{}, nil
}
func (p *controlledLSPProcess) Kill() error {
	select {
	case <-p.killed:
	default:
		close(p.killed)
	}
	return nil
}
func (p *controlledLSPProcess) exit() {
	if p.exitOnce.CompareAndSwap(false, true) {
		close(p.exited)
	}
}

type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }

type emptyReader struct{}

func (*emptyReader) Read([]byte) (int, error) { return 0, io.EOF }

func TestLSPSessionRegistryBlocksReplacementUntilProcessExitConfirmed(t *testing.T) {
	r := NewLSPSessionRegistry(2, time.Hour)
	t.Cleanup(r.Close)
	process := newControlledLSPProcess()
	cleaned := make(chan struct{})
	client, err := r.acquire("same", func() (*lspClient, error) {
		return &lspClient{
			process: process, shutdownTimeout: 10 * time.Millisecond,
			cleanup: func() { close(cleaned) },
		}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	r.release("same", client)
	r.invalidate("same", client)
	select {
	case <-process.killed:
	default:
		t.Fatal("unresponsive process was not killed")
	}
	select {
	case <-cleaned:
		t.Fatal("process resources were cleaned before exit was confirmed")
	default:
	}

	var starts atomic.Int32
	if _, err := r.acquire("same", func() (*lspClient, error) {
		starts.Add(1)
		return &lspClient{}, nil
	}); !errors.Is(err, errLSPClientReaping) {
		t.Fatalf("replacement error = %v, want %v", err, errLSPClientReaping)
	}
	if starts.Load() != 0 {
		t.Fatal("replacement process started before old process exit")
	}

	process.exit()
	select {
	case <-cleaned:
	case <-time.After(time.Second):
		t.Fatal("confirmed process exit did not release resources")
	}
	replacement, err := r.acquire("same", func() (*lspClient, error) {
		starts.Add(1)
		return &lspClient{}, nil
	})
	if err != nil {
		t.Fatalf("replacement after exit: %v", err)
	}
	if starts.Load() != 1 {
		t.Fatalf("replacement starts = %d", starts.Load())
	}
	r.release("same", replacement)
}
