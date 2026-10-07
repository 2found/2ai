package sandbox

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

const (
	defaultEvalIdleTimeout = 15 * time.Minute
	evalProcessLifetime    = time.Hour
)

var errEvalKernelClosed = errors.New("eval kernel is closed")

// EvalSessionRegistry retains language runtimes across the short-lived tool
// instances built for consecutive turns of one conversation. Capacity is a
// soft limit: active sessions are never killed to make room for another call.
type EvalSessionRegistry struct {
	mu       sync.Mutex
	gates    [64]sync.Mutex
	sessions map[string]*evalKernel
	capacity int
	idle     time.Duration
	now      func() time.Time
	closed   bool
}

func NewEvalSessionRegistry(capacity int, idle time.Duration) *EvalSessionRegistry {
	if capacity <= 0 {
		capacity = defaultEvalRegistryCapacity
	}
	if idle <= 0 {
		idle = defaultEvalIdleTimeout
	}
	return &EvalSessionRegistry{
		sessions: make(map[string]*evalKernel), capacity: capacity, idle: idle, now: time.Now,
	}
}

type evalRegistryEntry struct {
	refs     int
	lastUsed time.Time
	timer    *time.Timer
}

// evalKernel embeds its registry bookkeeping so one mutex protects map
// membership, reference counts, and idle timers. executionMu separately
// serializes cells without holding the registry lock while user code runs.
type evalKernel struct {
	executionMu sync.Mutex
	entry       evalRegistryEntry
	process     *evalProcess
	closed      bool
}

func (r *EvalSessionRegistry) acquire(key string, reset bool, start func() (*evalProcess, error)) (*evalKernel, error) {
	if r == nil {
		return nil, errors.New("eval session registry is nil")
	}
	// A reset removes and closes the old process before installing a new one.
	// Stripe the key rather than allocating an unbounded mutex map: this prevents
	// a concurrent ordinary acquire from slipping into the reset gap while still
	// allowing unrelated conversations to start in parallel.
	gate := &r.gates[evalRegistryStripe(key)]
	gate.Lock()
	defer gate.Unlock()
	var closeBefore []*evalKernel
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, errEvalKernelClosed
	}
	if reset {
		if old := r.sessions[key]; old != nil {
			delete(r.sessions, key)
			if old.entry.timer != nil {
				old.entry.timer.Stop()
			}
			closeBefore = append(closeBefore, old)
		}
	}
	if current := r.sessions[key]; current != nil {
		if current.entry.timer != nil {
			current.entry.timer.Stop()
			current.entry.timer = nil
		}
		current.entry.refs++
		r.mu.Unlock()
		for _, old := range closeBefore {
			old.close()
		}
		return current, nil
	}
	closeBefore = append(closeBefore, r.capacityVictimsLocked()...)
	r.mu.Unlock()
	for _, old := range closeBefore {
		old.close()
	}

	process, err := start()
	if err != nil {
		return nil, err
	}
	kernel := &evalKernel{process: process}
	kernel.entry.refs = 1
	kernel.entry.lastUsed = r.now()

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		kernel.close()
		return nil, errEvalKernelClosed
	}
	// Another caller may have installed the same key while the process was
	// starting. Keep the winner and close this redundant process.
	if winner := r.sessions[key]; winner != nil {
		if winner.entry.timer != nil {
			winner.entry.timer.Stop()
			winner.entry.timer = nil
		}
		winner.entry.refs++
		r.mu.Unlock()
		kernel.close()
		return winner, nil
	}
	r.sessions[key] = kernel
	r.mu.Unlock()
	return kernel, nil
}

func evalRegistryStripe(key string) uint8 {
	var hash uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		hash ^= uint32(key[i])
		hash *= 16777619
	}
	return uint8(hash % 64)
}

func (r *EvalSessionRegistry) capacityVictimsLocked() []*evalKernel {
	if len(r.sessions) < r.capacity {
		return nil
	}
	type candidate struct {
		key  string
		when time.Time
	}
	var candidates []candidate
	for key, kernel := range r.sessions {
		if kernel.entry.refs == 0 {
			candidates = append(candidates, candidate{key: key, when: kernel.entry.lastUsed})
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].when.Before(candidates[j].when) })
	need := len(r.sessions) - r.capacity + 1
	if need > len(candidates) {
		need = len(candidates)
	}
	victims := make([]*evalKernel, 0, need)
	for _, candidate := range candidates[:need] {
		kernel := r.sessions[candidate.key]
		delete(r.sessions, candidate.key)
		if kernel.entry.timer != nil {
			kernel.entry.timer.Stop()
			kernel.entry.timer = nil
		}
		victims = append(victims, kernel)
	}
	return victims
}

func (r *EvalSessionRegistry) release(key string, kernel *evalKernel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sessions[key] != kernel || kernel.entry.refs <= 0 {
		return
	}
	kernel.entry.refs--
	kernel.entry.lastUsed = r.now()
	if kernel.entry.refs != 0 {
		return
	}
	if kernel.entry.timer != nil {
		kernel.entry.timer.Stop()
	}
	kernel.entry.timer = time.AfterFunc(r.idle, func() { r.expire(key, kernel) })
}

func (r *EvalSessionRegistry) expire(key string, kernel *evalKernel) {
	r.mu.Lock()
	if r.sessions[key] != kernel || kernel.entry.refs != 0 || r.now().Sub(kernel.entry.lastUsed) < r.idle {
		r.mu.Unlock()
		return
	}
	delete(r.sessions, key)
	kernel.entry.timer = nil
	r.mu.Unlock()
	kernel.close()
}

func (r *EvalSessionRegistry) invalidate(key string, kernel *evalKernel) {
	r.mu.Lock()
	if r.sessions[key] == kernel {
		delete(r.sessions, key)
		if kernel.entry.timer != nil {
			kernel.entry.timer.Stop()
			kernel.entry.timer = nil
		}
	}
	r.mu.Unlock()
	kernel.close()
}

// Close releases every retained process. It is primarily useful to embedding
// hosts and tests; normal sessions also have idle and hard lifetime bounds.
func (r *EvalSessionRegistry) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	kernels := make([]*evalKernel, 0, len(r.sessions))
	for key, kernel := range r.sessions {
		delete(r.sessions, key)
		if kernel.entry.timer != nil {
			kernel.entry.timer.Stop()
			kernel.entry.timer = nil
		}
		kernels = append(kernels, kernel)
	}
	r.mu.Unlock()
	for _, kernel := range kernels {
		kernel.close()
	}
}

func (k *evalKernel) execute(ctx context.Context, code string, limit, maxBridgeCalls int, invoker agentcore.ToolInvoker) (evalCellResult, error) {
	k.executionMu.Lock()
	defer k.executionMu.Unlock()
	if k.closed || k.process == nil {
		return evalCellResult{}, errEvalKernelClosed
	}
	return k.process.execute(ctx, code, limit, maxBridgeCalls, invoker)
}

func (k *evalKernel) close() {
	k.executionMu.Lock()
	defer k.executionMu.Unlock()
	if k.closed {
		return
	}
	k.closed = true
	if k.process != nil {
		k.process.close()
	}
}
