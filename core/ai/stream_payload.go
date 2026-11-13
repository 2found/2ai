package ai

import (
	"errors"
	"sync"
)

// PayloadYield releases payload access while work waits, then reacquires it
// before returning (or propagating a panic). It is the explicit Go counterpart
// of yielding a JavaScript callback at await. Do not access live payload fields
// inside work; use them before yielding or after it returns.
type PayloadYield func(work func() error) error

// SynchronizeYielding protects a callback's live message access with the same
// lock as Synchronize. Unlike Synchronize, its callback can wait for a provider
// using yield without preventing that provider from updating the payload.
// Use yield synchronously in the callback. Nested/overlapping yields and calls
// retained after the callback returns are rejected without invoking their work.
func (s *AssistantMessageEventStream) SynchronizeYielding(callback func(PayloadYield) error) error {
	s.payloadMu.Lock()
	var state sync.Mutex
	active := true
	var waiting chan struct{}
	defer func() {
		state.Lock()
		active = false
		pending := waiting
		state.Unlock()
		// Also keep lock ownership balanced if a misbehaving caller launches
		// yield in a goroutine and returns before that yield has finished.
		if pending != nil {
			<-pending
		}
		s.payloadMu.Unlock()
	}()
	return callback(func(work func() error) error {
		state.Lock()
		if !active || waiting != nil {
			state.Unlock()
			return errors.New("ai: payload yield requires an active, non-yielding synchronization callback")
		}
		done := make(chan struct{})
		waiting = done
		state.Unlock()
		s.payloadMu.Unlock()
		defer func() {
			s.payloadMu.Lock()
			state.Lock()
			waiting = nil
			close(done)
			state.Unlock()
		}()
		return work()
	})
}
