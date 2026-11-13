package engine

import (
	"errors"
	"sync"
)

var defaultStream struct {
	sync.RWMutex
	fn StreamFn
}

// SetDefaultStreamFn supplies Pi's process-wide fallback. Passing nil clears it.
// Per-run stream arguments take precedence and avoid global configuration.
func SetDefaultStreamFn(stream StreamFn) {
	defaultStream.Lock()
	defaultStream.fn = stream
	defaultStream.Unlock()
}

func GetDefaultStreamFn() (StreamFn, error) {
	defaultStream.RLock()
	stream := defaultStream.fn
	defaultStream.RUnlock()
	if stream == nil {
		return nil, errors.New("No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn().")
	}
	return stream, nil
}
