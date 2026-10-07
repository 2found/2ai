package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/text/encoding/unicode"
)

type codexProtocolError struct {
	Message string
	Payload string
	Cause   error
}

func (e *codexProtocolError) Error() string { return e.Message }
func (e *codexProtocolError) Unwrap() error { return e.Cause }

// codexWebSocketCloseError keeps protocol metadata available to transport retry policy.
type codexWebSocketCloseError struct {
	message  string
	Code     *float64
	Reason   *string
	WasClean *bool
}

func (e *codexWebSocketCloseError) Error() string { return e.message }
func codexSocketEventError(raw json.RawMessage) error {
	var event map[string]json.RawMessage
	_ = json.Unmarshal(raw, &event)
	var message string
	if json.Unmarshal(event["message"], &message) == nil && message != "" {
		return errors.New(message)
	}
	var nested map[string]json.RawMessage
	_ = json.Unmarshal(event["error"], &nested)
	if json.Unmarshal(nested["message"], &message) == nil && message != "" {
		return errors.New(message)
	}
	return errors.New("WebSocket error")
}
func codexSocketCloseError(raw json.RawMessage) error {
	var event map[string]json.RawMessage
	if json.Unmarshal(raw, &event) != nil || event == nil {
		return errors.New("WebSocket closed")
	}
	e := &codexWebSocketCloseError{message: "WebSocket closed"}
	if json.Unmarshal(event["code"], &e.Code) != nil {
		e.Code = nil
	}
	if json.Unmarshal(event["reason"], &e.Reason) != nil {
		e.Reason = nil
	}
	if json.Unmarshal(event["wasClean"], &e.WasClean) != nil {
		e.WasClean = nil
	}
	if e.Code != nil {
		e.message += " " + strconv.FormatFloat(*e.Code, 'f', -1, 64)
	}
	if e.Reason != nil && *e.Reason != "" {
		e.message += " " + *e.Reason
	} else {
		e.Reason = nil
		if e.Code != nil && *e.Code == 1009 {
			e.message += " message too big"
		}
	}
	e.message = strings.TrimFunc(e.message, jsWhitespace)
	return e
}

// One parser owns one response. Transport listeners feed it while run drains the
// queue; it never closes a reusable socket on normal response completion.
type codexWebSocketParser struct {
	mu              sync.Mutex
	queue           []json.RawMessage
	done, completed bool
	failed          error
	wake            chan struct{}
}

func newCodexWebSocketParser() *codexWebSocketParser {
	return &codexWebSocketParser{wake: make(chan struct{}, 1)}
}
func (p *codexWebSocketParser) notify() {
	select {
	case p.wake <- struct{}{}:
	default:
	}
}
func (p *codexWebSocketParser) message(data []byte, binary bool) {
	if binary {
		data, _ = unicode.UTF8BOM.NewDecoder().Bytes(data)
	}
	if len(data) == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !json.Valid(data) || strings.TrimSpace(string(data)) == "null" {
		var value any
		cause := json.Unmarshal(data, &value)
		if cause == nil {
			cause = errors.New("Cannot read properties of null (reading 'type')")
		}
		p.failed = &codexProtocolError{Message: "Invalid Codex WebSocket JSON: " + cause.Error(), Payload: string(data), Cause: cause}
		p.done = true
		p.notify()
		return
	}
	var object map[string]json.RawMessage
	_ = json.Unmarshal(data, &object)
	var kind string
	_ = json.Unmarshal(object["type"], &kind)
	if kind == "response.completed" || kind == "response.done" || kind == "response.incomplete" {
		p.completed = true
		p.done = true
	}
	p.queue = append(p.queue, append(json.RawMessage(nil), data...))
	p.notify()
}
func (p *codexWebSocketParser) fail(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.failed = err
	p.done = true
	p.notify()
}
func (p *codexWebSocketParser) closed(event json.RawMessage) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.completed && p.failed == nil {
		p.failed = codexSocketCloseError(event)
	}
	p.done = true
	p.notify()
}
func (p *codexWebSocketParser) run(ctx context.Context, idle time.Duration, closeSocket func(), consume func(json.RawMessage) (bool, error)) error {
	for {
		if ctx.Err() != nil {
			return errors.New("Request was aborted")
		}
		p.mu.Lock()
		if len(p.queue) > 0 {
			event := p.queue[0]
			p.queue[0] = nil
			p.queue = p.queue[1:]
			p.mu.Unlock()
			stop, err := consume(event)
			if err != nil || stop {
				return err
			}
			continue
		}
		done, completed, failed := p.done, p.completed, p.failed
		// Drop notifications for events already drained before arming the idle timer.
		select {
		case <-p.wake:
		default:
		}
		p.mu.Unlock()
		if done {
			if failed != nil {
				return failed
			}
			if !completed {
				return errors.New("WebSocket stream closed before response.completed")
			}
			return nil
		}
		var timer *time.Timer
		var timeout <-chan time.Time
		if idle > 0 {
			timer = time.NewTimer(idle)
			timeout = timer.C
		}
		select {
		case <-ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return errors.New("Request was aborted")
		case <-p.wake:
			if timer != nil {
				timer.Stop()
			}
		case <-timeout:
			err := fmt.Errorf("WebSocket idle timeout after %sms", strconv.FormatFloat(float64(idle)/float64(time.Millisecond), 'f', -1, 64))
			p.fail(err)
			if closeSocket != nil {
				closeSocket()
			}
			return err
		}
	}
}
