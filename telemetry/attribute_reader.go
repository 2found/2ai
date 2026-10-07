package telemetry

// SetAttributesFrom defers reading an incoming attribute set until recording
// is active. Like Pi's property reads, a panic discards the outer update while
// retaining any reentrant mutations the reader already made. A successful read
// merges into the attributes captured before reading, so reentrant attribute
// updates are overwritten by that outer merge. Events and status are unaffected.
// Readers run without the recorder lock; concurrent settlement wins.
func (s *Span) SetAttributesFrom(read func() Attributes) {
	if s == nil {
		return
	}
	if s.callbacks.SetAttributesFrom != nil {
		s.callbacks.SetAttributesFrom(read)
		return
	}
	if s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	if s.context.parent.Settled {
		s.context.state.mu.Unlock()
		return
	}
	base, ok := copyAttributes(s.context.parent.Attributes)
	s.context.state.mu.Unlock()
	if !ok {
		return
	}
	incoming, ok := readAttributes(read)
	if !ok {
		return
	}
	copied, ok := copyAttributes(incoming)
	if !ok {
		return
	}
	for _, property := range copied.Entries() {
		base.Set(property.Name, property.Value)
	}
	s.context.state.mu.Lock()
	defer s.context.state.mu.Unlock()
	if !s.context.parent.Settled {
		s.context.parent.Attributes = base
	}
}

// AddEventFrom reads attributes only while recording is active. A read panic
// suppresses the outer event; nested events already emitted by the reader stay
// in order. No recorder lock is held while invoking the reader. An event is
// ignored if its span settles before the reader returns.
func (s *Span) AddEventFrom(name string, read func() Attributes) {
	if s == nil {
		return
	}
	if s.callbacks.AddEventFrom != nil {
		s.callbacks.AddEventFrom(name, read)
		return
	}
	if s.context.state == nil || s.context.parent == nil {
		return
	}
	s.context.state.mu.Lock()
	settled := s.context.parent.Settled
	s.context.state.mu.Unlock()
	if settled {
		return
	}
	if attributes, ok := readAttributes(read); ok {
		s.AddEvent(name, attributes)
	}
}

func readAttributes(read func() Attributes) (attributes Attributes, ok bool) {
	defer func() { _ = recover() }()
	return read(), true
}
