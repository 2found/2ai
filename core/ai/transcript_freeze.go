package ai

// freezeTranscript detaches a transcript before a provider's asynchronous
// producer reads it, so a host that mutates or appends to its messages after
// Stream returns cannot change the request. It copies the typed graph
// (streamSnapshot) instead of a JSON round trip: the round trip encoded and
// decoded the whole transcript on every model call, which grows with each turn
// of a run, and the provider then encodes it again to build its request body.
//
// Wire output is unchanged: the copy keeps each record's transcript encoding
// metadata, which a decode would have recaptured from the same bytes. The two
// fields JSON never carries are reset as a decode would leave them: the
// reminder mark (CollapseSystemMessages sets it again on the frozen copy) and a
// usage observation.
func freezeTranscript(transcript TranscriptContext) TranscriptContext {
	source := transcript.Messages()
	if source == nil {
		return TranscriptContext{messages: []Message{}}
	}
	copy := streamSnapshot{messages: map[*Message]*Message{}, blocks: map[*ContentBlock]*ContentBlock{}, usages: map[*Usage]*Usage{}}
	messages := make([]Message, len(source))
	for i := range source {
		message := copy.message(&source[i])
		message.reminder = false
		if message.Usage != nil {
			message.Usage.Observation = UsageObservation{}
		}
		for j, tool := range message.ToolsRemoved {
			message.ToolsRemoved[j].Extra = snapshotJSONFields(tool.Extra)
		}
		messages[i] = *message
	}
	return TranscriptContext{messages: messages}
}
