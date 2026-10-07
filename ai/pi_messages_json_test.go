package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/2found/2ai/jsonjs"
)

// Replay the unchanged-source protocol fixtures through the serialized host
// boundary, not only the internal ordered-object converter.
func TestPiMessagesJSONProtocol(t *testing.T) {
	f := readPiMessagesProtocolFixture(t)
	for i, tc := range f.Cases {
		t.Run(strconv.Itoa(i)+"-"+tc.Input.Name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			now := int64(1000000)
			data := tc.Input.Bytes
			if data == nil {
				data = []byte(string(jsonjs.StringCodePoints(tc.Input.Wire)))
			}
			rawEvents := NewArray()
			options := OpenAICompletionsStreamOptions{Options: json.RawMessage(`{"apiKey":"key"}`), Now: func() int64 { return now }, Client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: io.NopCloser(&piMessagesFixtureReader{data: data, size: tc.Input.ChunkSize, failure: tc.Input.ReadError})}, nil
			})}}
			options.OnProviderStreamEvent = func(_ context.Context, raw *json.RawMessage, _ json.RawMessage) error {
				if tc.Input.Action == "advance-clock" {
					now += 10
				}
				event, err := jsonjs.DecodeValue(*raw)
				if err != nil {
					return err
				}
				if tc.Input.Action == "mutate" && catalogProperty(event, "type") == "text_delta" {
					event.(*Object).Set("delta", "changed")
					encoded, err := jsonjs.MarshalValue(event)
					if err != nil {
						return err
					}
					*raw = encoded
				}
				rawEvents.Append(event)
				if tc.Input.Action == "callback-abort" {
					cancel()
				}
				if tc.Input.Action == "callback-error" || tc.Input.Action == "callback-abort" {
					return errors.New("callback failed")
				}
				return nil
			}
			stream, err := StreamPiMessagesJSON(ctx, f.Model, NormalizeContext(Context{}), options)
			if err != nil {
				t.Fatal(err)
			}
			wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
			defer stop()
			if err = stream.WaitForEnd(wait); err != nil {
				t.Fatal(err)
			}
			retained := []AssistantMessageEvent{}
			for {
				event, ok, err := stream.Next(wait)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				retained = append(retained, event)
			}
			result, err := stream.Result(wait)
			if err != nil {
				t.Fatal(err)
			}
			catalogCompare(t, rawEvents, tc.RawEvents)
			catalogCompare(t, piMessagesTestValue(t, retained), tc.Retained)
			catalogCompare(t, piMessagesTestValue(t, result), tc.Result)
		})
	}
}
