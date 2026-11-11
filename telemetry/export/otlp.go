package export

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"time"

	"github.com/2found/2ai/telemetry"
)

// OTLP returns a Sink that posts batches as OTLP/HTTP JSON to endpoint (for
// example http://collector:4318/v1/traces) with the given headers. Heartbeat
// records are skipped: an OTLP span is written once, when it ends. Span ids
// are the recorder's per-root ids, so they are unique within a trace.
func OTLP(endpoint, service string, headers map[string]string, client *http.Client) func(context.Context, []Record) error {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	return func(ctx context.Context, records []Record) error {
		spans := make([]otlpSpan, 0, len(records))
		for _, record := range records {
			if !record.Active {
				spans = append(spans, newOTLPSpan(record))
			}
		}
		if len(spans) == 0 {
			return nil
		}
		body, err := json.Marshal(map[string]any{"resourceSpans": []any{map[string]any{
			"resource":   map[string]any{"attributes": []otlpKeyValue{{Key: "service.name", Value: otlpValue(service)}}},
			"scopeSpans": []any{map[string]any{"scope": map[string]string{"name": "github.com/2found/2ai"}, "spans": spans}},
		}}})
		if err != nil {
			return err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return err
		}
		request.Header.Set("Content-Type", "application/json")
		for name, value := range headers {
			request.Header.Set(name, value)
		}
		response, err := client.Do(request)
		if err != nil {
			return err
		}
		defer response.Body.Close()
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 64<<10))
		if response.StatusCode/100 != 2 {
			return fmt.Errorf("otlp export: %s", response.Status)
		}
		return nil
	}
}

type otlpKeyValue struct {
	Key   string         `json:"key"`
	Value map[string]any `json:"value"`
}

type otlpEvent struct {
	Time       string         `json:"timeUnixNano"`
	Name       string         `json:"name"`
	Attributes []otlpKeyValue `json:"attributes,omitempty"`
}

type otlpSpan struct {
	TraceID    string         `json:"traceId"`
	SpanID     string         `json:"spanId"`
	ParentID   string         `json:"parentSpanId,omitempty"`
	Name       string         `json:"name"`
	Kind       int            `json:"kind"`
	Start      string         `json:"startTimeUnixNano"`
	End        string         `json:"endTimeUnixNano"`
	Attributes []otlpKeyValue `json:"attributes,omitempty"`
	Events     []otlpEvent    `json:"events,omitempty"`
	Status     map[string]any `json:"status"`
}

func newOTLPSpan(record Record) otlpSpan {
	end := record.Timing.StartedAt.Add(record.Timing.Duration)
	span := otlpSpan{TraceID: record.TraceID, SpanID: otlpSpanID(record.Span.ID), Name: record.Span.Name, Kind: 1,
		Start: unixNano(record.Timing.StartedAt), End: unixNano(end), Attributes: otlpAttributes(record.Span.Attributes),
		Status: map[string]any{"code": 1}}
	if record.Span.ParentID != nil {
		span.ParentID = otlpSpanID(*record.Span.ParentID)
	}
	if record.Span.Status.Status == "error" {
		span.Status = map[string]any{"code": 2}
		if record.Span.Status.Error != nil {
			span.Status["message"] = record.Span.Status.Error.Message
		}
	}
	for i, event := range record.Span.Events {
		at := end
		if i < len(record.Timing.EventTimes) {
			at = record.Timing.EventTimes[i]
		}
		span.Events = append(span.Events, otlpEvent{Time: unixNano(at), Name: event.Name, Attributes: otlpAttributes(event.Attributes)})
	}
	return span
}

func otlpSpanID(id int) string {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], uint64(id))
	return hex.EncodeToString(b[:])
}

func unixNano(at time.Time) string { return strconv.FormatInt(at.UnixNano(), 10) }

func otlpAttributes(attributes telemetry.Attributes) []otlpKeyValue {
	var out []otlpKeyValue
	for _, property := range attributes.Entries() {
		if value := otlpValue(property.Value); value != nil {
			out = append(out, otlpKeyValue{Key: property.Name, Value: value})
		}
	}
	return out
}

// otlpValue maps an attribute to an OTLP AnyValue; functions and other values
// JSON cannot carry are skipped.
func otlpValue(value any) map[string]any {
	switch v := value.(type) {
	case string:
		return map[string]any{"stringValue": v}
	case bool:
		return map[string]any{"boolValue": v}
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return map[string]any{"intValue": fmt.Sprint(v)}
	case float32, float64:
		return map[string]any{"doubleValue": v}
	case *telemetry.Array:
		return otlpArray(v.Values())
	}
	if reflected := reflect.ValueOf(value); reflected.Kind() == reflect.Slice {
		values := make([]any, reflected.Len())
		for i := range values {
			values[i] = reflected.Index(i).Interface()
		}
		return otlpArray(values)
	}
	if raw, err := json.Marshal(value); err == nil && string(raw) != "null" {
		return map[string]any{"stringValue": string(raw)}
	}
	return nil
}

func otlpArray(values []any) map[string]any {
	items := []map[string]any{}
	for _, value := range values {
		if item := otlpValue(value); item != nil {
			items = append(items, item)
		}
	}
	return map[string]any{"arrayValue": map[string]any{"values": items}}
}
