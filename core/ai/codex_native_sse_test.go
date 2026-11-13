package ai

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

type codexChunkReader struct {
	io.Reader
	size int
}

func (r codexChunkReader) Read(p []byte) (int, error) {
	if r.size > 0 && len(p) > r.size {
		p = p[:r.size]
	}
	return r.Reader.Read(p)
}

func TestPiCodexSSEOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-sse.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name      string
				Wire      *string
				ChunkSize int
			}
			Expected json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 17 {
		t.Fatal("unexpected SSE oracle revision/coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			var body io.ReadCloser
			if tc.Input.Wire != nil {
				body = io.NopCloser(codexChunkReader{strings.NewReader(*tc.Input.Wire), tc.Input.ChunkSize})
			}
			events := []json.RawMessage{}
			err := readCodexSSE(context.Background(), body, func(raw json.RawMessage) (bool, error) {
				events = append(events, append(json.RawMessage(nil), raw...))
				return false, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			assertPiJSON(t, tc.Expected, events)
		})
	}
}

func TestCodexSSETerminalAndFailure(t *testing.T) {
	body := io.NopCloser(strings.NewReader("data: {\"type\":\"response.done\"}\n\ndata: invalid\n\n"))
	count := 0
	if err := readCodexSSE(context.Background(), body, func(json.RawMessage) (bool, error) { count++; return true, nil }); err != nil || count != 1 {
		t.Fatalf("terminal did not stop admission: %d %v", count, err)
	}
	body = io.NopCloser(strings.NewReader("data: invalid"))
	if err := readCodexSSE(context.Background(), body, func(json.RawMessage) (bool, error) { t.Fatal("invalid JSON admitted"); return false, nil }); err == nil || !strings.HasPrefix(err.Error(), "Invalid Codex SSE JSON:") {
		t.Fatalf("malformed JSON: %v", err)
	}
}

type codexBlockingReader struct {
	io.ReadCloser
	entered chan struct{}
	once    sync.Once
}

func (r *codexBlockingReader) Read(p []byte) (int, error) {
	r.once.Do(func() { close(r.entered) })
	return r.ReadCloser.Read(p)
}

func TestCodexSSECancellationClosesBlockedRead(t *testing.T) {
	pipe, writer := io.Pipe()
	reader := &codexBlockingReader{ReadCloser: pipe, entered: make(chan struct{})}
	defer writer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- readCodexSSE(ctx, reader, func(json.RawMessage) (bool, error) { return false, nil }) }()
	select {
	case <-reader.entered:
	case <-time.After(time.Second):
		t.Fatal("reader never started")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil || err.Error() != "Request was aborted" {
			t.Fatalf("abort: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancel left reader blocked")
	}
}
