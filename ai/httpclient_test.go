package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// The chat clients must not cap a request like a handshake. A turn that writes a
// whole HTML page legitimately generates for minutes, so an absolute deadline
// sized for the handshake kills the slowest healthy turns — which is exactly
// what took out the collision and idle-game benchmarks at 120s.
func TestNewChatHTTPClient_AbsoluteCapLeavesRoomForLongGenerations(t *testing.T) {
	c := NewChatHTTPClient(0)
	if c.Timeout < 10*time.Minute {
		t.Fatalf("absolute timeout %s is too tight for a full-page generation turn", c.Timeout)
	}
}

// The non-streamed client must carry NO header deadline. A buffered completion's
// headers do not leave the gateway until the model has finished generating, so a
// header deadline there is a generation deadline in disguise: bench #4's first
// turn (9.5k output tokens) died at exactly 60.3s and the retry then paid 135s
// to regenerate the same answer.
func TestNewChatHTTPClient_NoHeaderDeadlineOnTheBufferedPath(t *testing.T) {
	tr, ok := NewChatHTTPClient(0).Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T, want *http.Transport", NewChatHTTPClient(0).Transport)
	}
	if tr.ResponseHeaderTimeout != 0 {
		t.Fatalf("ResponseHeaderTimeout = %s on the non-streamed client; a buffered "+
			"completion's headers arrive only after generation, so this caps generation",
			tr.ResponseHeaderTimeout)
	}
}

// The streamed client keeps the header deadline, where it means what it says:
// an SSE gateway flushes headers before the first token, so silence past the
// deadline is a dead endpoint, not a slow model.
func TestNewStreamHTTPClient_KeepsTheHeaderDeadline(t *testing.T) {
	c := NewStreamHTTPClient(0)
	if c.Timeout < 10*time.Minute {
		t.Fatalf("absolute timeout %s is too tight for a long stream", c.Timeout)
	}
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T, want *http.Transport", c.Transport)
	}
	if tr.ResponseHeaderTimeout <= 0 || tr.ResponseHeaderTimeout >= c.Timeout {
		t.Fatalf("ResponseHeaderTimeout = %s, want a short deadline below the %s cap",
			tr.ResponseHeaderTimeout, c.Timeout)
	}
	// The two clients must not share a transport: tightening one would silently
	// tighten the other.
	if tr == NewChatHTTPClient(0).Transport {
		t.Fatal("stream and chat clients share a transport")
	}
}

// The providers must actually route by path: Chat on the client with no header
// deadline, Stream on the one that has it. Wiring the field and never reading it
// is the failure mode this catches.
func TestProviders_RouteStreamAndChatToTheirOwnClients(t *testing.T) {
	oa := NewOpenAIProvider("k", "", Compat{})
	if oa.HTTP == oa.StreamHTTP {
		t.Fatal("openai: Chat and Stream share one client")
	}
	if got := oa.streamHTTP(); got != oa.StreamHTTP {
		t.Fatal("openai: streamHTTP() does not return StreamHTTP")
	}
	oa.StreamHTTP = nil
	if got := oa.streamHTTP(); got != oa.HTTP {
		t.Fatal("openai: streamHTTP() must fall back to a caller-supplied HTTP")
	}

	an := NewAnthropicProvider("k", "")
	if an.HTTP == an.StreamHTTP {
		t.Fatal("anthropic: Chat and Stream share one client")
	}
	if got := an.streamHTTP(); got != an.StreamHTTP {
		t.Fatal("anthropic: streamHTTP() does not return StreamHTTP")
	}
	an.StreamHTTP = nil
	if got := an.streamHTTP(); got != an.HTTP {
		t.Fatal("anthropic: streamHTTP() must fall back to a caller-supplied HTTP")
	}
}

// A slow body is the normal case for a streaming provider, not an error. This
// drives a real server that trickles its response well past the old 120s-style
// cap ratio to prove the client waits for it.
func TestNewChatHTTPClient_WaitsOutASlowTricklingBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		// Headers land at once; the body then arrives in slow instalments, the
		// shape of a model generating a large answer.
		for i := 0; i < 5; i++ {
			fmt.Fprintf(w, "data: chunk %d\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(40 * time.Millisecond)
		}
	}))
	defer srv.Close()

	// A cap far below total body time would have killed this; the point is that
	// the header deadline does not, because the headers were prompt.
	c := NewStreamHTTPClient(10 * time.Second)
	tr := c.Transport.(*http.Transport)
	tr.ResponseHeaderTimeout = 50 * time.Millisecond

	resp, err := c.Get(srv.URL)
	if err != nil {
		t.Fatalf("slow trickling body was rejected: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

// An unresponsive gateway — one that accepts the connection and then never
// writes headers — must still fail fast rather than sit for the absolute cap.
func TestNewChatHTTPClient_DeadGatewayFailsOnHeaderDeadline(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release // never writes headers until the test is done with it
	}))
	defer func() { close(release); srv.Close() }()

	c := NewStreamHTTPClient(30 * time.Second)
	c.Transport.(*http.Transport).ResponseHeaderTimeout = 100 * time.Millisecond

	start := time.Now()
	_, err := c.Get(srv.URL)
	if err == nil {
		t.Fatal("dead gateway returned a response")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("took %s to notice a dead gateway; header deadline is not doing its job", elapsed)
	}
}

// Go reports a caller cancellation and our own client cap with the identical
// string, and the two have opposite fixes. describeReadErr must tell them apart.
func TestDescribeReadErr_NamesTheDeadlineThatFired(t *testing.T) {
	client := &http.Client{Timeout: 120 * time.Second}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	got := describeReadErr(cancelled, client, os.ErrDeadlineExceeded)
	if !strings.Contains(got, "caller context") {
		t.Fatalf("cancelled caller described as %q, want it named as the caller's", got)
	}

	// Live caller, timed-out read: this is our cap, and the message should say so
	// and quote the value so the operator knows what to raise.
	got = describeReadErr(context.Background(), client, os.ErrDeadlineExceeded)
	if !strings.Contains(got, "client timeout") || !strings.Contains(got, "2m0s") {
		t.Fatalf("client-cap expiry described as %q, want it named with its value", got)
	}

	// A severed connection is neither; it must pass through untouched rather than
	// be mislabelled as a timeout.
	got = describeReadErr(context.Background(), client, errors.New("connection reset by peer"))
	if got != "connection reset by peer" {
		t.Fatalf("non-timeout read error described as %q, want it verbatim", got)
	}
}
