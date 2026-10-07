package sandbox

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebFetchStripsHTMLToText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(`<html><head><style>.x{}</style></head><body><h1>Title</h1><p>Hello <b>world</b></p><script>alert(1)</script></body></html>`))
	}))
	defer srv.Close()

	tool := NewWebFetchTool(nil, nil)
	tool.AllowAllIPsForTest()
	out, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`"}`)
	if err != nil {
		t.Fatalf("web_fetch Run: %v", err)
	}
	if !strings.Contains(out, "Title") || !strings.Contains(out, "Hello") || !strings.Contains(out, "world") {
		t.Fatalf("missing text content: %q", out)
	}
	if strings.Contains(out, "alert(1)") || strings.Contains(out, ".x{}") {
		t.Fatalf("script/style leaked into output: %q", out)
	}
}

func TestWebFetchRefusesLoopbackSSRF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("secret"))
	}))
	defer srv.Close()

	// No AllowAllIPsForTest: the guarded dialer must refuse loopback.
	tool := NewWebFetchTool(nil, nil)
	if _, err := tool.Run(context.Background(), `{"url":"`+srv.URL+`"}`); err == nil {
		t.Fatal("expected loopback fetch to be refused")
	}
}

func TestWebFetchRejectsRelativeURL(t *testing.T) {
	tool := NewWebFetchTool(nil, nil)
	if _, err := tool.Run(context.Background(), `{"url":"/just/a/path"}`); err == nil {
		t.Fatal("expected relative url to be rejected")
	}
}

func TestHTMLDetection(t *testing.T) {
	if !isHTML("text/html; charset=utf-8", nil) {
		t.Fatal("content-type html not detected")
	}
	if !isHTML("", []byte("<!DOCTYPE html><html>")) {
		t.Fatal("doctype sniff failed")
	}
	if isHTML("application/json", []byte(`{"a":1}`)) {
		t.Fatal("json wrongly detected as html")
	}
}
