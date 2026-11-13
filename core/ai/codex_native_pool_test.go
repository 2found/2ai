package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func nativePoolToken(account string) OAuthToken {
	claims, _ := json.Marshal(map[string]any{"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": account}})
	return OAuthToken{AccountID: account, AccessToken: "h." + base64.StdEncoding.EncodeToString(claims) + ".s"}
}
func TestNativeCodexPoolRotationAndReporting(t *testing.T) {
	for _, mode := range []string{"rotate", "repeat", "quota", "concurrency", "callback", "success"} {
		t.Run(mode, func(t *testing.T) {
			source := &fakeTokenSource{tokens: []OAuthToken{nativePoolToken("a"), nativePoolToken("b")}}
			if mode == "repeat" {
				source.tokens = source.tokens[:1]
			}
			calls := 0
			identities := []string{}
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				identities = append(identities, r.Header.Get("Chatgpt-Account-Id"))
				status := 200
				body := `data: {"type":"response.done","response":{"id":"r","status":"completed","output":[]}}` + "\n\n"
				if calls == 1 && mode != "success" && mode != "callback" {
					status = 401
					message := "unauthorized"
					if mode == "quota" {
						status = 429
					}
					if mode == "concurrency" {
						status = 403
						message = "concurrency cap"
					}
					body = fmt.Sprintf(`{"error":{"message":%q}}`, message)
				}
				return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: http.Header{"Retry-After": []string{"2"}}, Body: io.NopCloser(strings.NewReader(body))}, nil
			})}
			options := CodexResponsesStreamOptions{Options: json.RawMessage(`{"transport":"sse"}`), Client: client}
			if mode == "callback" {
				options.OnProviderStreamEvent = func(context.Context, *json.RawMessage, json.RawMessage) error {
					return &codexHTTPError{Status: 401, Message: "callback unauthorized"}
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			stream, err := StreamCodexResponsesPooled(ctx, json.RawMessage(`{"id":"test","api":"openai-codex-responses","provider":"openai-codex","input":["text"]}`), NormalizeContext(Context{}), options, source)
			if err != nil {
				t.Fatal(err)
			}
			starts, terminals := 0, 0
			for {
				event, ok, err := stream.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
				if event.Type == "start" {
					starts++
				}
				if event.Type == "done" || event.Type == "error" {
					terminals++
				}
				if _, err = stream.SnapshotEvent(event); err != nil {
					t.Fatal(err)
				}
			}
			result, err := stream.Result(ctx)
			if err != nil {
				t.Fatal(err)
			}
			wantCalls := 1
			if mode == "rotate" {
				wantCalls = 2
			}
			if calls != wantCalls || terminals != 1 {
				t.Fatalf("calls=%d terminals=%d", calls, terminals)
			}
			if mode == "rotate" {
				if source.acquireN != 2 || len(source.reports) != 2 || source.reports[0].tok.AccountID != "a" || !isOAuthAuthFailure(source.reports[0].err) || source.reports[1].tok.AccountID != "b" || source.reports[1].err != nil || strings.Join(identities, ",") != "a,b" || starts != 1 {
					t.Fatalf("wrong rotation/report: %+v starts=%d", source, starts)
				}
			}
			if mode == "concurrency" && len(source.reports) != 0 {
				t.Fatal("concurrency cap reported against account")
			}
			if mode == "success" || mode == "rotate" {
				if result.StopReason != "stop" {
					t.Fatal(result)
				}
			} else {
				if result.StopReason != "error" {
					t.Fatal(result)
				}
			}
		})
	}
}

func TestNativeCodexPoolProgressivePointerIdentity(t *testing.T) {
	source := &fakeTokenSource{tokens: []OAuthToken{nativePoolToken("a")}}
	release := make(chan struct{})
	defer close(release)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		reader, writer := io.Pipe()
		go func() {
			defer writer.Close()
			for _, event := range []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}`,
				`{"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hello"}`,
			} {
				if _, err := fmt.Fprintf(writer, "data: %s\n\n", event); err != nil {
					return
				}
			}
			select {
			case <-release:
			case <-ctx.Done():
				return
			}
			fmt.Fprint(writer, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}}\n\ndata: {\"type\":\"response.done\",\"response\":{\"id\":\"r\",\"status\":\"completed\",\"output\":[]}}\n\n")
		}()
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: reader}, nil
	})}
	stream, err := StreamCodexResponsesPooled(ctx, json.RawMessage(`{"id":"test","api":"openai-codex-responses","provider":"openai-codex","input":["text"]}`), NormalizeContext(Context{}), CodexResponsesStreamOptions{Options: json.RawMessage(`{"transport":"sse"}`), Client: client}, source)
	if err != nil {
		t.Fatal(err)
	}
	var partial *Message
	deltaSeen := false
	for !deltaSeen {
		event, ok, err := stream.Next(ctx)
		if err != nil || !ok {
			t.Fatalf("stream ended before progressive text: %v", err)
		}
		if event.Partial != nil {
			if partial != nil && partial != event.Partial {
				t.Fatal("partial identity changed")
			}
			partial = event.Partial
		}
		if _, err = stream.SnapshotEvent(event); err != nil {
			t.Fatal(err)
		}
		deltaSeen = event.Type == "text_delta" && event.Delta == "hello"
	}
	release <- struct{}{}
	result, err := stream.Result(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result != partial {
		t.Fatal("terminal message replaced live partial pointer")
	}
	if result.StopReason != "stop" || len(source.reports) != 1 || source.reports[0].err != nil {
		t.Fatalf("unexpected settlement: %+v", result)
	}
}
