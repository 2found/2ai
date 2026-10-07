package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/2found/2ai/ai/protocol"
)

type SpeechSpec struct{ Vendor, BaseURL, APIKey, Model, Voice, Text string }

// Synthesize buffers bounded audio, so an unsuccessful attempt can safely fall
// back before any audio is exposed. Redirects must not forward provider keys.
func Synthesize(ctx context.Context, s SpeechSpec) ([]byte, string, error) {
	if strings.TrimSpace(s.Text) == "" || len(s.Text) > 12000 {
		return nil, "", errors.New("speech text must be 1..12000 bytes")
	}
	base := strings.TrimRight(s.BaseURL, "/")
	if base == "" {
		base = ProviderDefaultURL(s.Vendor)
	}
	endpoint := base + "/audio/speech"
	payload := map[string]any{"model": s.Model, "voice": s.Voice, "input": s.Text, "response_format": "mp3"}
	switch s.Vendor {
	case "kokoro":
		endpoint = base + "/v1/audio/speech"
		if strings.HasSuffix(base, "/v1") {
			endpoint = base + "/audio/speech"
		}
		if s.Model == "" {
			payload["model"] = "kokoro"
		}
		if s.Voice == "" {
			payload["voice"] = "af_heart"
		}
	case "elevenlabs":
		if s.Voice == "" {
			return nil, "", errors.New("ElevenLabs voice ID is required")
		}
		endpoint = base + "/text-to-speech/" + url.PathEscape(s.Voice)
		payload = map[string]any{"text": s.Text, "model_id": s.Model}
		if s.Model == "" {
			payload["model_id"] = "eleven_multilingual_v2"
		}
	default:
		return nil, "", errors.New("unsupported speech provider")
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(raw))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.APIKey != "" {
		if s.Vendor == "elevenlabs" {
			req.Header.Set("xi-api-key", s.APIKey)
		} else {
			req.Header.Set("Authorization", "Bearer "+s.APIKey)
		}
	}
	client := &http.Client{Timeout: 20 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", errors.New("speech provider connection failed")
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", &protocol.ProviderError{Status: resp.StatusCode, Message: "speech provider rejected the request"}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20+1))
	if err != nil || len(data) == 0 || len(data) > 16<<20 {
		return nil, "", errors.New("invalid or oversized audio response")
	}
	mime := strings.Split(resp.Header.Get("Content-Type"), ";")[0]
	if !strings.HasPrefix(mime, "audio/") {
		return nil, "", errors.New("speech provider returned non-audio content")
	}
	return data, mime, nil
}
