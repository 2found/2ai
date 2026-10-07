package ai

// ProviderOption describes supported host setup; credentials are never metadata.
type ProviderOption struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Auth    string `json:"auth"`
	Purpose string `json:"purpose"`
	BaseURL string `json:"base_url,omitempty"`
}

func ProviderOptions() []ProviderOption {
	return []ProviderOption{
		{VendorOpenAICodex, "Codex OAuth", "oauth", "llm", ""},
		{VendorClaudeCode, "Claude Code OAuth", "oauth", "llm", ""},
		{VendorGoogleAntigravity, "Antigravity OAuth", "oauth", "llm", ""},
		{VendorDevin, "Devin OAuth", "oauth", "llm", ""},
		{VendorXaiOAuth, "Grok OAuth", "oauth", "llm", "https://api.x.ai/v1"},
		{"deepseek", "DeepSeek API", "api_key", "llm", "https://api.deepseek.com/v1"},
		{"anthropic", "Claude API", "api_key", "llm", "https://api.anthropic.com"},
		{"openai", "OpenAI API", "api_key", "llm", "https://api.openai.com/v1"},
		{"google", "Gemini API", "api_key", "llm", "https://generativelanguage.googleapis.com/v1beta/openai"},
		{"opencode-zen", "OpenCode Zen", "api_key", "llm", "https://opencode.ai/zen/v1"},
		{"opencode-go", "OpenCode Go", "api_key", "llm", "https://opencode.ai/zen/go/v1"},
		{"kokoro", "Kokoro TTS", "optional_key", "tts", "http://127.0.0.1:8880"},
		{"elevenlabs", "ElevenLabs", "api_key", "tts", "https://api.elevenlabs.io/v1"},
	}
}

func ProviderDefaultURL(id string) string {
	if id == "opencode" {
		id = "opencode-zen"
	}
	for _, p := range ProviderOptions() {
		if p.ID == id {
			return p.BaseURL
		}
	}
	return ""
}

func ProviderPurpose(id string) string {
	for _, p := range ProviderOptions() {
		if p.ID == id {
			return p.Purpose
		}
	}
	return "llm"
}
