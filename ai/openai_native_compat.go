package ai

import (
	"encoding/json"
	"strings"
)

// OpenAICompletionsCompat is the resolved Pi contract. Explicit model.compat
// overrides are applied by ResolveOpenAICompletionsCompat; this is independent
// of the legacy provider's smaller Compat table.
type OpenAICompletionsCompat struct {
	SupportsStore                               bool            `json:"supportsStore"`
	SupportsDeveloperRole                       bool            `json:"supportsDeveloperRole"`
	SupportsReasoningEffort                     bool            `json:"supportsReasoningEffort"`
	SupportsUsageInStreaming                    bool            `json:"supportsUsageInStreaming"`
	SupportsFinishReason                        bool            `json:"supportsFinishReason"`
	MaxTokensField                              string          `json:"maxTokensField"`
	RequiresToolResultName                      bool            `json:"requiresToolResultName"`
	RequiresAssistantAfterToolResult            bool            `json:"requiresAssistantAfterToolResult"`
	RequiresThinkingAsText                      bool            `json:"requiresThinkingAsText"`
	RequiresReasoningContentOnAssistantMessages bool            `json:"requiresReasoningContentOnAssistantMessages"`
	ThinkingFormat                              string          `json:"thinkingFormat"`
	OpenRouterRouting                           json.RawMessage `json:"openRouterRouting"`
	VercelGatewayRouting                        json.RawMessage `json:"vercelGatewayRouting"`
	ChatTemplateKwargs                          json.RawMessage `json:"chatTemplateKwargs"`
	ChatTemplateArgs                            json.RawMessage `json:"chatTemplateArgs"`
	ZaiToolStream                               bool            `json:"zaiToolStream"`
	SupportsThinkingTokenBudget                 bool            `json:"supportsThinkingTokenBudget"`
	ThinkingTokenBudgetField                    string          `json:"thinkingTokenBudgetField,omitempty"`
	SupportsStrictMode                          bool            `json:"supportsStrictMode"`
	SupportsOpenAIGrammarTools                  bool            `json:"supportsOpenAIGrammarTools"`
	SupportsMidConvoSystemMessages              bool            `json:"supportsMidConvoSystemMessages"`
	SupportsMidConvoToolAdditions               bool            `json:"supportsMidConvoToolAdditions"`
	CacheControlFormat                          string          `json:"cacheControlFormat,omitempty"`
	SendSessionAffinityHeaders                  bool            `json:"sendSessionAffinityHeaders"`
	SessionAffinityFormat                       string          `json:"sessionAffinityFormat"`
	SupportsLongCacheRetention                  bool            `json:"supportsLongCacheRetention"`
	VLLMPriority                                json.RawMessage `json:"vllmPriority,omitempty"`
}

type completionsModel struct {
	ID, API, Provider, BaseURL    string
	Input                         []string
	Reasoning                     bool
	MaxTokens                     float64
	ContextWindow                 float64
	ThinkingLevelMap              map[string]json.RawMessage
	SamplingParams                map[string]json.RawMessage
	SamplingParamsByThinkingLevel map[string]json.RawMessage
	Compat                        map[string]json.RawMessage
	Cost                          completionsCost
}

func ResolveOpenAICompletionsCompat(raw json.RawMessage) (OpenAICompletionsCompat, error) {
	var model completionsModel
	if err := json.Unmarshal(raw, &model); err != nil {
		return OpenAICompletionsCompat{}, err
	}
	provider, url := model.Provider, model.BaseURL
	has := func(value string) bool { return strings.Contains(url, value) }
	zai := provider == "zai" || provider == "zai-coding-cn" || has("api.z.ai") || has("open.bigmodel.cn")
	together := provider == "together" || has("api.together.ai") || has("api.together.xyz")
	moonshot := provider == "moonshotai" || provider == "moonshotai-cn" || has("api.moonshot.")
	router := provider == "openrouter" || has("openrouter.ai")
	workers := provider == "cloudflare-workers-ai" || has("api.cloudflare.com")
	gateway := provider == "cloudflare-ai-gateway" || has("gateway.ai.cloudflare.com")
	nvidia := provider == "nvidia" || has("integrate.api.nvidia.com")
	ant := provider == "ant-ling" || has("api.ant-ling.com")
	cerebras := provider == "cerebras" || has("cerebras.ai")
	deepseek := provider == "deepseek" || strings.Contains(strings.ToLower(url), "deepseek.com")
	grok := provider == "xai" || has("api.x.ai")
	nonstandard := nvidia || cerebras || grok || together || has("chutes.ai") || deepseek || zai || moonshot || provider == "opencode" || has("opencode.ai") || workers || gateway || ant
	compat := OpenAICompletionsCompat{
		SupportsStore:            !nonstandard,
		SupportsDeveloperRole:    router && (strings.HasPrefix(model.ID, "anthropic/") || strings.HasPrefix(model.ID, "openai/")) || (!nonstandard && !router),
		SupportsReasoningEffort:  !grok && !zai && !moonshot && !together && !gateway && !nvidia && !ant,
		SupportsUsageInStreaming: true, SupportsFinishReason: true, MaxTokensField: "max_completion_tokens",
		RequiresReasoningContentOnAssistantMessages: deepseek, ThinkingFormat: "openai",
		OpenRouterRouting: json.RawMessage(`{}`), VercelGatewayRouting: json.RawMessage(`{}`), ChatTemplateKwargs: json.RawMessage(`{}`), ChatTemplateArgs: json.RawMessage(`{}`),
		SendSessionAffinityHeaders: router, SessionAffinityFormat: "openai", SupportsLongCacheRetention: !(together || workers || gateway || nvidia || ant),
	}
	if has("chutes.ai") || deepseek || moonshot || gateway || together || nvidia || ant || zai {
		compat.MaxTokensField = "max_tokens"
	}
	switch {
	case deepseek:
		compat.ThinkingFormat = "deepseek"
	case zai:
		compat.ThinkingFormat = "zai"
	case together:
		compat.ThinkingFormat = "together"
	case ant:
		compat.ThinkingFormat = "ant-ling"
	case router:
		compat.ThinkingFormat = "openrouter"
	}
	if provider == "openrouter" && strings.HasPrefix(model.ID, "anthropic/") {
		compat.CacheControlFormat = "anthropic"
	}
	if router {
		compat.SessionAffinityFormat = "openrouter"
	}
	// Pi uses nullish fallback for every override except vllmPriority.
	overrides := map[string]json.RawMessage{}
	for key, value := range model.Compat {
		if key == "vllmPriority" || strings.TrimSpace(string(value)) != "null" {
			overrides[key] = value
		}
	}
	encoded, err := json.Marshal(overrides)
	if err != nil {
		return compat, err
	}
	err = json.Unmarshal(encoded, &compat)
	return compat, err
}
