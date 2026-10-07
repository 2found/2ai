package ai

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"

	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

// AzureResponsesConfig resolves Pi's Azure endpoint and deployment settings.
// It performs no HTTP requests or credential lookup.
type AzureResponsesConfig struct {
	BaseURL        string `json:"baseUrl"`
	APIVersion     string `json:"apiVersion"`
	DeploymentName string `json:"deploymentName"`
}

func azureEnv(options map[string]json.RawMessage, name string) string {
	env, _ := samplingObject(options["env"])
	if value := samplingString(env[name]); value != "" {
		return value
	}
	return os.Getenv(name)
}

func azureDeployment(model completionsModel, options map[string]json.RawMessage) string {
	if value := samplingString(options["azureDeploymentName"]); value != "" {
		return value
	}
	mapping := map[string]string{}
	for _, entry := range strings.Split(azureEnv(options, "AZURE_OPENAI_DEPLOYMENT_NAME_MAP"), ",") {
		entry = strings.TrimFunc(entry, jsWhitespace)
		parts := strings.SplitN(entry, "=", 3)
		if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
			continue
		}
		mapping[strings.TrimFunc(parts[0], jsWhitespace)] = strings.TrimFunc(parts[1], jsWhitespace)
	}
	if value := mapping[model.ID]; value != "" {
		return value
	}
	return model.ID
}

func ResolveAzureResponsesConfig(rawModel, rawOptions json.RawMessage) (AzureResponsesConfig, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return AzureResponsesConfig{}, err
	}
	options := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &options); err != nil {
			return AzureResponsesConfig{}, err
		}
	}
	version := samplingString(options["azureApiVersion"])
	if version == "" {
		version = azureEnv(options, "AZURE_OPENAI_API_VERSION")
	}
	if version == "" {
		version = "v1"
	}
	base := strings.TrimFunc(samplingString(options["azureBaseUrl"]), jsWhitespace)
	if base == "" {
		base = strings.TrimFunc(azureEnv(options, "AZURE_OPENAI_BASE_URL"), jsWhitespace)
	}
	resource := samplingString(options["azureResourceName"])
	if resource == "" {
		resource = azureEnv(options, "AZURE_OPENAI_RESOURCE_NAME")
	}
	if base == "" && resource != "" {
		base = "https://" + resource + ".openai.azure.com/openai/v1"
	}
	if base == "" {
		base = model.BaseURL
	}
	if base == "" {
		return AzureResponsesConfig{}, errors.New("Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.")
	}
	url, err := whatwg.Parse(strings.TrimRight(strings.TrimFunc(base, jsWhitespace), "/"))
	if err != nil {
		return AzureResponsesConfig{}, fmt.Errorf("Invalid Azure OpenAI base URL: %s", base)
	}
	host, path := url.Hostname(), strings.TrimRight(url.Pathname(), "/")
	azure := strings.HasSuffix(host, ".openai.azure.com") || strings.HasSuffix(host, ".cognitiveservices.azure.com") || strings.HasSuffix(host, ".ai.azure.com")
	if azure && (path == "" || path == "/openai" || path == "/openai/v1/responses") {
		url.SetPathname("/openai/v1")
		url.SetSearch("")
	}
	return AzureResponsesConfig{BaseURL: strings.TrimRight(url.String(), "/"), APIVersion: version, DeploymentName: azureDeployment(model, options)}, nil
}

func azureResponsesCompat(model completionsModel) (OpenAIResponsesCompat, error) {
	compat := OpenAIResponsesCompat{SupportsDeveloperRole: true, SupportsStrictMode: true}
	overrides := map[string]json.RawMessage{}
	for key, value := range model.Compat {
		if samplingNonNull(value) {
			overrides[key] = value
		}
	}
	raw, err := json.Marshal(overrides)
	if err == nil {
		err = json.Unmarshal(raw, &compat)
	}
	return compat, err
}

// BuildAzureResponsesParams ports the Azure request builder, including its
// distinct strict-tool defaults, deployment name and prompt-cache behavior.
// The input is a normalized transcript; no SDK or TypeScript code is involved.
func BuildAzureResponsesParams(rawModel json.RawMessage, transcript TranscriptContext, rawOptions json.RawMessage) (json.RawMessage, error) {
	var model completionsModel
	if err := json.Unmarshal(rawModel, &model); err != nil {
		return nil, err
	}
	options := map[string]json.RawMessage{}
	if len(rawOptions) > 0 {
		if err := json.Unmarshal(rawOptions, &options); err != nil {
			return nil, err
		}
	}
	compat, err := azureResponsesCompat(model)
	if err != nil {
		return nil, err
	}
	grammar, err := CreateGrammarToolInputProperties(GetDeclaredTools(transcript.Messages()), compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return nil, err
	}
	toolOptions := ResponsesToolsOptions{SupportsStrictMode: &compat.SupportsStrictMode, SupportsOpenAIGrammarTools: compat.SupportsOpenAIGrammarTools}
	messages, err := ConvertResponsesMessages(rawModel, transcript, []string{"openai", "openai-codex", "opencode", "azure-openai-responses"}, ResponsesMessagesOptions{
		GrammarToolInputProperties: grammar, SupportsMidConvoSystemMessages: compat.SupportsMidConvoSystemMessages,
		SupportsAdditionalTools: compat.SupportsAdditionalTools, SupportsToolSearch: compat.SupportsToolSearch, ToolOptions: toolOptions,
	})
	if err != nil {
		return nil, err
	}
	params := map[string]json.RawMessage{"input": messages, "stream": json.RawMessage(`true`), "store": json.RawMessage(`false`)}
	set := func(key string, value any) { params[key], _ = json.Marshal(value) }
	set("model", azureDeployment(model, options))
	if raw, exists := options["sessionId"]; exists {
		params["prompt_cache_key"], err = ClampOpenAIPromptCacheKey(raw)
		if err != nil {
			return nil, err
		}
	}
	if samplingTruthy(options["maxTokens"]) {
		var tokens float64
		if err := json.Unmarshal(options["maxTokens"], &tokens); err != nil {
			return nil, err
		}
		set("max_output_tokens", math.Max(tokens, 16))
	}
	for _, field := range []struct{ option, param string }{{"temperature", "temperature"}, {"toolChoice", "tool_choice"}} {
		if value, exists := options[field.option]; exists {
			params[field.param] = value
		}
	}
	state := ResolveTranscriptTools(transcript.Messages(), compat.SupportsAdditionalTools || compat.SupportsToolSearch)
	if len(state.RequestTools) > 0 {
		params["tools"], err = ConvertResponsesTools(state.RequestTools, toolOptions)
		if err != nil {
			return nil, err
		}
	}
	reasoningEffort := options["reasoningEffort"]
	if !samplingNonNull(reasoningEffort) && samplingTruthy(options["reasoningSummary"]) {
		reasoningEffort = json.RawMessage(`"medium"`)
	}
	if model.Reasoning {
		if samplingTruthy(reasoningEffort) {
			effort := reasoningEffort
			if samplingTruthy(options["reasoningEffort"]) {
				effort = options["reasoningEffort"]
				if mapped := model.ThinkingLevelMap[samplingString(effort)]; samplingNonNull(mapped) {
					effort = mapped
				}
			}
			summary := options["reasoningSummary"]
			if !samplingTruthy(summary) {
				summary = json.RawMessage(`"auto"`)
			}
			set("reasoning", map[string]json.RawMessage{"effort": effort, "summary": summary})
			set("include", []string{"reasoning.encrypted_content"})
		} else if strings.TrimSpace(string(model.ThinkingLevelMap["off"])) != "null" {
			effort := model.ThinkingLevelMap["off"]
			if !samplingNonNull(effort) {
				effort = json.RawMessage(`"none"`)
			}
			set("reasoning", map[string]json.RawMessage{"effort": effort})
		}
	}
	for key, value := range resolveSamplingParams(model, samplingString(reasoningEffort), options["samplingParams"]) {
		params[key] = value
	}
	return json.Marshal(params)
}
