package ai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/2found/2ai/jsonjs"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

const DefaultRadiusGateway = "https://radius.pi.dev"

var radiusHTTPScheme = regexp.MustCompile(`(?i)^https?://`)

// NormalizeRadiusGatewayURL retains the source text, adding a scheme only
// when no HTTP(S) prefix is present and removing trailing slashes.
func NormalizeRadiusGatewayURL(value string) string {
	if !radiusHTTPScheme.MatchString(value) {
		value = "https://" + value
	}
	return strings.TrimRight(value, "/")
}

func isRadiusGatewayModel(value any) bool {
	model, ok := value.(*Object)
	if !ok || model == nil {
		return false
	}
	_, id := model.Get("id").(string)
	_, name := model.Get("name").(string)
	_, reasoning := model.Get("reasoning").(bool)
	input, inputOK := model.Get("input").(*Array)
	cost, costOK := model.Get("cost").(*Object)
	_, window := deviceCodeNumber(model.Get("contextWindow"))
	_, max := deviceCodeNumber(model.Get("maxTokens"))
	return id && name && reasoning && inputOK && input != nil && costOK && cost != nil && window && max
}

// Sanitization drops invalid rows and shallow-copies valid rows. Nested input,
// cost and extension values retain their identity, matching the source spread.
func sanitizeRadiusGatewayConfig(value any) *Object {
	config, ok := value.(*Object)
	if !ok || config == nil {
		return nil
	}
	base, ok := config.Get("baseUrl").(string)
	if !ok {
		return nil
	}
	models, ok := config.Get("models").(*Array)
	if !ok || models == nil {
		return nil
	}
	valid := NewArray()
	for _, index := range models.Keys() {
		model := models.Get(index)
		if isRadiusGatewayModel(model) {
			valid.Append(authSpread(model))
		}
	}
	return NewObject(Property{Name: "baseUrl", Value: base}, Property{Name: "models", Value: valid})
}

// A nil result corresponds to the source's undefined configuration.
func GetRadiusCredentialConfig(credential any) *Object {
	return sanitizeRadiusGatewayConfig(catalogProperty(credential, "gatewayConfig"))
}

func GetRadiusModelsFromConfig(providerID string, config *Object) (*Array, error) {
	if config == nil {
		return nil, jsonjs.PropertyReadError(true, "config.models")
	}
	value := catalogProperty(config, "models")
	models, ok := value.(*Array)
	if !ok {
		if jsonjs.IsNullish(value) {
			return nil, jsonjs.PropertyReadError(!jsonjs.IsUndefined(value), "config.models.map")
		}
		return nil, errors.New("config.models.map is not a function. (In 'config.models.map', 'config.models.map' is undefined)")
	}
	result := NewArray()
	result.SetLength(models.Len())
	for _, index := range models.Keys() {
		model := authSpread(models.Get(index))
		model.Set("api", "pi-messages")
		model.Set("provider", providerID)
		model.Set("baseUrl", catalogProperty(config, "baseUrl"))
		result.Set(index, model)
	}
	return result, nil
}

func GetRadiusModels(providerID string, credential any) *Array {
	config := GetRadiusCredentialConfig(credential)
	if config == nil {
		return NewArray()
	}
	models, _ := GetRadiusModelsFromConfig(providerID, config)
	return models
}

func truncateRadiusHTTPBody(body string) string {
	trimmed := strings.TrimFunc(body, jsWhitespace)
	points := jsonjs.StringCodePoints(trimmed)
	var result strings.Builder
	units := 0
	for index, point := range points {
		if units == 512 {
			return result.String() + "…"
		}
		if point > 0xffff {
			if units == 511 {
				return result.String() + catalogCodeUnit(0xd800+(point-0x10000)>>10) + "…"
			}
			result.WriteRune(point)
			units += 2
		} else {
			result.WriteString(catalogCodeUnit(point))
			units++
		}
		if index == len(points)-1 {
			return trimmed
		}
	}
	return trimmed
}

func LoadRadiusGatewayConfig(ctx context.Context, gateway string, apiKey any, clients ...*http.Client) (*Object, error) {
	client := http.DefaultClient
	if len(clients) > 0 && clients[0] != nil {
		client = clients[0]
	}
	headers := http.Header{"Accept": {"application/json"}}
	if catalogEntryTruthy(apiKey) {
		key, err := catalogKey(apiKey, make(map[*Array]bool))
		if err != nil {
			return nil, err
		}
		headers.Set("Authorization", "Bearer "+key)
	}
	endpoint, err := radiusEndpoint(gateway, "/v1/config")
	if err != nil {
		return nil, err
	}
	response, err := oauthFetch(ctx, client, endpoint, "GET", headers, "")
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, err := oauthRead(response)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("Could not load Radius config from %s: %d: %s", gateway, response.StatusCode, truncateRadiusHTTPBody(string(body)))
	}
	value, err := oauthReadJSON(response)
	if err != nil {
		return nil, err
	}
	config := sanitizeRadiusGatewayConfig(value)
	if config == nil {
		return nil, errors.New("Invalid Radius config from " + gateway)
	}
	return config, nil
}

func radiusEndpoint(gateway, path string) (string, error) {
	base, err := whatwg.Parse(string(jsonjs.StringCodePoints(gateway)))
	if err == nil {
		var parsed *whatwg.Url
		parsed, err = base.Parse(path)
		if err == nil {
			return parsed.Href(false), nil
		}
	}
	return "", fmt.Errorf("%q cannot be parsed as a URL against %q", path, gateway)
}
