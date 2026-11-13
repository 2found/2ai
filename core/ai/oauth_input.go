package ai

import (
	"math"
	"strings"

	"github.com/2found/2ai/core/ai/jscompat"
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
)

func parseOAuthAuthorizationInput(input string) (string, *string) {
	value := strings.TrimFunc(input, jscompat.IsWhitespace)
	if value == "" {
		return "", nil
	}
	var params map[string]string
	if parsed, err := whatwg.Parse(value); err == nil {
		params = jscompat.QueryValues(parsed.Query())
	} else if strings.Contains(value, "#") {
		parts := strings.SplitN(value, "#", 3)
		return parts[0], &parts[1]
	} else if strings.Contains(value, "code=") {
		params = jscompat.QueryValues(strings.TrimPrefix(value, "?"))
	} else {
		return value, nil
	}
	if state, present := params["state"]; present {
		return params["code"], &state
	}
	return params["code"], nil
}

func oauthPositiveNumber(value any) bool {
	number, ok := deviceCodeNumber(value)
	return ok && !math.IsNaN(number) && !math.IsInf(number, 0) && number > 0
}
