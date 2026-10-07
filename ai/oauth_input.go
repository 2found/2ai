package ai

import (
	whatwg "github.com/nationallibraryofnorway/whatwg-url/url"
	"math"
	"strings"
)

func parseOAuthAuthorizationInput(input string) (string, *string) {
	value := strings.TrimFunc(input, jsWhitespace)
	if value == "" {
		return "", nil
	}
	var params map[string]string
	if parsed, err := whatwg.Parse(value); err == nil {
		params = urlQueryValues(parsed.Query())
	} else if strings.Contains(value, "#") {
		parts := strings.SplitN(value, "#", 3)
		return parts[0], &parts[1]
	} else if strings.Contains(value, "code=") {
		params = urlQueryValues(strings.TrimPrefix(value, "?"))
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
