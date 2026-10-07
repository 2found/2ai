package ai

import (
	"encoding/json"
	"errors"
)

// FallbackSelection records the selected candidate without credentials or pool
// handles. Provider-row identity disambiguates candidates with equal models.
type FallbackSelection struct {
	Version    int             `json:"version"`
	Generation uint64          `json:"generation"`
	Rung       int             `json:"rung"`
	ProviderID string          `json:"providerId"`
	Model      json.RawMessage `json:"model"`
}

// ParseFallbackSelection validates one committed transition after the previous
// selection. The host must additionally check the selected provider/model against
// its bound candidates before opening a request.
func ParseFallbackSelection(raw string, previous *FallbackSelection) (FallbackSelection, error) {
	var record FallbackSelection
	var fields map[string]json.RawMessage
	invalid := func() (FallbackSelection, error) {
		return record, errors.New("corrupt native ladder selection record")
	}
	if json.Unmarshal([]byte(raw), &record) != nil || json.Unmarshal([]byte(raw), &fields) != nil {
		return invalid()
	}
	for _, name := range []string{"version", "generation", "rung", "providerId", "model"} {
		if len(fields[name]) == 0 || string(fields[name]) == "null" {
			return invalid()
		}
	}
	var model struct{ ID, API, Provider string }
	if record.Version != 1 || record.Rung < 0 || json.Unmarshal(record.Model, &model) != nil || model.ID == "" || model.API == "" || model.Provider == "" {
		return invalid()
	}
	generation, prior := uint64(0), 0
	if previous != nil {
		generation, prior = previous.Generation, previous.Rung
	}
	if generation == ^uint64(0) || record.Generation != generation+1 || record.Rung == prior {
		return invalid()
	}
	return record, nil
}
