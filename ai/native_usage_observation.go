package ai

import (
	"encoding/json"
	"math"
	"strconv"
)

// Presence of all explicit disjoint rates establishes price provenance, even
// when every rate is zero. Empty native-client metadata stays unknown. This
// does not change rate selection or any existing arithmetic.
func (c *completionsCost) UnmarshalJSON(raw []byte) error {
	type plain completionsCost
	var value plain
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	*c = completionsCost(value)
	var fields map[string]json.RawMessage
	_ = json.Unmarshal(raw, &fields)
	c.observed = explicitRates(fields)
	var tiers []map[string]json.RawMessage
	if len(fields["tiers"]) > 0 {
		if err := json.Unmarshal(fields["tiers"], &tiers); err != nil {
			c.observed = false
		}
		for _, tier := range tiers {
			if !explicitRates(tier) || !validObservedNumber(tier["inputTokensAbove"]) {
				c.observed = false
			}
		}
	}
	return nil
}
func explicitRates(fields map[string]json.RawMessage) bool {
	for _, key := range []string{"input", "output", "cacheRead", "cacheWrite"} {
		if !validObservedNumber(fields[key]) {
			return false
		}
	}
	return true
}
func validObservedNumber(raw json.RawMessage) bool {
	v, err := strconv.ParseFloat(string(raw), 64)
	return err == nil && !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0
}
func observeWireUsage(u *Usage, fields map[string]json.RawMessage, required ...string) {
	if u == nil {
		return
	}
	for _, key := range required {
		if !validObservedNumber(fields[key]) {
			u.Observation.UsageObserved = false
			u.Observation.UsageSource = "invalid"
			return
		}
	}
	if u.Observation.UsageSource != "invalid" {
		u.Observation.UsageObserved = true
		u.Observation.UsageSource = "wire"
	}
}

func observeOptionalWireUsage(u *Usage, fields map[string]json.RawMessage, keys ...string) {
	for _, key := range keys {
		if raw, present := fields[key]; present && !validObservedNumber(raw) {
			u.Observation.UsageObserved = false
			u.Observation.UsageSource = "invalid"
		}
	}
}

// A decoded empty usage object cannot establish observed zero counters.
// Keep native Antigravity arithmetic unchanged while retaining wire presence.
type nativeAntigravityUsage struct {
	PromptTokenCount, CandidatesTokenCount, CachedContentTokenCount, ThoughtsTokenCount float64
	observation                                                                         UsageObservation
}

func (u *nativeAntigravityUsage) UnmarshalJSON(raw []byte) error {
	type plain nativeAntigravityUsage
	var decoded plain
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*u = nativeAntigravityUsage(decoded)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	usage := &Usage{}
	observeWireUsage(usage, fields, "promptTokenCount", "candidatesTokenCount")
	observeOptionalWireUsage(usage, fields, "cachedContentTokenCount", "thoughtsTokenCount")
	u.observation = usage.Observation
	return nil
}
