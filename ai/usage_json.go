package ai

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"strconv"

	"github.com/2found/2ai/jsonjs"
)

// Retain negative zero until presence restoration has compared the current
// value with its decoded baseline. The final stringify then exports it as 0.
// Nonfinite numbers are valid live JS values and export as null.
type usageNumber float64

func (n usageNumber) MarshalJSON() ([]byte, error) {
	if math.IsNaN(float64(n)) || math.IsInf(float64(n), 0) {
		return []byte("null"), nil
	}
	return json.Marshal(float64(n))
}

func (n *usageNumber) UnmarshalJSON(raw []byte) error {
	raw = bytes.TrimSpace(raw)
	if bytes.Equal(raw, []byte("null")) {
		*n = 0
		return nil
	}
	value, err := strconv.ParseFloat(string(raw), 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		// Preserve the typed boundary for non-number JSON. Overflow itself is
		// valid JSON.parse input and must retain +/-Infinity in memory.
		return json.Unmarshal(raw, new(float64))
	}
	*n = usageNumber(value)
	return nil
}

type usageJSON struct {
	Input        usageNumber  `json:"input"`
	Output       usageNumber  `json:"output"`
	CacheRead    usageNumber  `json:"cacheRead"`
	CacheWrite   usageNumber  `json:"cacheWrite"`
	CacheWrite1h *usageNumber `json:"cacheWrite1h,omitempty"`
	Reasoning    *usageNumber `json:"reasoning,omitempty"`
	TotalTokens  usageNumber  `json:"totalTokens"`
	Cost         rawUsageCost `json:"cost"`
}

type usageCostJSON struct {
	Input      usageNumber `json:"input"`
	Output     usageNumber `json:"output"`
	CacheRead  usageNumber `json:"cacheRead"`
	CacheWrite usageNumber `json:"cacheWrite"`
	Total      usageNumber `json:"total"`
}

// Nested cost must keep -0 until the enclosing usage restores field presence.
// Public cost JSON still normalizes it, just like a standalone JS object.
type rawUsageCost UsageCost

func (c rawUsageCost) MarshalJSON() ([]byte, error) { return UsageCost(c).marshalJSON() }
func (c *rawUsageCost) UnmarshalJSON(raw []byte) error {
	return (*UsageCost)(c).UnmarshalJSON(raw)
}

func (u Usage) MarshalJSON() ([]byte, error) {
	wire := usageJSON{usageNumber(u.Input), usageNumber(u.Output), usageNumber(u.CacheRead), usageNumber(u.CacheWrite),
		(*usageNumber)(u.CacheWrite1h), (*usageNumber)(u.Reasoning), usageNumber(u.TotalTokens), rawUsageCost(u.Cost)}
	raw, err := json.Marshal(wire)
	raw, err = restoreTranscriptEncoding(raw, err, u.encoding)
	if err != nil {
		return nil, err
	}
	return jsonjs.StringifyJSON(raw)
}

func (u *Usage) UnmarshalJSON(raw []byte) error {
	*u = Usage{}
	var wire usageJSON
	if err := json.Unmarshal(raw, &wire); err != nil {
		return err
	}
	*u = Usage{Input: float64(wire.Input), Output: float64(wire.Output), CacheRead: float64(wire.CacheRead), CacheWrite: float64(wire.CacheWrite),
		CacheWrite1h: (*float64)(wire.CacheWrite1h), Reasoning: (*float64)(wire.Reasoning), TotalTokens: float64(wire.TotalTokens), Cost: UsageCost(wire.Cost)}
	canonical, err := jsonjs.StringifyJSON(raw)
	if err == nil {
		u.encoding, err = captureTranscriptEncoding(canonical, u)
	}
	return err
}

func (c UsageCost) marshalJSON() ([]byte, error) {
	wire := usageCostJSON{usageNumber(c.Input), usageNumber(c.Output), usageNumber(c.CacheRead), usageNumber(c.CacheWrite), usageNumber(c.Total)}
	raw, err := json.Marshal(wire)
	return restoreTranscriptEncoding(raw, err, c.encoding)
}

func (c UsageCost) MarshalJSON() ([]byte, error) {
	raw, err := c.marshalJSON()
	if err != nil {
		return nil, err
	}
	return jsonjs.StringifyJSON(raw)
}

func (c *UsageCost) UnmarshalJSON(raw []byte) error {
	*c = UsageCost{}
	var wire usageCostJSON
	if err := json.Unmarshal(raw, &wire); err != nil {
		return err
	}
	*c = UsageCost{Input: float64(wire.Input), Output: float64(wire.Output), CacheRead: float64(wire.CacheRead), CacheWrite: float64(wire.CacheWrite), Total: float64(wire.Total)}
	canonical, err := jsonjs.StringifyJSON(raw)
	if err == nil {
		c.encoding, err = captureTranscriptEncoding(canonical, c)
	}
	return err
}
