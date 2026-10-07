package ai

import (
	"github.com/2found/2ai/telemetry"
	"math"
)

// UsageObservation distinguishes a wire/explicit observation from initialized
// zero values. Custom producers must supply explicit evidence; JSON decode and
// model cost-object presence cannot establish it.
type UsageObservation struct {
	UsageObserved   bool
	PricingObserved bool
	UsageSource     string
	PricingSource   string
}

func safeNativeAccounting(message *Message) telemetry.SafeAccounting {
	a := telemetry.SafeAccounting{AttemptCoverage: "observed"}
	if message == nil || message.Usage == nil {
		return a
	}
	u := message.Usage
	a.UsageObserved, a.PricingObserved = u.Observation.UsageObserved, u.Observation.PricingObserved
	a.UsageSource, a.PricingSource = u.Observation.UsageSource, u.Observation.PricingSource
	for _, component := range []float64{u.Cost.Input, u.Cost.Output, u.Cost.CacheRead, u.Cost.CacheWrite, u.Cost.Total} {
		if component < 0 || math.IsNaN(component) || math.IsInf(component, 0) {
			a.PricingObserved = false
			a.PricingSource = "invalid"
		}
	}
	if u.CacheWrite1h != nil && (*u.CacheWrite1h < 0 || *u.CacheWrite1h > u.CacheWrite || math.IsNaN(*u.CacheWrite1h) || math.IsInf(*u.CacheWrite1h, 0)) {
		a.PricingObserved = false
		a.PricingSource = "invalid"
	}
	a.Tokens = &telemetry.SafeTokens{Input: u.Input, Output: u.Output, CacheRead: u.CacheRead, CacheWrite: u.CacheWrite}
	cost := u.Cost.Total
	a.KnownCostUSD = &cost
	return a
}

func finishSafeNative(attempt *telemetry.SafeAttempt, status, kind string, message *Message, admitted bool) {
	provider, model := "", ""
	if message != nil {
		provider, model = message.Provider, message.Model
		if message.ResponseModel != nil {
			model = *message.ResponseModel
		}
	}
	attempt.FinishAdmission(status, kind, safeNativeAccounting(message), provider, model, admitted)
}
