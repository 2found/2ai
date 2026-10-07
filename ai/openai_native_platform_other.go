//go:build !darwin && !linux

package ai

// Match Pi's fallback when OS release metadata is unavailable.
func completionsUserAgent() string { return "pi (browser)" }
