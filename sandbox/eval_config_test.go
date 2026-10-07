package sandbox

import (
	"strings"
	"testing"
)

func TestParseEvalConfigDefaultsAndValidation(t *testing.T) {
	cfg, err := ParseEvalConfig(`{}`)
	if err != nil {
		t.Fatalf("ParseEvalConfig: %v", err)
	}
	if cfg.Python.Command != "python3" || cfg.JavaScript.Command != "node" || cfg.TimeoutSeconds != 30 || cfg.MaxOutputBytes != 50*1024 || cfg.MaxBridgeCalls != 16 {
		t.Fatalf("defaults = %+v", cfg)
	}

	for _, raw := range []string{
		`{"unknown":true}`,
		`{} {}`,
		`{"timeout_seconds":301}`,
		`{"max_output_bytes":100}`,
		`{"max_bridge_calls":65}`,
		`{"javascript":{"command":"node\u0000bad"}}`,
		`{"javascript":{"args":["--eval\u0000bad"]}}`,
	} {
		if _, err := ParseEvalConfig(raw); err == nil {
			t.Errorf("ParseEvalConfig(%s) unexpectedly succeeded", raw)
		}
	}
}

func TestParseEvalConfigNormalizesRuntime(t *testing.T) {
	cfg, err := ParseEvalConfig(`{
		"python":{"command":"  /opt/python  ","args":["-X","dev"],"image":" python:3.12 "},
		"javascript":{"command":"  /opt/node  ","args":["--no-deprecation"],"image":" node:22 "},
		"timeout_seconds":12,"max_output_bytes":4096,"max_bridge_calls":7
	}`)
	if err != nil {
		t.Fatalf("ParseEvalConfig: %v", err)
	}
	if cfg.Python.Command != "/opt/python" || cfg.Python.Image != "python:3.12" || strings.Join(cfg.Python.Args, " ") != "-X dev" {
		t.Fatalf("runtime = %+v", cfg.Python)
	}
	if cfg.JavaScript.Command != "/opt/node" || cfg.JavaScript.Image != "node:22" || strings.Join(cfg.JavaScript.Args, " ") != "--no-deprecation" {
		t.Fatalf("javascript runtime = %+v", cfg.JavaScript)
	}
	if cfg.MaxBridgeCalls != 7 {
		t.Fatalf("max bridge calls = %d", cfg.MaxBridgeCalls)
	}
}
