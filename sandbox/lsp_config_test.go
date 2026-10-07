package sandbox

import (
	"strings"
	"testing"
)

func TestParseLSPConfigDefaultsAndMatches(t *testing.T) {
	cfg, err := ParseLSPConfig(`{"servers":[
		{"name":"gopls","command":"gopls","extensions":[".GO"]},
		{"name":"golangci","command":"golangci-lint-langserver","extensions":[".go"],"diagnostics_only":true}
	]}`)
	if err != nil {
		t.Fatalf("ParseLSPConfig: %v", err)
	}
	if cfg.TimeoutSeconds != 20 || cfg.Servers[0].LanguageID != "go" || cfg.Servers[0].Extensions[0] != ".go" {
		t.Fatalf("normalized config = %+v", cfg)
	}
	if got := cfg.matchingServers("main.go", true); len(got) != 1 || got[0].Name != "gopls" {
		t.Fatalf("navigation servers = %+v", got)
	}
	if got := cfg.matchingServers("main.go", false); len(got) != 2 {
		t.Fatalf("diagnostic servers = %+v", got)
	}
}

func TestParseLSPConfigRejectsUnsafeOrAmbiguousShapes(t *testing.T) {
	cases := []string{
		`{}`,
		`{"servers":[]}`,
		`{"servers":[{"name":"x","command":"x","extensions":[]}]}`,
		`{"servers":[{"name":"x","command":"x","extensions":["go"]}]}`,
		`{"servers":[{"name":"x","command":"x","extensions":["../go"]}]}`,
		`{"servers":[{"name":"x","command":"x","extensions":[".go"]},{"name":"x","command":"y","extensions":[".py"]}]}`,
		`{"servers":[{"name":"x","command":"x","extensions":[".go"]}],"unknown":true}`,
		`{"servers":[{"name":"x","command":"x","extensions":[".go"]}],"timeout_seconds":121}`,
	}
	for _, raw := range cases {
		if _, err := ParseLSPConfig(raw); err == nil || !strings.Contains(err.Error(), "lsp config") {
			t.Errorf("ParseLSPConfig(%s) error = %v", raw, err)
		}
	}
}
