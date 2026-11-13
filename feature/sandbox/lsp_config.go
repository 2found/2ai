package sandbox

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
)

const (
	ToolLSP       = "lsp"
	maxLSPServers = 20
)

// LSPConfig is operator-authored configuration for the read-only language
// intelligence tool. Binaries are deliberately provisioned out of band: the
// agent may use a configured server, but never install or download one.
type LSPConfig struct {
	Servers        []LSPServerConfig `json:"servers"`
	TimeoutSeconds int               `json:"timeout_seconds,omitempty"`
}

type LSPServerConfig struct {
	Name                  string         `json:"name"`
	Command               string         `json:"command"`
	Args                  []string       `json:"args,omitempty"`
	Extensions            []string       `json:"extensions"`
	LanguageID            string         `json:"language_id,omitempty"`
	DiagnosticsOnly       bool           `json:"diagnostics_only,omitempty"`
	Image                 string         `json:"image,omitempty"`
	InitializationOptions map[string]any `json:"initialization_options,omitempty"`
	Settings              map[string]any `json:"settings,omitempty"`
}

func ParseLSPConfig(raw string) (LSPConfig, error) {
	var cfg LSPConfig
	if strings.TrimSpace(raw) == "" {
		return cfg, fmt.Errorf("lsp config is required; configure at least one language server")
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("invalid lsp config: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return cfg, fmt.Errorf("invalid lsp config: expected one JSON object")
	}
	if err := cfg.validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func ValidateLSPConfig(raw string) error {
	_, err := ParseLSPConfig(raw)
	return err
}

func (c *LSPConfig) validate() error {
	if len(c.Servers) == 0 {
		return fmt.Errorf("invalid lsp config: servers must contain at least one server")
	}
	if len(c.Servers) > maxLSPServers {
		return fmt.Errorf("invalid lsp config: got %d servers, maximum is %d", len(c.Servers), maxLSPServers)
	}
	if c.TimeoutSeconds == 0 {
		c.TimeoutSeconds = 20
	}
	if c.TimeoutSeconds < 2 || c.TimeoutSeconds > 120 {
		return fmt.Errorf("invalid lsp config: timeout_seconds must be between 2 and 120")
	}
	names := make(map[string]bool, len(c.Servers))
	for i := range c.Servers {
		s := &c.Servers[i]
		s.Name = strings.TrimSpace(s.Name)
		s.Command = strings.TrimSpace(s.Command)
		s.LanguageID = strings.TrimSpace(s.LanguageID)
		s.Image = strings.TrimSpace(s.Image)
		if s.Name == "" || s.Command == "" {
			return fmt.Errorf("invalid lsp config: server %d requires non-empty name and command", i+1)
		}
		if names[s.Name] {
			return fmt.Errorf("invalid lsp config: duplicate server name %q", s.Name)
		}
		names[s.Name] = true
		if len(s.Extensions) == 0 {
			return fmt.Errorf("invalid lsp config: server %q requires at least one extension", s.Name)
		}
		seenExt := map[string]bool{}
		for j, ext := range s.Extensions {
			ext = strings.ToLower(strings.TrimSpace(ext))
			if ext == "" || ext[0] != '.' || strings.ContainsAny(ext, `/\\`) || filepath.Clean(ext) != ext {
				return fmt.Errorf("invalid lsp config: server %q has invalid extension %q", s.Name, s.Extensions[j])
			}
			if seenExt[ext] {
				return fmt.Errorf("invalid lsp config: server %q repeats extension %q", s.Name, ext)
			}
			seenExt[ext] = true
			s.Extensions[j] = ext
		}
		if s.LanguageID == "" {
			s.LanguageID = inferLanguageID(s.Extensions[0])
		}
	}
	return nil
}

func inferLanguageID(ext string) string {
	switch strings.ToLower(ext) {
	case ".go":
		return "go"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx", ".mjs", ".cjs":
		return "javascript"
	case ".py":
		return "python"
	case ".rs":
		return "rust"
	case ".java":
		return "java"
	case ".c", ".h":
		return "c"
	case ".cc", ".cpp", ".cxx", ".hpp":
		return "cpp"
	case ".json":
		return "json"
	case ".html", ".htm":
		return "html"
	case ".css", ".scss", ".less":
		return strings.TrimPrefix(strings.ToLower(ext), ".")
	default:
		return strings.TrimPrefix(strings.ToLower(ext), ".")
	}
}

func (c LSPConfig) matchingServers(path string, navigation bool) []LSPServerConfig {
	ext := strings.ToLower(filepath.Ext(path))
	out := make([]LSPServerConfig, 0, len(c.Servers))
	for _, server := range c.Servers {
		if navigation && server.DiagnosticsOnly {
			continue
		}
		for _, candidate := range server.Extensions {
			if ext == candidate {
				out = append(out, server)
				break
			}
		}
	}
	return out
}

func (c LSPConfig) serverNames() []string {
	names := make([]string, 0, len(c.Servers))
	for _, server := range c.Servers {
		names = append(names, server.Name)
	}
	sort.Strings(names)
	return names
}
