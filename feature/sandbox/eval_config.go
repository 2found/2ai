package sandbox

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	ToolEval                    = "eval"
	defaultEvalTimeoutSeconds   = 30
	defaultEvalMaxOutputBytes   = 50 * 1024
	defaultEvalMaxBridgeCalls   = 16
	maxEvalBridgeCalls          = 64
	maxEvalOutputBytes          = 1024 * 1024
	maxEvalCellTimeoutSeconds   = 300
	defaultEvalRegistryCapacity = 16
)

// EvalConfig selects the operator-provisioned Python and JavaScript runtimes
// used by the persistent eval tool. The zero-value JSON object is useful on a
// laptop (python3 and node on PATH); a server normally supplies language images
// or the combined AgentRay eval image.
type EvalConfig struct {
	Python         EvalPythonConfig     `json:"python,omitempty"`
	JavaScript     EvalJavaScriptConfig `json:"javascript,omitempty"`
	TimeoutSeconds int                  `json:"timeout_seconds,omitempty"`
	MaxOutputBytes int                  `json:"max_output_bytes,omitempty"`
	MaxBridgeCalls int                  `json:"max_bridge_calls,omitempty"`
}

type EvalPythonConfig struct {
	Command string   `json:"command,omitempty"`
	Args    []string `json:"args,omitempty"`
	Image   string   `json:"image,omitempty"`
}

type EvalJavaScriptConfig struct {
	Command string   `json:"command,omitempty"`
	Args    []string `json:"args,omitempty"`
	Image   string   `json:"image,omitempty"`
}

func ParseEvalConfig(raw string) (EvalConfig, error) {
	var cfg EvalConfig
	if strings.TrimSpace(raw) == "" {
		raw = "{}"
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("invalid eval config: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return cfg, fmt.Errorf("invalid eval config: expected one JSON object")
	}
	if err := cfg.validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func ValidateEvalConfig(raw string) error {
	_, err := ParseEvalConfig(raw)
	return err
}

func (c *EvalConfig) validate() error {
	c.Python.Command = strings.TrimSpace(c.Python.Command)
	c.Python.Image = strings.TrimSpace(c.Python.Image)
	if c.Python.Command == "" {
		c.Python.Command = "python3"
	}
	c.JavaScript.Command = strings.TrimSpace(c.JavaScript.Command)
	c.JavaScript.Image = strings.TrimSpace(c.JavaScript.Image)
	if c.JavaScript.Command == "" {
		c.JavaScript.Command = "node"
	}
	if err := validateEvalRuntime("python", c.Python.Command, c.Python.Args); err != nil {
		return err
	}
	if err := validateEvalRuntime("javascript", c.JavaScript.Command, c.JavaScript.Args); err != nil {
		return err
	}
	if c.TimeoutSeconds == 0 {
		c.TimeoutSeconds = defaultEvalTimeoutSeconds
	}
	if c.TimeoutSeconds < 1 || c.TimeoutSeconds > maxEvalCellTimeoutSeconds {
		return fmt.Errorf("invalid eval config: timeout_seconds must be between 1 and %d", maxEvalCellTimeoutSeconds)
	}
	if c.MaxOutputBytes == 0 {
		c.MaxOutputBytes = defaultEvalMaxOutputBytes
	}
	if c.MaxOutputBytes < 4096 || c.MaxOutputBytes > maxEvalOutputBytes {
		return fmt.Errorf("invalid eval config: max_output_bytes must be between 4096 and %d", maxEvalOutputBytes)
	}
	if c.MaxBridgeCalls == 0 {
		c.MaxBridgeCalls = defaultEvalMaxBridgeCalls
	}
	if c.MaxBridgeCalls < 1 || c.MaxBridgeCalls > maxEvalBridgeCalls {
		return fmt.Errorf("invalid eval config: max_bridge_calls must be between 1 and %d", maxEvalBridgeCalls)
	}
	return nil
}

func validateEvalRuntime(name, command string, args []string) error {
	if strings.IndexByte(command, 0) >= 0 {
		return fmt.Errorf("invalid eval config: %s.command contains NUL", name)
	}
	for i, arg := range args {
		if strings.IndexByte(arg, 0) >= 0 {
			return fmt.Errorf("invalid eval config: %s.args[%d] contains NUL", name, i)
		}
	}
	return nil
}
