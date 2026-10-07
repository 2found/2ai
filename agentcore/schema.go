package agentcore

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// outputValidator is the compiled, run-time half of OutputSchema. Keeping the
// compiler behind this small function type means the rest of the loop only
// knows the policy it must enforce: a final text answer either satisfies the
// configured schema or the run fails without accepting it as its result.
type outputValidator func(string) error

// toolValidatorEntry is the cached, compiled form of one tool's canonical
// parameter schema. Compilation failures are cached too: a broken dynamic tool
// schema fails closed consistently without doing expensive work on every call.
type toolValidatorEntry struct {
	fingerprint [sha256.Size]byte
	validator   *jsonschema.Schema
	err         error
}

// compileOutputValidator turns the provider-neutral schema into a local
// validator at build time. Providers may ignore ChatRequest.OutputSchema, so a
// successful build must establish that the fallback validator itself is valid
// before the first (potentially billable) model call is made.
func compileOutputValidator(spec *OutputSchema) (outputValidator, error) {
	if spec == nil {
		return nil, nil
	}
	if len(spec.Schema) == 0 {
		return nil, fmt.Errorf("agentcore: output schema must be a non-empty JSON Schema object")
	}

	// OutputSchema is commonly authored as Go maps containing []string and
	// integer values. Normalize it through JSON first so the compiler sees the
	// same JSON-native shapes a schema loaded from the wire would contain.
	raw, err := json.Marshal(spec.Schema)
	if err != nil {
		return nil, fmt.Errorf("agentcore: output schema is not JSON-serializable: %w", err)
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("agentcore: output schema is not valid JSON: %w", err)
	}

	const resource = "https://agentcore.local/output-schema.json"
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(nil) // local fragments only; validation must never fetch
	if err := compiler.AddResource(resource, doc); err != nil {
		return nil, fmt.Errorf("agentcore: output schema: %w", err)
	}
	compiled, err := compiler.Compile(resource)
	if err != nil {
		return nil, fmt.Errorf("agentcore: output schema does not compile: %w", err)
	}

	return func(answer string) error {
		doc, err := jsonschema.UnmarshalJSON(strings.NewReader(answer))
		if err != nil {
			return fmt.Errorf("agentcore: structured output is not valid JSON: %w", err)
		}
		if err := compiled.Validate(doc); err != nil {
			return fmt.Errorf("agentcore: structured output does not match schema: %w", err)
		}
		return nil
	}, nil
}

// validateToolArgs performs complete local JSON Schema validation against the
// canonical tool contract. Provider adapters may project a more conservative
// wire schema for compatibility, but that projection never becomes the
// execution policy enforced here.
func (ts *ToolSet) validateToolArgs(name, args string, schema map[string]any) error {
	value, err := parseToolArgs(args)
	if err != nil {
		return err
	}
	if len(schema) == 0 {
		return nil
	}

	// Preserve concise, self-correctable errors for the most common failures;
	// the compiled validator below remains authoritative for nested constraints.
	if obj, ok := value.(map[string]any); ok {
		if err := validateObject(obj, schema); err != nil {
			return err
		}
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("tool schema is not JSON-serializable: %w", err)
	}
	fingerprint := sha256.Sum256(raw)
	cache := ts.validators
	if cache == nil {
		// Defensive support for a zero-value ToolSet created inside this package.
		cache = newToolValidatorCache()
	}

	cache.mu.Lock()
	entry, ok := cache.entries[name]
	if !ok || entry.fingerprint != fingerprint {
		entry = compileToolValidator(raw, fingerprint)
		if !ok && len(cache.entries) >= maxToolValidatorCacheEntries {
			// Extension-contributed tools may be named dynamically across runs.
			// Evict one old entry rather than turning the Agent-owned cache into
			// unbounded server-process state; a miss merely recompiles a schema.
			for oldName := range cache.entries {
				delete(cache.entries, oldName)
				break
			}
		}
		cache.entries[name] = entry
	}
	cache.mu.Unlock()

	if entry.err != nil {
		return entry.err
	}
	if err := entry.validator.Validate(value); err != nil {
		return fmt.Errorf("arguments do not match tool schema: %w", err)
	}
	return nil
}

func parseToolArgs(args string) (any, error) {
	args = strings.TrimSpace(args)
	if args == "" {
		// Providers sometimes serialize a no-argument call as an empty string.
		// Treat it as the empty object that tool parameter schemas describe.
		return map[string]any{}, nil
	}
	value, err := jsonschema.UnmarshalJSON(strings.NewReader(args))
	if err != nil {
		return nil, fmt.Errorf("arguments are not valid JSON")
	}
	return value, nil
}

func compileToolValidator(raw []byte, fingerprint [sha256.Size]byte) toolValidatorEntry {
	entry := toolValidatorEntry{fingerprint: fingerprint}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		entry.err = fmt.Errorf("tool schema is not valid JSON: %w", err)
		return entry
	}

	const resource = "https://agentcore.local/tool-schema.json"
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	// A tool contract must be self-contained. The library has no default URL
	// loader, and setting nil explicitly makes the no-network policy clear.
	compiler.UseLoader(nil)
	if err := compiler.AddResource(resource, doc); err != nil {
		entry.err = fmt.Errorf("tool schema is invalid: %w", err)
		return entry
	}
	compiled, err := compiler.Compile(resource)
	if err != nil {
		entry.err = fmt.Errorf("tool schema does not compile: %w", err)
		return entry
	}
	entry.validator = compiled
	return entry
}
