// Package layering enforces the module's layer rule from import paths alone.
//
//	core    imports only core
//	feature imports core and itself; one unit never imports another unit
//	        (a unit is feature/plugins/<name>, or feature/<name> otherwise)
//	compose imports core and feature
//	test    may import anything; no production package imports test/
package layering

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const modulePath = "github.com/2found/2ai"

var roots = []string{"core", "feature", "compose", "test", "examples"}

// unit names the isolation unit of a module-relative directory.
func unit(dir string) string {
	parts := strings.Split(dir, "/")
	if len(parts) >= 3 && parts[0] == "feature" && parts[1] == "plugins" {
		return strings.Join(parts[:3], "/")
	}
	if len(parts) >= 2 && parts[0] == "feature" {
		return strings.Join(parts[:2], "/")
	}
	return parts[0]
}

// violations checks a graph of module-relative directory -> module-relative
// directories it imports in production (non-test) files.
func violations(graph map[string][]string) []string {
	var out []string
	for dir, imps := range graph {
		layer := strings.SplitN(dir, "/", 2)[0]
		allowedRoot := false
		for _, r := range roots {
			allowedRoot = allowedRoot || r == layer
		}
		if !allowedRoot {
			out = append(out, dir+": package directory outside core/, feature/, compose/, test/, examples/")
			continue
		}
		for _, imp := range imps {
			il := strings.SplitN(imp, "/", 2)[0]
			switch {
			case il == "test" && layer != "test":
				out = append(out, dir+" imports "+imp+": production code must not import test/")
			case layer == "core" && il != "core":
				out = append(out, dir+" imports "+imp+": core imports only core")
			case layer == "feature" && il == "feature" && unit(dir) != unit(imp):
				out = append(out, dir+" imports "+imp+": a feature unit must not import another feature unit")
			case layer == "feature" && il != "core" && il != "feature":
				out = append(out, dir+" imports "+imp+": feature imports only core and itself")
			}
		}
	}
	sort.Strings(out)
	return out
}

func TestModuleLayering(t *testing.T) {
	root := filepath.Join("..", "..")
	graph := map[string][]string{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path != root && (name == "testdata" || name == "third_party" || strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(root, filepath.Dir(path))
		rel = filepath.ToSlash(rel)
		if _, ok := graph[rel]; !ok {
			graph[rel] = nil
		}
		if strings.HasSuffix(name, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, spec := range f.Imports {
			p, _ := strconv.Unquote(spec.Path.Value)
			if p == modulePath || strings.HasPrefix(p, modulePath+"/") {
				graph[rel] = append(graph[rel], strings.TrimPrefix(strings.TrimPrefix(p, modulePath), "/"))
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil || len(graph) < 20 {
		t.Fatalf("walked %d package directories from %s: the check is not looking at the module", len(graph), root)
	}
	for _, v := range violations(graph) {
		t.Error(v)
	}
}

func TestViolationsRefuseBadGraphs(t *testing.T) {
	good := map[string][]string{
		"core/ai":                {},
		"core/agentcore":         {"core/ai"},
		"feature/plugins/todo":   {"core/agentcore"},
		"feature/sandbox":        {"core/agentcore"},
		"feature/telemetry/llm":  {"core/telemetry"},
		"compose/preset":         {"core/agentcore", "feature/plugins/todo", "feature/sandbox"},
		"test/agentcore":         {"core/agentcore", "feature/sandbox"},
		"examples/scripted":      {"compose/preset"},
		"feature/plugins/todo/x": {"feature/plugins/todo"},
	}
	if v := violations(good); len(v) != 0 {
		t.Fatalf("good graph refused: %v", v)
	}
	bad := map[string]map[string][]string{
		"core imports feature":      {"core/ai": {"feature/sandbox"}},
		"feature unit imports unit": {"feature/plugins/todo": {"feature/plugins/goal"}},
		"feature imports sibling":   {"feature/sandbox": {"feature/credential"}},
		"feature imports compose":   {"feature/sandbox": {"compose/preset"}},
		"production imports test":   {"compose/preset": {"test/agentcore"}},
		"core imports test":         {"core/ai": {"test/telemetry"}},
		"dir outside roots":         {"jsonjs": {}},
	}
	for name, g := range bad {
		if len(violations(g)) == 0 {
			t.Errorf("%s: violation not refused", name)
		}
	}
}
