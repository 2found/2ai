package agentcore_test

import (
	"go/build"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modulePath is this repo's module path. Anything under it is "our own code",
// which is exactly what the kernel is not allowed to reach for.
const modulePath = "github.com/2found/2ai"

// pluginsPrefix is the directory the rule is about.
const pluginsPrefix = modulePath + "/agentcore/plugins/"

// TestKernelNamesNoPlugin enforces plugins/README.md's one rule as a build
// artifact rather than a promise:
//
//	The loop names no plugin. Delete every package under agentcore/plugins and
//	agentcore still compiles and runs — it just does less.
//
// Go's own import cycle check covers most of this for free: every plugin today
// imports agentcore, so importing one back is already a build error. What it
// does NOT cover is a plugin package that does not import the kernel — a pure
// data or prompt helper under plugins/ — which the kernel could reach for with
// no cycle and no complaint. That is the case this test exists for, and it is
// the likely shape of the first violation, because it is the one that compiles.
//
// It checks the package's DIRECT imports, which is sufficient rather than
// sloppy: the runtime dependency checks below keep provider and host modules
// free of application policy and plugin registration.
func TestKernelNamesNoPlugin(t *testing.T) {
	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatalf("reading the agentcore package: %v", err)
	}
	for _, imp := range pkg.Imports {
		if strings.HasPrefix(imp, pluginsPrefix) {
			t.Errorf("package agentcore imports %s — the kernel must not name a plugin.\n"+
				"Reach it through an interface in extension.go (ToolInterceptor, StepInterceptor, "+
				"ExtensionFactory, …) and let the composition supply it.", imp)
		}
	}
}

// TestKernelRuntimeDependencies keeps composition above the native runtime
// modules. Providers depend on ai/protocol, never back on the agent facade.
func TestKernelRuntimeDependencies(t *testing.T) {
	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatalf("reading the agentcore package: %v", err)
	}
	for _, imp := range pkg.Imports {
		if strings.HasPrefix(imp, modulePath+"/") {
			allowed := map[string]bool{
				modulePath + "/ai/protocol":      true,
				modulePath + "/ai":               true,
				modulePath + "/agentcore/engine": true,
				modulePath + "/agentcore/host":   true,
				modulePath + "/telemetry":        true,
				modulePath + "/jsonjs":           true,
			}
			if !allowed[imp] {
				t.Errorf("agentcore imports application/plugin package %s", imp)
			}
		}
	}

	// Guard against a vacuous pass: a package that read as having no imports at
	// all would satisfy every check above.
	if len(pkg.Imports) == 0 {
		t.Fatal("read 0 imports for package agentcore — the check is not looking at the real package")
	}
	if len(pkg.GoFiles) < 10 {
		t.Fatalf("read only %d Go files for package agentcore — the check is not looking at the real package", len(pkg.GoFiles))
	}
}

// TestKernelTreeHoldsOnlyDeclaredBoundaries enforces README.md's opening claim:
// the runtime is one flat package, plugins are ejectable capabilities, and
// integration is a black-box test suite. engine is the native Pi port, with
// its own dependency boundary while callers migrate from the flat kernel.
// host exposes reusable native policy/checkpoint utilities to external consumers.
//
// The rule matters more than tidiness. "The kernel" has to name a tree a reader
// can enumerate, or the boundary tests above are checking one package while the
// directory quietly accumulates others that inherit the kernel's reputation
// without its constraints. A package that extends a running agent belongs in
// plugins/; a test that composes several public packages belongs in
// integration/. Production packages that merely import agentcore belong beside
// it (authoring/, ai/, sandbox/).
func TestKernelTreeHoldsOnlyDeclaredBoundaries(t *testing.T) {
	allowed := map[string]bool{"plugins": true, "integration": true, "engine": true, "host": true}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading the agentcore directory: %v", err)
	}
	seen := map[string]bool{}
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") || strings.HasPrefix(e.Name(), "_") {
			continue
		}
		if allowed[e.Name()] {
			seen[e.Name()] = true
			continue
		}
		t.Errorf("agentcore/%s/ is a subdirectory of the kernel.\n"+
			"Only engine/, plugins/ and the black-box integration/ suite may live here. If it extends "+
			"a running agent it is a plugin; otherwise it belongs beside agentcore.", e.Name())
	}
	for name := range allowed {
		if !seen[name] {
			t.Errorf("found no %s/ directory — the check is not looking at the real tree", name)
		}
	}
}

// The native port may consume provider-neutral AI contracts and telemetry, but
// cannot reach application policy, storage, plugins, or the legacy core directly.
func TestNativeEngineNamesNoHost(t *testing.T) {
	pkg, err := build.ImportDir("engine", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, imp := range pkg.Imports {
		if strings.HasPrefix(imp, modulePath+"/") && imp != modulePath+"/ai" && imp != modulePath+"/telemetry" && imp != modulePath+"/jsonjs" {
			t.Errorf("native engine imports host/legacy package %s", imp)
		}
		if imp == "os/exec" {
			t.Error("native engine must not launch a runtime subprocess")
		}
	}
}

// Shared host utilities must be importable by the root facade and by Soot,
// without importing either application's runtime or the facade itself.
func TestNativeHostNamesNoApplication(t *testing.T) {
	pkg, err := build.ImportDir("host", 0)
	if err != nil {
		t.Fatal(err)
	}
	allowed := map[string]bool{
		modulePath + "/ai/protocol":      true,
		modulePath + "/ai":               true,
		modulePath + "/agentcore/engine": true,
		modulePath + "/telemetry":        true,
		modulePath + "/jsonjs":           true,
	}
	for _, imported := range pkg.Imports {
		if strings.HasPrefix(imported, modulePath+"/") && !allowed[imported] {
			t.Errorf("native host utility imports facade/application package %s", imported)
		}
	}
}

// TestPluginsDoNotNameEachOther enforces the second half of plugins/README.md's
// rule, the half Go's cycle check cannot cover:
//
//	Plugins do not name each other. Where one capability needs to know something
//	about another, the question is asked generically (RunInfo.Bookkeeping), never
//	by importing the other package.
//
// Nothing prevents these imports mechanically — plugins are siblings, so any of
// them may import any other with no cycle and no complaint. That makes this the
// cheapest rule in the system to break and the most expensive to unwind: once
// the repeat guard imports the todo plugin to ask whether a call was
// bookkeeping, ejecting todo stops being a composition change and becomes a
// build error, and the property the whole directory exists for is gone.
//
// preset is the one exemption, and it is exempt because it is not a capability:
// it is the aggregator that names the default set so a caller does not have to.
func TestPluginsDoNotNameEachOther(t *testing.T) {
	const aggregator = "preset"

	dirs, err := os.ReadDir("plugins")
	if err != nil {
		t.Fatalf("reading agentcore/plugins: %v", err)
	}
	var checked int
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		name := d.Name()
		pkg, err := build.ImportDir(filepath.Join("plugins", name), 0)
		if err != nil {
			continue // not a Go package (docs-only folder); nothing to check
		}
		checked++
		if name == aggregator {
			continue
		}
		for _, imp := range pkg.Imports {
			if strings.HasPrefix(imp, pluginsPrefix) {
				t.Errorf("plugins/%s imports %s — plugins must not name each other.\n"+
					"Ask the question generically instead (RunInfo.Bookkeeping, an extension "+
					"interface in agentcore/extension.go), so either plugin can be ejected "+
					"without touching the other.", name, imp)
			}
		}
	}
	if checked < 10 {
		t.Fatalf("read only %d plugin packages — the check is not looking at the real tree", checked)
	}
}

// TestEveryPluginDocumentsItself keeps plugins/README.md's opening promise —
// "each folder has a README.md explaining what it does to the agent" — true for
// folders added later than the sentence.
//
// A capability's cost is invisible at its call site: a plugin list says which
// capabilities are installed and nothing about what each one puts in the prompt,
// what it does to the provider's prefix cache, or what it cannot do. The folder
// README is where that is written down, so an undocumented plugin is one a
// composer cannot price.
func TestEveryPluginDocumentsItself(t *testing.T) {
	dirs, err := os.ReadDir("plugins")
	if err != nil {
		t.Fatalf("reading agentcore/plugins: %v", err)
	}
	var checked int
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		if _, err := build.ImportDir(filepath.Join("plugins", d.Name()), 0); err != nil {
			continue // not a Go package
		}
		checked++
		readme := filepath.Join("plugins", d.Name(), "README.md")
		if _, err := os.Stat(readme); err != nil {
			t.Errorf("plugins/%s has no README.md.\n"+
				"Say what it does to the agent: what the model sees, what it costs in "+
				"tokens, what it does to the prefix cache, and what it cannot do.", d.Name())
		}
	}
	if checked < 10 {
		t.Fatalf("read only %d plugin packages — the check is not looking at the real tree", checked)
	}
}

// TestEveryKernelFileJustifiesItself keeps README.md's file map honest.
//
// The map is the answer to the only organizing question a flat package has —
// "why is this file in the kernel and not a plugin?" — and an unlisted file is
// a file nobody had to answer it for. That is how a kernel grows a plugin's
// worth of code back: one unremarkable addition at a time, each individually
// fine, none of them ever compared against the rule.
//
// So the map is not documentation of the package; it is a gate on adding to it.
func TestEveryKernelFileJustifiesItself(t *testing.T) {
	readme, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("reading agentcore/README.md: %v", err)
	}
	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatalf("reading the agentcore package: %v", err)
	}
	if len(pkg.GoFiles) < 10 {
		t.Fatalf("read only %d Go files — the check is not looking at the real package", len(pkg.GoFiles))
	}
	doc := string(readme)
	for _, f := range pkg.GoFiles {
		if !strings.Contains(doc, "`"+f+"`") {
			t.Errorf("%s is in the kernel but not in agentcore/README.md.\n"+
				"Add a row saying why it is core — loop, contract, log, or seam default. "+
				"If it cannot claim one, it belongs in a plugin.", f)
		}
	}
}
