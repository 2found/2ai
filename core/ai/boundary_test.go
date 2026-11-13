package ai

import (
	"go/build"
	"strings"
	"testing"
)

func TestAIAndProtocolDoNotImportAgentcore(t *testing.T) {
	for _, dir := range []string{".", "protocol", "jscompat"} {
		pkg, err := build.ImportDir(dir, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, imported := range pkg.Imports {
			if strings.HasPrefix(imported, "github.com/2found/2ai/core/agentcore") {
				t.Errorf("%s imports %s: providers must remain below the agent runtime", dir, imported)
			}
		}
	}
}

// jscompat is the leaf of the core/ai tree: it may import core/jsonjs and the
// standard library, never core/ai or any sibling provider package.
func TestJSCompatIsALeaf(t *testing.T) {
	pkg, err := build.ImportDir("jscompat", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, imported := range pkg.Imports {
		if strings.HasPrefix(imported, "github.com/2found/2ai/") && imported != "github.com/2found/2ai/core/jsonjs" {
			t.Errorf("jscompat imports %s: it may depend only on core/jsonjs", imported)
		}
	}
}
