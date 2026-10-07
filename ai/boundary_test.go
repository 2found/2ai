package ai

import (
	"go/build"
	"strings"
	"testing"
)

func TestAIAndProtocolDoNotImportAgentcore(t *testing.T) {
	for _, dir := range []string{".", "protocol"} {
		pkg, err := build.ImportDir(dir, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, imported := range pkg.Imports {
			if strings.HasPrefix(imported, "github.com/2found/2ai/agentcore") {
				t.Errorf("%s imports %s: providers must remain below the agent runtime", dir, imported)
			}
		}
	}
}
