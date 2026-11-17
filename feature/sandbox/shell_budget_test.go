package sandbox

import (
	"strings"
	"testing"

	"github.com/2found/2ai/core/agentcore"
)

// One noisy stream may show all of the 30,000-byte budget; two share it; a
// short stream is never cut to make room.
func TestShellOutputSharesOneBudget(t *testing.T) {
	render := func(stdout, stderr int) (string, bool) {
		o := newShellOutput(nil)
		o.stream(0).Write([]byte(strings.Repeat("#", stdout)))
		o.stream(1).Write([]byte(strings.Repeat("@", stderr)))
		return o.format(agentcore.SandboxResult{}), o.truncated()
	}
	out, cut := render(25_000, 100)
	if cut || strings.Count(out, "#") != 25_000 || strings.Count(out, "@") != 100 {
		t.Fatalf("25 KB fits whole: cut=%v o=%d", cut, strings.Count(out, "#"))
	}
	out, cut = render(80_000, 100)
	if !cut || strings.Count(out, "@") != 100 || strings.Count(out, "#") != shellOutputBudget-100 {
		t.Fatalf("noisy stdout: cut=%v o=%d e=%d", cut, strings.Count(out, "#"), strings.Count(out, "@"))
	}
	out, _ = render(80_000, 80_000)
	if strings.Count(out, "#") != shellOutputBudget/2 || strings.Count(out, "@") != shellOutputBudget/2 {
		t.Fatalf("two noisy streams: o=%d e=%d", strings.Count(out, "#"), strings.Count(out, "@"))
	}
	if len(out) > 32<<10 {
		t.Fatalf("result is %d bytes, over the 32 KiB result cap", len(out))
	}
}
