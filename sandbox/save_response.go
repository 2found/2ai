package sandbox

import (
	"context"
	"fmt"
	"strings"
)

// Saving a response to the workspace is what makes the network tools part of the
// same agent as the file and shell tools rather than a separate one that only
// speaks in context windows.
//
// Without it, everything an agent fetches has to come back through the model:
// a 4 MB CSV is truncated, a PDF is unreadable, and a dataset it wanted to
// analyze is a summary of a dataset. With it, http_request and web_fetch put the
// bytes in the same directory read_file, grep, and run_shell already work in, so
// "download this and count the rows" is two ordinary tool calls instead of an
// impossibility.
//
// The saved path goes through the same workspace guard as write_file — relative
// only, no escaping the root, no following a symlink out — so a model that
// asks to save to "../../.ssh/authorized_keys" gets an error, not a foothold.

// responseSink writes tool output into the agent's workspace. A zero value (no
// workspace configured) refuses the save with an explanation rather than
// silently returning the body inline, because an agent that asked for a file and
// got prose has no way to tell that its next step will fail.
type responseSink struct {
	fs workspaceFS
}

// save writes data to rel and returns the one-line receipt to hand the model.
// The receipt names the path the agent must use to read the file back, which is
// the workspace-relative one — never the host path, which the agent cannot use
// and should not learn.
// `truncated` says the body hit the tool's read limit, so what landed on disk is
// a prefix of the document. It has to be in the receipt: an agent told "saved
// 262144 bytes" reads that as the whole file, greps it, and reports a row count
// that is confidently wrong. A partial file is still worth keeping — it is the
// silence about it that turns a limit into a bad answer.
func (s responseSink) save(ctx context.Context, tool, rel string, data []byte, truncated bool) (string, error) {
	rel = strings.TrimSpace(rel)
	if rel == "" {
		return "", fmt.Errorf("%s: save_as must not be empty", tool)
	}
	if s.fs == nil {
		return "", fmt.Errorf("%s: save_as needs the agent workspace, which is not configured", tool)
	}
	if err := s.fs.WriteFile(ctx, rel, data); err != nil {
		return "", fmt.Errorf("%s: save %s: %w", tool, rel, err)
	}
	if truncated {
		return fmt.Sprintf("saved %d bytes to %s — TRUNCATED at this tool's read limit; "+
			"the document continues past what was written, so do not treat this file as complete",
			len(data), rel), nil
	}
	return fmt.Sprintf("saved %d bytes to %s", len(data), rel), nil
}

// available reports whether this sink can actually write. It gates the schema:
// a tool that advertises save_as it cannot perform teaches the model to call it,
// and every such call costs a real request whose body is then refused.
func (s responseSink) available() bool { return s.fs != nil }

// respond turns a fetched body into the tool's answer: inline when no save was
// asked for, a receipt when one was and it worked, and — when the save FAILS —
// the body inline anyway with the failure named.
//
// That last case is the point of routing both tools through here. The request
// has already been sent by the time a save can fail, and a POST that created an
// order cannot be un-sent; answering with only an error tells the model the call
// did nothing, and the obvious next move is to repeat it. Returning the body it
// paid for keeps a failed save from becoming a duplicated side effect.
//
// format renders the tool's own status line; passing it nil body is how a
// successful save reports the status without repeating the bytes it just wrote.
func (s responseSink) respond(ctx context.Context, tool, saveAs string, body []byte, truncated bool, format func(body []byte) string) (string, error) {
	if strings.TrimSpace(saveAs) == "" {
		return format(body), nil
	}
	receipt, err := s.save(ctx, tool, saveAs, body, truncated)
	if err != nil {
		return format(body) + "\nsave_as FAILED: " + err.Error() +
			" — the response above is what the request returned; it has already been sent, so do not repeat it just to retry the save", nil
	}
	return format(nil) + "\n" + receipt, nil
}

// saveAsParam is the shared schema fragment, so the two tools describe the same
// capability in the same words and an agent that has learned one has learned the
// other.
func saveAsParam() map[string]any {
	return map[string]any{
		"type": "string",
		"description": "Optional workspace-relative path to save the response body to " +
			"(e.g. \"data/report.csv\"). When set, the body is written to the agent's " +
			"workspace instead of being returned inline — use this for anything large or " +
			"binary, then read or process it with the file and shell tools.",
	}
}
