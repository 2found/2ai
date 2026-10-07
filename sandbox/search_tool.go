package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

const (
	ToolGrep = "grep"
	ToolGlob = "glob"
)

const (
	maxGrepMatches    = 200
	maxGlobMatches    = 500
	maxGrepFileBytes  = 2 * 1024 * 1024 // skip files larger than this when searching content
	maxGrepLineLength = 400
	maxGrepContext    = 10 // cap on before/after context lines per match
)

// skipDir lists directories never worth walking for search: VCS metadata and
// dependency caches. Mirrors the practical defaults of ripgrep/Claude Code's
// grep without a full .gitignore parser.
var skipDir = map[string]struct{}{
	".git":         {},
	"node_modules": {},
	".next":        {},
	"dist":         {},
	"vendor":       {},
}

// GrepTool searches file contents in the workspace by regular expression
// (Claude Code's Grep / pi's grep). The matching is always pure Go regexp in
// this process — only the file I/O moves: a guarded host walk by default, or a
// batched listing + read inside the sandbox when one is provided. Keeping RE2
// above the substrate seam is what makes both paths return the same lines in
// the same order. Returns file:line:match lines, capped for token safety.
type GrepTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewGrepTool builds grep over the given sandbox. sb is optional: nil walks and
// reads the host filesystem under the Workspace guards (the default), non-nil
// lists and reads inside the sandbox — batched into one exec per call, never
// one per file.
func NewGrepTool(sb agentcore.Sandbox, workspace *Workspace) *GrepTool {
	return &GrepTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *GrepTool) Name() string   { return ToolGrep }
func (t *GrepTool) Parallel() bool { return true }

func (t *GrepTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolGrep,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Search file contents in the agent workspace by regular expression (Go/RE2 syntax). " +
			"Returns matching lines as path:line:text, capped at " + fmt.Sprint(maxGrepMatches) + " matches. " +
			"Honors nested .gitignore files and excludes dependency/VCS directories. Files above 2MB are reported as skipped; read_file byte paging can inspect them. Use offset to continue a result page. " +
			"Use glob to narrow which files are searched, path to scope to a subdirectory, and context " +
			"to see surrounding lines without a follow-up read_file.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "RE2 regular expression to match against each line (or a plain string with literal)."},
				"path":    map[string]any{"type": "string", "description": "Optional workspace-relative subdirectory to search. Defaults to the whole workspace."},
				"glob":    map[string]any{"type": "string", "description": "Optional filename filter, e.g. *.go or **/*.ts. Matches the workspace-relative path."},
				"case_insensitive": map[string]any{
					"type":        "boolean",
					"description": "Case-insensitive match. Defaults to false.",
				},
				"literal": map[string]any{
					"type":        "boolean",
					"description": "Treat pattern as a literal string instead of a regex. Defaults to false.",
				},
				"context": map[string]any{
					"type":        "integer",
					"description": "Lines of context to show before and after each match (like grep -C). Defaults to 0, max " + fmt.Sprint(maxGrepContext) + ".",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Maximum matches to return. Defaults to " + fmt.Sprint(maxGrepMatches) + " (also the hard cap).",
				},
				"offset": map[string]any{"type": "integer", "minimum": 0, "maximum": 1000000, "description": "Number of matching lines to skip; continuation notices supply the next offset."},
			},
			"required": []string{"pattern"},
		},
	}
}

func (t *GrepTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Pattern         string `json:"pattern"`
		Path            string `json:"path"`
		Glob            string `json:"glob"`
		CaseInsensitive bool   `json:"case_insensitive"`
		Literal         bool   `json:"literal"`
		Context         int    `json:"context"`
		Limit           int    `json:"limit"`
		Offset          int    `json:"offset"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("grep: invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.Pattern) == "" {
		return "", fmt.Errorf("grep: pattern is empty")
	}
	if in.Offset < 0 || in.Offset > 1000000 {
		return "", fmt.Errorf("grep: offset must be between 0 and 1000000")
	}
	expr := in.Pattern
	if in.Literal {
		expr = regexp.QuoteMeta(expr)
	}
	if in.CaseInsensitive {
		expr = "(?i)" + expr
	}
	limit := in.Limit
	if limit <= 0 || limit > maxGrepMatches {
		limit = maxGrepMatches
	}
	ctxLines := in.Context
	if ctxLines < 0 {
		ctxLines = 0
	}
	if ctxLines > maxGrepContext {
		ctxLines = maxGrepContext
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return "", fmt.Errorf("grep: invalid pattern: %w", err)
	}

	var globRe *regexp.Regexp
	if g := strings.TrimSpace(in.Glob); g != "" {
		globRe, err = compileGlob(g)
		if err != nil {
			return "", fmt.Errorf("grep: invalid glob: %w", err)
		}
	}

	// List first, filter by glob, then read what survives. The glob filter has to
	// happen before the read so the sandbox substrate only ships the files that
	// can actually match, instead of the whole tree.
	all, err := t.fs.List(ctx, in.Path)
	if err != nil {
		return "", fmt.Errorf("grep: %w", err)
	}
	candidates := all
	if globRe != nil {
		candidates = candidates[:0:0]
		for _, rel := range all {
			if globRe.MatchString(rel) {
				candidates = append(candidates, rel)
			}
		}
	}

	var out []string
	var skipped []string
	count := 0
	seen := 0
	truncated := false
	readErr := t.fs.ReadEach(ctx, candidates, maxGrepFileBytes, func(rel string, data []byte) error {
		if data == nil {
			if len(skipped) < 20 {
				skipped = append(skipped, rel)
			}
			return nil
		}
		if !utf8.Valid(data) {
			return nil // binary
		}
		lines := strings.Split(string(data), "\n")
		var hits []int
		for i, line := range lines {
			if i%1024 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if re.MatchString(line) {
				seen++
				if seen <= in.Offset {
					continue
				}
				hits = append(hits, i)
				if count+len(hits) > limit {
					truncated = true
					break
				}
			}
		}
		if len(hits) == 0 {
			return nil
		}
		if count+len(hits) > limit {
			hits = hits[:limit-count]
			truncated = true
		}
		count += len(hits)
		appendGrepHits(&out, rel, lines, hits, ctxLines)
		if truncated {
			return errStopRead
		}
		return nil
	})
	if readErr != nil && !errors.Is(readErr, errStopRead) {
		return "", fmt.Errorf("grep: %w", readErr)
	}
	joined := strings.Join(out, "\n")
	if count == 0 {
		joined = "no matches"
	}
	if truncated {
		joined += fmt.Sprintf("\n…[truncated at %d matches — use offset=%d to continue, or narrow path/glob; maximum limit is %d]", limit, in.Offset+count, maxGrepMatches)
	}
	if len(skipped) > 0 {
		joined += "\n[Search coverage incomplete: unreadable files or files above 2MB were skipped (up to 20 shown): " + strings.Join(skipped, ", ") + ". Use read_file byte_offset/byte_limit to inspect large files.]"
	}
	for _, line := range out {
		if strings.HasSuffix(line, "…") {
			joined += "\n[Long matching lines were shortened; use read_file byte paging for full text.]"
			break
		}
	}
	return joined, nil
}

// appendGrepHits renders one file's accepted matches. Without context each hit
// is a path:line:text row. With context, overlapping windows are merged into
// groups (grep -C style): matched lines keep the ':' separators, context lines
// use '-', and groups are divided by a "--" row.
func appendGrepHits(out *[]string, rel string, lines []string, hits []int, ctxLines int) {
	if ctxLines == 0 {
		for _, h := range hits {
			*out = append(*out, fmt.Sprintf("%s:%d:%s", rel, h+1, clampLine(lines[h])))
		}
		return
	}
	isHit := make(map[int]bool, len(hits))
	for _, h := range hits {
		isHit[h] = true
	}
	type span struct{ start, end int }
	var spans []span
	for _, h := range hits {
		s, e := max(0, h-ctxLines), min(len(lines)-1, h+ctxLines)
		if n := len(spans); n > 0 && s <= spans[n-1].end+1 {
			spans[n-1].end = max(spans[n-1].end, e)
			continue
		}
		spans = append(spans, span{s, e})
	}
	for _, sp := range spans {
		if n := len(*out); n > 0 && (*out)[n-1] != "--" {
			*out = append(*out, "--")
		}
		for i := sp.start; i <= sp.end; i++ {
			sep := "-"
			if isHit[i] {
				sep = ":"
			}
			*out = append(*out, fmt.Sprintf("%s%s%d%s%s", rel, sep, i+1, sep, clampLine(lines[i])))
		}
	}
}

// GlobTool lists workspace files whose relative path matches a glob pattern
// (Claude Code's Glob), supporting * ? and ** segments. Results are sorted for
// stable output and capped for token safety.
type GlobTool struct {
	workspace *Workspace
	fs        workspaceFS
}

// NewGlobTool builds glob over the given sandbox. sb is optional: nil walks the
// host filesystem under the Workspace guards (the default), non-nil lists the
// tree inside the sandbox in a single exec.
func NewGlobTool(sb agentcore.Sandbox, workspace *Workspace) *GlobTool {
	return &GlobTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
}

func (t *GlobTool) Name() string   { return ToolGlob }
func (t *GlobTool) Parallel() bool { return true }

func (t *GlobTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolGlob,
		Strict: agentcore.ToolStrictEnabled,
		Description: "List files in the agent workspace, respecting nested .gitignore and dependency exclusions, whose relative path matches a glob pattern " +
			"(supports *, ?, and ** for any depth, e.g. **/*.go or src/**/test_*.ts). " +
			"Returns up to " + fmt.Sprint(maxGlobMatches) + " sorted paths.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "Glob pattern matched against the workspace-relative path."},
				"path":    map[string]any{"type": "string", "description": "Optional workspace-relative subdirectory to search within."},
			},
			"required": []string{"pattern"},
		},
	}
}

func (t *GlobTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", fmt.Errorf("glob: invalid arguments: %w", err)
	}
	if strings.TrimSpace(in.Pattern) == "" {
		return "", fmt.Errorf("glob: pattern is empty")
	}
	globRe, err := compileGlob(in.Pattern)
	if err != nil {
		return "", fmt.Errorf("glob: invalid pattern: %w", err)
	}
	all, err := t.fs.List(ctx, in.Path)
	if err != nil {
		return "", fmt.Errorf("glob: %w", err)
	}

	var hits []string
	truncated := false
	for _, rel := range all {
		if !globRe.MatchString(rel) {
			continue
		}
		if len(hits) >= maxGlobMatches {
			truncated = true
			break
		}
		hits = append(hits, rel)
	}
	if len(hits) == 0 {
		return "no files match", nil
	}
	sort.Strings(hits)
	out := strings.Join(hits, "\n")
	if truncated {
		out += fmt.Sprintf("\n…[truncated at %d files]", maxGlobMatches)
	}
	return out, nil
}

// searchRoot resolves an optional relative subdirectory to an absolute path
// inside the workspace, defaulting to the workspace root. It is the host
// substrate's half of scoping a search; the sandbox substrate scopes with the
// same relative path inside the mount.
func searchRoot(ws *Workspace, sub string) (string, error) {
	if strings.TrimSpace(sub) == "" || filepath.Clean(sub) == "." {
		if ws.Root() == "" {
			return "", fmt.Errorf("workspace is not configured")
		}
		return ws.Root(), nil
	}
	abs, _, err := ws.Resolve(sub)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	if !inside(ws.Root(), resolved) {
		return "", fmt.Errorf("path escapes workspace")
	}
	return resolved, nil
}

func workspaceRel(ws *Workspace, abs string) string {
	rel, err := filepath.Rel(ws.Root(), abs)
	if err != nil {
		return abs
	}
	return filepath.ToSlash(rel)
}

func clampLine(s string) string {
	s = strings.TrimRight(s, "\r")
	if len(s) > maxGrepLineLength {
		return s[:maxGrepLineLength] + "…"
	}
	return s
}

// compileGlob converts a shell-style glob (with *, ?, and ** segments) into an
// anchored RE2 regexp matched against a slash-separated relative path. ** spans
// any number of path segments; * does not cross a slash.
func compileGlob(glob string) (*regexp.Regexp, error) {
	glob = filepath.ToSlash(strings.TrimSpace(glob))
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(glob); i++ {
		c := glob[i]
		switch c {
		case '*':
			if i+1 < len(glob) && glob[i+1] == '*' {
				i++ // consume second *
				// "**/" or "**" — match across directory separators.
				if i+1 < len(glob) && glob[i+1] == '/' {
					i++
					b.WriteString("(?:.*/)?")
				} else {
					b.WriteString(".*")
				}
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}
