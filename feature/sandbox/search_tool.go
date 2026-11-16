package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/2found/2ai/core/agentcore"
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
// (Claude Code's Grep / pi's grep). Two engines return the same rows in the
// same order: ripgrep, when the tool runs on the host and an rg binary is
// found (fast on large trees — what makes a code checkout searchable), and
// pure Go RE2 otherwise — a guarded host walk, or a batched listing + read
// inside the sandbox when one is provided. The pattern is always compiled as
// RE2 first, so an invalid pattern fails the same way on both, and any ripgrep
// failure falls back to the Go engine rather than failing the call.
type GrepTool struct {
	workspace *Workspace
	fs        workspaceFS
	// rg is the ripgrep binary the host engine uses; empty uses the Go engine.
	rg string
}

// GrepOption configures a GrepTool.
type GrepOption func(*GrepTool)

// WithGrepRipgrep sets the ripgrep binary for host searches. An empty path
// disables ripgrep, so every search uses the Go engine. Without this option a
// host-substrate grep looks rg up on PATH. Ignored with a sandbox: the search
// then runs inside it.
func WithGrepRipgrep(path string) GrepOption {
	return func(t *GrepTool) {
		if t.rg != "-" {
			t.rg = path
		}
	}
}

// NewGrepTool builds grep over the given sandbox. sb is optional: nil walks and
// reads the host filesystem under the Workspace guards (the default), non-nil
// lists and reads inside the sandbox — batched into one exec per call, never
// one per file.
func NewGrepTool(sb agentcore.Sandbox, workspace *Workspace, options ...GrepOption) *GrepTool {
	t := &GrepTool{workspace: workspace, fs: newWorkspaceFS(sb, workspace)}
	if sb != nil {
		t.rg = "-" // the sandbox substrate never execs on the host
	} else if path, err := exec.LookPath("rg"); err == nil {
		t.rg = path
	}
	for _, option := range options {
		option(t)
	}
	if t.rg == "-" {
		t.rg = ""
	}
	return t
}

func (t *GrepTool) Name() string   { return ToolGrep }
func (t *GrepTool) Parallel() bool { return true }

// Output modes, as Claude Code's Grep names them.
const (
	grepModeContent = "content"
	grepModeFiles   = "files_with_matches"
	grepModeCount   = "count"
)

func (t *GrepTool) Schema() agentcore.ToolSchema {
	return agentcore.ToolSchema{
		Name:   ToolGrep,
		Strict: agentcore.ToolStrictEnabled,
		Description: "Search file contents in the agent workspace by regular expression (RE2 syntax). " +
			"output_mode content (default) returns matching lines as path:line:text; files_with_matches returns only the paths; count returns path:matches. " +
			"Results are capped at " + fmt.Sprint(maxGrepMatches) + " rows; use offset to continue a page. " +
			"Honors nested .gitignore files and excludes dependency/VCS directories. Files that could not be searched (for example above 2MB) are listed; read_file byte paging can inspect them. " +
			"Use glob to narrow which files are searched, path to scope to a subdirectory, and context " +
			"to see surrounding lines without a follow-up read_file. Prefer this tool over grep or rg in the shell.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "RE2 regular expression to match against each line (or a plain string with literal)."},
				"path":    map[string]any{"type": "string", "description": "Optional workspace-relative subdirectory to search. Defaults to the whole workspace."},
				"glob":    map[string]any{"type": "string", "description": "Optional filename filter, e.g. *.go or **/*.ts. Matches the workspace-relative path."},
				"output_mode": map[string]any{
					"type":        "string",
					"enum":        []string{grepModeContent, grepModeFiles, grepModeCount},
					"description": "content (default): matching lines; files_with_matches: one path per matching file; count: path:number of matching lines.",
				},
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
					"description": "content mode: lines of context before and after each match (like grep -C). Defaults to 0, max " + fmt.Sprint(maxGrepContext) + ".",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Maximum rows to return. Defaults to " + fmt.Sprint(maxGrepMatches) + " (also the hard cap).",
				},
				"offset": map[string]any{"type": "integer", "minimum": 0, "maximum": 1000000, "description": "Number of rows (matching lines, or files in the other modes) to skip; continuation notices supply the next offset."},
			},
			"required": []string{"pattern"},
		},
	}
}

// grepQuery is one validated search.
type grepQuery struct {
	pattern  string // as given; ripgrep compiles it itself
	re       *regexp.Regexp
	glob     *regexp.Regexp
	path     string
	mode     string
	literal  bool
	fold     bool
	ctxLines int
	limit    int
	offset   int
}

func (t *GrepTool) Run(ctx context.Context, args string) (string, error) {
	var in struct {
		Pattern         string `json:"pattern"`
		Path            string `json:"path"`
		Glob            string `json:"glob"`
		OutputMode      string `json:"output_mode"`
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
	q := grepQuery{pattern: in.Pattern, path: in.Path, literal: in.Literal, fold: in.CaseInsensitive, offset: in.Offset}
	switch q.mode = strings.TrimSpace(in.OutputMode); q.mode {
	case "":
		q.mode = grepModeContent
	case grepModeContent, grepModeFiles, grepModeCount:
	default:
		return "", fmt.Errorf("grep: output_mode must be content, files_with_matches or count")
	}
	expr := in.Pattern
	if in.Literal {
		expr = regexp.QuoteMeta(expr)
	}
	if in.CaseInsensitive {
		expr = "(?i)" + expr
	}
	q.limit = in.Limit
	if q.limit <= 0 || q.limit > maxGrepMatches {
		q.limit = maxGrepMatches
	}
	q.ctxLines = min(max(in.Context, 0), maxGrepContext)
	if q.mode != grepModeContent {
		q.ctxLines = 0
	}
	var err error
	if q.re, err = regexp.Compile(expr); err != nil {
		return "", fmt.Errorf("grep: invalid pattern: %w", err)
	}
	if g := strings.TrimSpace(in.Glob); g != "" {
		if q.glob, err = compileGlob(g); err != nil {
			return "", fmt.Errorf("grep: invalid glob: %w", err)
		}
	}
	if t.rg != "" {
		if res, ok := t.ripgrep(ctx, q); ok {
			return res.render(), nil
		}
		if err := ctx.Err(); err != nil {
			return "", fmt.Errorf("grep: %w", err)
		}
	}
	res, err := t.goGrep(ctx, q)
	if err != nil {
		return "", fmt.Errorf("grep: %w", err)
	}
	return res.render(), nil
}

// goGrep is the pure Go engine over the tool's file substrate.
func (t *GrepTool) goGrep(ctx context.Context, q grepQuery) (*grepResult, error) {
	// List first, filter by glob, then read what survives. The glob filter has to
	// happen before the read so the sandbox substrate only ships the files that
	// can actually match, instead of the whole tree.
	all, err := t.fs.List(ctx, q.path)
	if err != nil {
		return nil, err
	}
	candidates := all
	if q.glob != nil {
		candidates = candidates[:0:0]
		for _, rel := range all {
			if q.glob.MatchString(rel) {
				candidates = append(candidates, rel)
			}
		}
	}
	res := newGrepResult(q)
	readErr := t.fs.ReadEach(ctx, candidates, maxGrepFileBytes, func(rel string, data []byte) error {
		if data == nil {
			res.skip(rel)
			return nil
		}
		if !utf8.Valid(data) {
			return nil // binary
		}
		lines := strings.Split(string(data), "\n")
		if n := len(lines); n > 1 && lines[n-1] == "" {
			lines = lines[:n-1] // the newline ending the last line opens no line
		}
		var hits []int
		for i, line := range lines {
			if i%1024 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if q.re.MatchString(line) {
				hits = append(hits, i)
			}
		}
		if res.addFile(rel, hits, func(i int) (string, bool) {
			if i < 0 || i >= len(lines) {
				return "", false
			}
			return lines[i], true
		}) {
			return errStopRead
		}
		return nil
	})
	if readErr != nil && !errors.Is(readErr, errStopRead) {
		return nil, readErr
	}
	return res, nil
}

// grepResult accumulates rows across files, applying offset and limit to
// matching lines (content mode) or to files (the other modes).
type grepResult struct {
	q         grepQuery
	out       []string
	seen      int // rows passed, including those skipped by offset
	count     int // rows emitted
	truncated bool
	skipped   []string
}

func newGrepResult(q grepQuery) *grepResult { return &grepResult{q: q} }

// room is how many more hits of one file the page can use: those still to be
// skipped by offset, those still to be emitted, and one that proves the page
// was truncated. Outside content mode a file is one row, so one hit is enough.
func (r *grepResult) room() int {
	if r.q.mode != grepModeContent {
		return 1
	}
	return max(r.q.offset-r.seen, 0) + r.q.limit - r.count + 1
}

func (r *grepResult) skip(rel string) {
	if len(r.skipped) < 20 {
		r.skipped = append(r.skipped, rel)
	}
}

// addFile takes one file's matching line indexes (0-based, ascending) and a
// reader for its lines; it reports true once the limit is reached.
func (r *grepResult) addFile(rel string, hits []int, line func(int) (string, bool)) bool {
	if len(hits) == 0 {
		return false
	}
	if r.q.mode != grepModeContent {
		r.seen++
		if r.seen <= r.q.offset {
			return false
		}
		if r.count == r.q.limit {
			r.truncated = true
			return true
		}
		r.count++
		if r.q.mode == grepModeFiles {
			r.out = append(r.out, rel)
		} else {
			r.out = append(r.out, fmt.Sprintf("%s:%d", rel, len(hits)))
		}
		return false
	}
	var kept []int
	for _, h := range hits {
		r.seen++
		if r.seen <= r.q.offset {
			continue
		}
		if r.count+len(kept) == r.q.limit {
			r.truncated = true
			break
		}
		kept = append(kept, h)
	}
	r.count += len(kept)
	appendGrepHits(&r.out, rel, kept, r.q.ctxLines, line)
	return r.truncated
}

func (r *grepResult) render() string {
	joined := strings.Join(r.out, "\n")
	if r.count == 0 {
		joined = "no matches"
	}
	if r.truncated {
		unit := "matches"
		if r.q.mode != grepModeContent {
			unit = "files"
		}
		joined += fmt.Sprintf("\n…[truncated at %d %s — use offset=%d to continue, or narrow path/glob; maximum limit is %d]", r.q.limit, unit, r.q.offset+r.count, maxGrepMatches)
	}
	if len(r.skipped) > 0 {
		joined += "\n[Search coverage incomplete: unreadable files or files above 2MB were skipped (up to 20 shown): " + strings.Join(r.skipped, ", ") + ". Use read_file byte_offset/byte_limit to inspect large files.]"
	}
	for _, line := range r.out {
		if strings.HasSuffix(line, "…") {
			joined += "\n[Long matching lines were shortened; use read_file byte paging for full text.]"
			break
		}
	}
	return joined
}

// appendGrepHits renders one file's accepted matches. Without context each hit
// is a path:line:text row. With context, overlapping windows are merged into
// groups (grep -C style): matched lines keep the ':' separators, context lines
// use '-', and groups are divided by a "--" row. line reads a 0-based line and
// reports false past the end of the file.
func appendGrepHits(out *[]string, rel string, hits []int, ctxLines int, line func(int) (string, bool)) {
	if ctxLines == 0 {
		for _, h := range hits {
			text, _ := line(h)
			*out = append(*out, fmt.Sprintf("%s:%d:%s", rel, h+1, clampLine(text)))
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
		s, e := max(0, h-ctxLines), h+ctxLines
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
			text, ok := line(i)
			if !ok {
				if i > sp.start && !isHit[i] {
					break
				}
				continue
			}
			sep := "-"
			if isHit[i] {
				sep = ":"
			}
			*out = append(*out, fmt.Sprintf("%s%s%d%s%s", rel, sep, i+1, sep, clampLine(text)))
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
