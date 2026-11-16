package sandbox

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// ripgrep runs the query with the rg binary over the host workspace and feeds
// its per-file results through the same grepResult the Go engine uses, so the
// rows, paging and notices are produced by one renderer. It reports false when
// rg cannot serve the query (missing binary, a pattern rg's regex dialect
// rejects, a scope path that is or crosses a symlink, a crash); the caller then
// runs the Go engine, so a ripgrep problem never fails a search.
//
// The flags make rg walk what the Go engine walks: hidden files included,
// nested .gitignore files honored without requiring a git repository, global
// and parent ignore files and rg config ignored, the dependency/VCS directories
// pruned, symlinks not followed, results sorted by path so offset paging is
// stable. rg streams large files, so it has no 2 MB per-file bound.
func (t *GrepTool) ripgrep(ctx context.Context, q grepQuery) (*grepResult, bool) {
	if t.workspace == nil || t.workspace.Root() == "" {
		return nil, false
	}
	root := t.workspace.Root()
	target := "."
	if sub := strings.TrimSpace(q.path); sub != "" && filepath.Clean(sub) != "." {
		abs, clean, err := t.workspace.Resolve(sub)
		if err != nil {
			return nil, false
		}
		// rg follows a symlink named on its command line. The Go engine reaches
		// the scope through os.Root, which refuses one that leaves the workspace;
		// rather than re-deriving that rule, any symlink in the scope path goes
		// to the Go engine.
		realRoot, err := filepath.EvalSymlinks(root)
		if err != nil {
			return nil, false
		}
		realAbs, err := filepath.EvalSymlinks(abs)
		if err != nil || realAbs != filepath.Join(realRoot, clean) {
			return nil, false
		}
		target = filepath.ToSlash(clean)
		// The Go engine applies the .gitignore of every directory from the
		// workspace root down to the scope; rg reads ancestors' rules only by
		// climbing past the workspace root too. A scope under a rule-bearing
		// ancestor goes to the Go engine, so the coverage stays the same.
		for dir := filepath.Dir(clean); ; dir = filepath.Dir(dir) {
			if _, err := os.Lstat(filepath.Join(root, dir, ".gitignore")); err == nil {
				return nil, false
			}
			if dir == "." || dir == "/" {
				break
			}
		}
	}

	expr := q.pattern
	args := []string{
		"--json", "--no-config", "--sort", "path",
		"--hidden", "--no-require-git", "--no-ignore-global", "--no-ignore-parent",
		"--no-ignore-exclude", "--no-ignore-dot", "--no-messages",
	}
	for dir := range skipDir {
		args = append(args, "--glob", "!"+dir+"/")
	}
	if q.literal {
		args = append(args, "--fixed-strings")
	}
	if q.fold {
		args = append(args, "--ignore-case")
	}
	if q.ctxLines > 0 {
		args = append(args, "--context", strconv.Itoa(q.ctxLines))
	}
	args = append(args, "--regexp", expr, "--", target)

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(runCtx, t.rg, args...)
	cmd.Dir = root
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "LC_ALL=C.UTF-8"}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, false
	}
	if err := cmd.Start(); err != nil {
		return nil, false
	}

	res := newGrepResult(q)
	stopped := false
	var file rgFile
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	for sc.Scan() {
		var msg rgMessage
		if json.Unmarshal(sc.Bytes(), &msg) != nil {
			continue
		}
		switch msg.Type {
		case "begin":
			// rg prints paths as reached from its argument: "./x" under ".".
			path := strings.TrimPrefix(msg.Data.Path.Text, "./")
			file = rgFile{path: path, ok: path != ""}
		case "match", "context":
			if !file.ok {
				continue
			}
			if msg.Data.Lines.Text == nil {
				file.ok = false // a line that is not UTF-8: the Go engine skips the file
				continue
			}
			n := msg.Data.LineNumber - 1
			if msg.Type == "match" {
				file.matched++
			}
			// Hold no more of a file than the page can use: the hits that can
			// still be emitted plus one that proves truncation, and the lines
			// around them. count mode keeps only the number.
			full := len(file.hits) >= res.room()
			if msg.Type == "match" && !full && q.mode != grepModeCount {
				file.hits = append(file.hits, n)
			}
			if q.mode == grepModeContent && (!full || (len(file.hits) > 0 && n <= file.hits[len(file.hits)-1]+q.ctxLines)) {
				if file.lines == nil {
					file.lines = map[int]string{}
				}
				file.lines[n] = strings.TrimSuffix(*msg.Data.Lines.Text, "\n")
			}
		case "end":
			if !file.ok || msg.Data.BinaryOffset != nil {
				file = rgFile{}
				continue
			}
			if q.glob != nil && !q.glob.MatchString(file.path) {
				file = rgFile{}
				continue
			}
			lines, hits := file.lines, file.hits
			if q.mode == grepModeCount {
				hits = make([]int, file.matched) // addFile reads only the length
			}
			if res.addFile(file.path, hits, func(i int) (string, bool) {
				text, ok := lines[i]
				return text, ok
			}) {
				stopped = true
			}
			file = rgFile{}
		}
		if stopped {
			cancel()
			break
		}
	}
	scanErr := sc.Err()
	waitErr := cmd.Wait()
	if stopped {
		return res, true
	}
	if ctx.Err() != nil || scanErr != nil {
		return nil, false
	}
	if waitErr != nil {
		// Exit status 1 is "no match"; anything else (2 = error, a signal) is a
		// search rg could not complete.
		if ee, ok := waitErr.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			return nil, false
		}
	}
	return res, true
}

// rgFile is one file's results between rg's begin and end messages.
type rgFile struct {
	path    string
	ok      bool
	matched int
	hits    []int
	lines   map[int]string
}

// rgMessage is the part of an rg --json message the engine reads. Paths and
// lines that are not valid UTF-8 come as base64 "bytes" instead of "text" and
// are skipped.
type rgMessage struct {
	Type string `json:"type"`
	Data struct {
		Path struct {
			Text string `json:"text"`
		} `json:"path"`
		Lines struct {
			Text *string `json:"text"`
		} `json:"lines"`
		LineNumber   int  `json:"line_number"`
		BinaryOffset *int `json:"binary_offset"`
	} `json:"data"`
}
