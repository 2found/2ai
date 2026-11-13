package sandbox

import (
	"context"
	"fmt"
	"path"
	"strings"

	gitignore "github.com/sabhiram/go-gitignore"
)

// Apply scoped .gitignore rules on both file substrates. Rules are cached per
// directory for this listing only; a later search observes edited ignore files.
func filterIgnored(ctx context.Context, filesystem workspaceFS, files []string, root string) ([]string, error) {
	root = strings.Trim(strings.TrimSpace(root), "/")
	root = path.Clean(root)
	if root == "." {
		root = ""
	}
	type rule struct {
		match  *gitignore.GitIgnore
		negate bool
	}
	cache := map[string][]rule{}
	// The listing already tells us which directories contain ignore files.
	// Avoid a stat/exec for every absent .gitignore in large sandbox trees.
	present := map[string]bool{}
	for _, file := range files {
		if path.Base(file) == ".gitignore" {
			present[path.Dir(file)] = true
		}
	}
	load := func(dir string) ([]rule, error) {
		if rules, ok := cache[dir]; ok {
			return rules, nil
		}
		name := path.Join(dir, ".gitignore")
		key := dir
		if key == "" {
			key = "."
		}
		if !present[key] {
			// A search rooted below the workspace may omit its parent ignore
			// files. Only those ancestors require a discovery stat.
			ancestor := root != "" && (dir == "" || strings.HasPrefix(root, dir+"/"))
			if !ancestor {
				cache[dir] = nil
				return nil, nil
			}
		}
		info, err := filesystem.Stat(ctx, name)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			cache[dir] = nil
			return nil, nil
		}
		if info.Size > 64*1024 {
			return nil, fmt.Errorf("%s exceeds 64KB ignore-rule cap; narrow the workspace", name)
		}
		data, err := filesystem.ReadFile(ctx, name)
		if err != nil {
			return nil, err
		}
		var rules []rule
		for _, line := range strings.Split(string(data), "\n") {
			negative := strings.HasPrefix(line, "!")
			if negative {
				line = line[1:]
			}
			rules = append(rules, rule{gitignore.CompileIgnoreLines(line), negative})
		}
		cache[dir] = rules
		return rules, nil
	}
	out := make([]string, 0, len(files))
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		parts := strings.Split(file, "/")
		ignored := false
		for depth := 0; depth < len(parts); depth++ {
			dir := strings.Join(parts[:depth], "/")
			rules, err := load(dir)
			if err != nil {
				return nil, err
			}
			for _, r := range rules {
				if r.match.MatchesPath(strings.Join(parts[depth:], "/")) {
					ignored = !r.negate
				}
			}
			// Git cannot re-include a child of an excluded directory. Test parent
			// directories separately before loading that directory's own rules.
			if depth < len(parts)-1 {
				parentIgnored := false
				for ancestor := 0; ancestor <= depth; ancestor++ {
					for _, r := range cache[strings.Join(parts[:ancestor], "/")] {
						if r.match.MatchesPath(strings.Join(parts[ancestor:depth+1], "/") + "/") {
							parentIgnored = !r.negate
						}
					}
				}
				if parentIgnored {
					ignored = true
					break
				}
			}
		}
		if !ignored {
			out = append(out, file)
		}
	}
	return out, nil
}
