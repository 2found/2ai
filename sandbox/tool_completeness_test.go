package sandbox

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/2found/2ai/agentcore"
)

func TestSearchCoveragePaginationAndRoot(t *testing.T) {
	ws := mustWorkspace(t)
	mustWrite(t, ws, "huge.txt", strings.Repeat("filler\n", 310000)+"NEEDLE_AT_END\n")
	mustWrite(t, ws, "many.txt", strings.Repeat("HIT\n", 250))
	for _, sb := range []agentcore.Sandbox{nil, NewHostSandbox()} {
		grep := NewGrepTool(sb, ws)
		out, err := grep.Run(context.Background(), `{"pattern":"NEEDLE_AT_END","path":"."}`)
		if err != nil || !strings.Contains(out, "coverage incomplete") || !strings.Contains(out, "huge.txt") || !strings.Contains(out, "byte_offset") {
			t.Fatal("incomplete search was hidden", out, err)
		}
		first, err := grep.Run(context.Background(), `{"pattern":"HIT","glob":"many.txt","limit":500}`)
		if err != nil || strings.Count(first, "many.txt:") != 200 || !strings.Contains(first, "offset=200") || strings.Contains(first, "raise limit") {
			t.Fatal("invalid first page", err)
		}
		second, err := grep.Run(context.Background(), `{"pattern":"HIT","glob":"many.txt","offset":200}`)
		if err != nil || strings.Count(second, "many.txt:") != 50 || !strings.Contains(second, "many.txt:201:") || strings.Contains(second, "truncated at") {
			t.Fatal("invalid continuation", second, err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := grep.Run(ctx, `{"pattern":"HIT"}`); err == nil {
			t.Fatal("search ignored cancellation")
		}
	}
}

func TestSearchNestedIgnoreRulesAndNegations(t *testing.T) {
	ws := mustWorkspace(t)
	for name, data := range map[string]string{".gitignore": "*.log\n!keep.log\nexcluded/\n", "hidden.log": "NEEDLE", "keep.log": "NEEDLE", "nested/.gitignore": "!local.log\n", "nested/local.log": "NEEDLE", "nested/hidden.log": "NEEDLE", "excluded/.gitignore": "!restore.txt\n", "excluded/restore.txt": "NEEDLE"} {
		mustWrite(t, ws, name, data)
	}
	for _, sb := range []agentcore.Sandbox{nil, NewHostSandbox()} {
		out, err := NewGrepTool(sb, ws).Run(context.Background(), `{"pattern":"NEEDLE"}`)
		if err != nil || !strings.Contains(out, "keep.log:1:") || !strings.Contains(out, "nested/local.log:1:") || strings.Contains(out, "hidden.log:") || strings.Contains(out, "restore.txt:") {
			t.Fatal("ignore rules differ from expected coverage", out, err)
		}
		scoped, err := NewGrepTool(sb, ws).Run(context.Background(), `{"pattern":"NEEDLE","path":"nested"}`)
		if err != nil || strings.Contains(scoped, "hidden.log:") || !strings.Contains(scoped, "local.log:") {
			t.Fatal("scoped search ignored ancestor rules", scoped, err)
		}
	}
}

func TestReadBytePagingAndPartialOverwriteGuard(t *testing.T) {
	for _, sb := range []agentcore.Sandbox{nil, NewHostSandbox()} {
		ws := mustWorkspace(t)
		full := strings.Repeat("x", 80*1024) + "SUFFIX_EVIDENCE"
		mustWrite(t, ws, "wide.txt", full)
		read := NewReadFileTool(sb, ws)
		write := NewWriteFileTool(sb, ws)
		first, err := read.Run(context.Background(), `{"path":"wide.txt"}`)
		if err != nil || !strings.Contains(first, "truncated: true") || !strings.Contains(first, "byte_offset=65536") {
			t.Fatal("oversized line was not recoverable", err)
		}
		second, err := read.Run(context.Background(), `{"path":"wide.txt","byte_offset":65536}`)
		if err != nil || !strings.Contains(second, "SUFFIX_EVIDENCE") {
			t.Fatal("suffix missing", err)
		}
		if _, err := write.Run(context.Background(), `{"path":"wide.txt","content":"partial projection"}`); err == nil || mustRead(t, ws, "wide.txt") != full {
			t.Fatal("partial read allowed destructive overwrite", err)
		}
		args := fmt.Sprintf(`{"path":"wide.txt","content":"intentional replacement","allow_partial_overwrite":true,"expected_hash":%q}`, fileContentHash([]byte(full)))
		if _, err := write.Run(context.Background(), args); err != nil {
			t.Fatal("explicit guarded replacement failed", err)
		}
		mustWrite(t, ws, "unicode.txt", "ếếTAIL")
		page, err := read.Run(context.Background(), `{"path":"unicode.txt","byte_offset":1,"byte_limit":1}`)
		if err != nil || !strings.Contains(page, "next_byte_offset: 3") || !strings.Contains(page, "content:\nế") {
			t.Fatal("UTF-8 paging failed to advance", page, err)
		}
	}
	ws := mustWorkspace(t)
	prefix := strings.Repeat("q", 17*1024*1024)
	mustWrite(t, ws, "large.txt", prefix+"TAIL")
	out, err := NewReadFileTool(nil, ws).Run(context.Background(), fmt.Sprintf(`{"path":"large.txt","byte_offset":%d,"byte_limit":4}`, len(prefix)))
	if err != nil || !strings.Contains(out, "TAIL") {
		t.Fatal("large file could not be paged", err)
	}
}
