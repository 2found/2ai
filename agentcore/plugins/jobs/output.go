package jobs

import (
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	// liveHead and liveTail bound the live output one job retains: its first
	// bytes and its newest bytes, with the middle counted and dropped, so a job
	// that prints gigabytes costs about ten kilobytes of memory.
	liveHead = 2 * 1024
	liveTail = 8 * 1024
	// defaultStatusTail is how much live output job_status returns unless asked
	// for more; keepOutput bounds what a finished record carries (and so the
	// checkpoint).
	defaultStatusTail = 2 * 1024
	waitTail          = 1024
	keepOutput        = 4 * 1024
	// lastLineBytes bounds the output excerpt job_list shows per running job.
	lastLineBytes = 100
)

// outputBuffer is a job's live output: bounded, safe for concurrent writers.
type outputBuffer struct {
	mu         sync.Mutex
	head, tail []byte
	total      int64
}

func (o *outputBuffer) write(s string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.total += int64(len(s))
	p := []byte(s)
	if n := min(liveHead-len(o.head), len(p)); n > 0 {
		o.head = append(o.head, p[:n]...)
		p = p[n:]
	}
	if len(p) == 0 {
		return
	}
	o.tail = append(o.tail, p...)
	if len(o.tail) > 2*liveTail { // compact: amortized O(1) per byte
		o.tail = append([]byte(nil), o.tail[len(o.tail)-liveTail:]...)
	}
}

// size is how many bytes the job has published in all.
func (o *outputBuffer) size() int64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.total
}

// text renders the retained output (head, an omission marker, tail), clipped to
// its last n bytes, never splitting a UTF-8 sequence.
func (o *outputBuffer) text(n int) string {
	o.mu.Lock()
	defer o.mu.Unlock()
	tail := o.tail
	if len(tail) > liveTail {
		tail = tail[len(tail)-liveTail:]
	}
	omitted := o.total - int64(len(o.head)) - int64(len(tail))
	var full string
	if omitted == 0 {
		full = string(o.head) + string(tail)
	} else {
		for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
			tail, omitted = tail[1:], omitted+1
		}
		head := o.head
		if cut := partialRune(head); cut > 0 {
			head, omitted = head[:len(head)-cut], omitted+int64(cut)
		}
		full = fmt.Sprintf("%s\n[... %d bytes omitted ...]\n%s", head, omitted, tail)
	}
	if n > 0 && len(full) > n {
		full = full[len(full)-n:]
		for len(full) > 0 && !utf8.RuneStart(full[0]) {
			full = full[1:]
		}
	}
	return full
}

// lastLine is the newest non-empty line of output, clipped for a listing.
func (o *outputBuffer) lastLine() string {
	o.mu.Lock()
	src := o.tail
	if len(src) == 0 {
		src = o.head
	}
	if len(src) > 4*lastLineBytes {
		src = src[len(src)-4*lastLineBytes:]
	}
	s := string(src)
	o.mu.Unlock()
	lines := strings.Split(strings.ReplaceAll(s, "\r", "\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.ToValidUTF8(strings.TrimSpace(lines[i]), "")
		if line == "" {
			continue
		}
		if len(line) > lastLineBytes {
			line = line[:lastLineBytes]
			line = strings.ToValidUTF8(line, "") + "…"
		}
		return line
	}
	return ""
}

// partialRune is the length of an incomplete UTF-8 sequence ending p.
func partialRune(p []byte) int {
	for i := 1; i <= utf8.UTFMax && i <= len(p); i++ {
		if utf8.RuneStart(p[len(p)-i]) {
			if utf8.FullRune(p[len(p)-i:]) {
				return 0
			}
			return i
		}
	}
	return 0
}
