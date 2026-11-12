package sandbox

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/2found/2ai/agentcore"
)

// headTail keeps the first headMax and the last tailMax bytes written to it,
// counting everything in between, so a command that prints gigabytes costs a
// few kilobytes of memory.
type headTail struct {
	headMax, tailMax int
	head, tail       []byte
	total            int64
}

func (b *headTail) write(p []byte) {
	b.total += int64(len(p))
	if n := min(b.headMax-len(b.head), len(p)); n > 0 {
		b.head = append(b.head, p[:n]...)
		p = p[n:]
	}
	if len(p) == 0 || b.tailMax == 0 {
		return
	}
	b.tail = append(b.tail, p...)
	if len(b.tail) > 2*b.tailMax { // compact: amortized O(1) per byte
		b.tail = append(b.tail[:0:0], b.tail[len(b.tail)-b.tailMax:]...)
	}
}

func (b *headTail) tailBytes() []byte {
	if len(b.tail) > b.tailMax {
		return b.tail[len(b.tail)-b.tailMax:]
	}
	return b.tail
}

// omitted is how many bytes fell between the head and the tail.
func (b *headTail) omitted() int64 {
	return b.total - int64(len(b.head)) - int64(len(b.tailBytes()))
}

// String renders head, an omitted-bytes marker when anything was dropped, and
// tail, never splitting a UTF-8 sequence at either cut.
func (b *headTail) String() string {
	omitted := b.omitted()
	if omitted == 0 {
		return string(b.head) + string(b.tail)
	}
	tail := b.tailBytes()
	for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
		tail, omitted = tail[1:], omitted+1
	}
	head := clampUTF8(string(b.head), len(b.head)-trailingPartialRune(b.head))
	omitted += int64(len(b.head) - len(head))
	return fmt.Sprintf("%s\n[... %d bytes omitted ...]\n%s", head, omitted, tail)
}

// trailingPartialRune is the length of an incomplete UTF-8 sequence at the end
// of p (0 when p ends on a rune boundary).
func trailingPartialRune(p []byte) int {
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

// shellLogMaxBytes bounds one command's full-output log. A chatty command kept
// running in the background (job_keep lets one outlive its run by up to an
// hour) would otherwise fill the disk with a log nobody reads past its tail.
// A variable only so a test can lower it.
var shellLogMaxBytes int64 = 64 << 20

// shellLog is the full-output file in the workspace. With a threshold it is
// created only once that many bytes have arrived (held in memory until then),
// so a short command leaves no file behind; threshold 0 creates it up front.
// It keeps the first shellLogMaxBytes; past them it notes the cut once and
// counts the rest, which the model still sees in the bounded tail.
type shellLog struct {
	ws        *Workspace
	threshold int
	rel       string
	f         *os.File
	pending   []byte
	failed    bool
	written   int64
	dropped   int64
}

func (l *shellLog) open() error {
	root, err := l.ws.openRoot()
	if err != nil {
		return err
	}
	defer root.Close()
	if err := root.MkdirAll(shellLogDir, 0o755); err != nil {
		return err
	}
	rel := shellLogDir + "/shell-" + time.Now().UTC().Format("20060102-150405") + "-" + randHex(4) + ".log"
	f, err := root.OpenFile(filepath.FromSlash(rel), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	l.f, l.rel = f, rel
	return nil
}

func (l *shellLog) write(p []byte) {
	if l == nil || l.failed {
		return
	}
	if l.f == nil {
		l.pending = append(l.pending, p...)
		if len(l.pending) <= l.threshold {
			return
		}
		if err := l.open(); err != nil {
			l.failed, l.pending = true, nil
			return
		}
		p, l.pending = l.pending, nil
	}
	if l.dropped > 0 {
		l.dropped += int64(len(p))
		return
	}
	if room := shellLogMaxBytes - l.written; int64(len(p)) > room {
		l.dropped = int64(len(p)) - room
		p = p[:room]
	}
	n, err := l.f.Write(p)
	l.written += int64(n)
	if err != nil {
		l.failed = true
		return
	}
	if l.dropped > 0 {
		_, _ = fmt.Fprintf(l.f, "\n[... log capped at %d bytes; later output is not saved ...]\n", shellLogMaxBytes)
	}
}

func (l *shellLog) close() {
	if l != nil && l.f != nil {
		_ = l.f.Close()
	}
}

// remove deletes the log file (a short output did not need it).
func (l *shellLog) remove() {
	if l == nil || l.f == nil {
		return
	}
	l.close()
	if root, err := l.ws.openRoot(); err == nil {
		_ = root.Remove(filepath.FromSlash(l.rel))
		_ = root.Close()
	}
	l.f, l.rel = nil, ""
}

// shellOutput collects one command's stdout and stderr: bounded per stream for
// the model, a short combined tail for progress, and everything, in arrival
// order, in the log.
type shellOutput struct {
	mu      sync.Mutex
	streams [2]headTail // stdout, stderr
	recent  headTail    // combined tail, for progress reports
	writes  int
	log     *shellLog
	// background logs are kept even when short: their path was already given
	// to the model.
	keepLog bool
	// live, when set, receives every chunk as it arrives: a background job
	// publishes it as the job's live output.
	live func(string)
}

func newShellOutput(log *shellLog) *shellOutput {
	o := &shellOutput{log: log, recent: headTail{tailMax: shellProgressBytes}}
	o.keepLog = log != nil && log.f != nil
	for i := range o.streams {
		o.streams[i] = headTail{headMax: shellHeadBytes, tailMax: shellTailBytes}
	}
	return o
}

type shellStream struct {
	o *shellOutput
	i int
}

func (w shellStream) Write(p []byte) (int, error) {
	w.o.mu.Lock()
	defer w.o.mu.Unlock()
	w.o.streams[w.i].write(p)
	w.o.recent.write(p)
	w.o.writes++
	w.o.log.write(p)
	if w.o.live != nil {
		w.o.live(string(p))
	}
	return len(p), nil
}

func (o *shellOutput) stream(i int) shellStream { return shellStream{o, i} }

func (o *shellOutput) truncated() bool {
	return o.streams[0].omitted() > 0 || o.streams[1].omitted() > 0
}

// report emits the newest output tail every shellProgressInterval while the
// command runs; the returned stop waits for the reporter to exit, so no emit
// happens after the tool returns.
func (o *shellOutput) report(emit func(string)) (stop func()) {
	if emit == nil {
		return func() {}
	}
	done, exited := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(exited)
		tick := time.NewTicker(shellProgressInterval)
		defer tick.Stop()
		seen := 0
		for {
			select {
			case <-done:
				return
			case <-tick.C:
				o.mu.Lock()
				writes, snapshot := o.writes, o.recent.String()
				o.mu.Unlock()
				if writes != seen {
					seen = writes
					emit(snapshot)
				}
			}
		}
	}()
	return func() { close(done); <-exited }
}

// finish closes the log, deleting it when the visible result already holds
// everything.
func (o *shellOutput) finish() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.truncated() || o.keepLog {
		o.log.close()
		return
	}
	o.log.remove()
}

// format renders the result into a compact, model-readable block.
func (o *shellOutput) format(res agentcore.SandboxResult) string {
	o.mu.Lock()
	defer o.mu.Unlock()
	var b strings.Builder
	if res.Killed {
		fmt.Fprintf(&b, "killed: %s\n", res.KillReason)
	}
	fmt.Fprintf(&b, "exit_code: %d\n", res.ExitCode)
	for i, label := range []string{"stdout", "stderr"} {
		if text := strings.TrimRight(o.streams[i].String(), "\n"); text != "" {
			fmt.Fprintf(&b, "%s:\n%s\n", label, text)
		}
	}
	if o.truncated() {
		switch {
		case o.log != nil && o.log.f != nil && !o.log.failed && o.log.dropped > 0:
			fmt.Fprintf(&b, "[output truncated — the first %d bytes of the output (stdout and stderr interleaved) are saved to %s, the last %d bytes were not saved; the tail above is the end]\n", o.log.written, o.log.rel, o.log.dropped)
		case o.log != nil && o.log.f != nil && !o.log.failed:
			fmt.Fprintf(&b, "[output truncated — the full output (stdout and stderr interleaved) is saved to %s; read it with read_file or grep]\n", o.log.rel)
		default:
			b.WriteString("[output truncated — no workspace log is available; rerun with less output (e.g. pipe through tail or grep)]\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}
