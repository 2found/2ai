package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/core/agentcore"
)

// maxEvalFrameBytes bounds one protocol line: a display carries one picture
// of up to maxEvalImageBase64 characters plus its text.
const maxEvalFrameBytes = 24 << 20

const (
	// maxEvalImageBytes bounds the pictures of one cell together (decoded
	// bytes): its own displays and the pictures the host tools it calls show.
	// There is no count bound; the provider decides how many pictures a turn
	// may carry. Pictures past it are named in the output, never dropped
	// silently.
	maxEvalImageBytes = 64 << 20
	// maxEvalImageBase64 bounds one displayed picture (base64 characters); the
	// runners enforce the same number.
	maxEvalImageBase64 = 16 << 20
	// maxEvalBridgeValue bounds the text a host tool call returns to the cell
	// (with a note about its pictures, which go to the model instead). A larger
	// result is refused, never cut: a cut JSON document or table would parse as
	// something else.
	maxEvalBridgeValue = 1536 * 1024
	// evalOmittedHint tells the model how to get what a bounded output left out.
	evalOmittedHint = "print a narrower slice, or write the rest to a file"
)

// Error kinds a failed host tool call carries to the cell (ToolError.kind).
const (
	evalKindBlocked     = "blocked"     // policy, a gate or schema validation refused the call
	evalKindFailed      = "failed"      // the tool ran and returned an error
	evalKindBudget      = "budget"      // the run's tool-call budget is spent
	evalKindApproval    = "approval"    // the call needs a human, which a nested call cannot collect
	evalKindLimit       = "limit"       // the cell's call count or result size bound
	evalKindInvalid     = "invalid"     // malformed call: name or arguments
	evalKindUnavailable = "unavailable" // no live agent run, or the kernel is closing
	evalKindAborted     = "aborted"     // the cell was cancelled or timed out
)

type evalFrame struct {
	Type           string                     `json:"type"`
	ID             string                     `json:"id,omitempty"`
	Data           string                     `json:"data,omitempty"`
	Status         string                     `json:"status,omitempty"`
	ExecutionCount int                        `json:"execution_count,omitempty"`
	Bundle         map[string]json.RawMessage `json:"bundle,omitempty"`
	RequestID      string                     `json:"request_id,omitempty"`
	Name           string                     `json:"name,omitempty"`
	Arguments      json.RawMessage            `json:"arguments,omitempty"`
}

var (
	evalFrameTypeRE      = regexp.MustCompile(`"type"\s*:\s*"([a-z_]+)"`)
	evalFrameIDRE        = regexp.MustCompile(`"id"\s*:\s*"([^"\\]*)"`)
	evalFrameRequestIDRE = regexp.MustCompile(`"request_id"\s*:\s*"([^"\\]*)"`)
)

// oversizeFrame stands for a frame that passed maxEvalFrameBytes: only the
// first bytes were kept, enough to say whose it was and what it was.
func oversizeFrame(head []byte, size int) evalFrame {
	frame := evalFrame{Type: "oversize", Data: strconv.Itoa(size)}
	if m := evalFrameTypeRE.FindSubmatch(head); m != nil {
		frame.Name = string(m[1])
	}
	if m := evalFrameIDRE.FindSubmatch(head); m != nil {
		frame.ID = string(m[1])
	}
	if m := evalFrameRequestIDRE.FindSubmatch(head); m != nil {
		frame.RequestID = string(m[1])
	}
	return frame
}

// readEvalLine reads one newline-terminated line without ever holding more
// than limit bytes of it: a longer line is read through and dropped, keeping
// its first bytes in head and its length in size.
func readEvalLine(r *bufio.Reader, limit int) (line, head []byte, size int, over bool, err error) {
	for {
		chunk, readErr := r.ReadSlice('\n')
		if len(head) < 512 && len(chunk) > 0 {
			head = append(head, chunk[:min(len(chunk), 512-len(head))]...)
		}
		size += len(chunk)
		if size > limit {
			over, line = true, nil
		} else {
			line = append(line, chunk...)
		}
		if errors.Is(readErr, bufio.ErrBufferFull) {
			continue
		}
		return line, head, size, over, readErr
	}
}

type evalCellRequest struct {
	ID   string `json:"id"`
	Code string `json:"code"`
}

type evalCellResult struct {
	Output         string
	Parts          []agentcore.ContentPart
	ExecutionCount int
	// Fresh reports that the cell ran in a kernel that had run no cell
	// before: nothing an earlier cell defined exists.
	Fresh        bool
	RuntimeError bool
	Truncated    bool
	Invocations  []agentcore.ToolInvocation
	Extra        []agentcore.Message
	Terminate    bool
	// External reports that a host tool the cell called returned untrusted
	// content: whatever shows the cell's output to the model must frame it.
	External bool
}

type evalToolResultFrame struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id"`
	OK        bool   `json:"ok"`
	Value     string `json:"value"`
	Error     string `json:"error,omitempty"`
	Kind      string `json:"kind,omitempty"`
}

type evalBridgeResult struct {
	invocations []agentcore.ToolInvocation
	extra       []agentcore.Message
	terminate   bool
	external    bool
}

// errEvalNotSent marks a cell the kernel certainly did not receive, so it can
// run on a fresh kernel without running twice.
var errEvalNotSent = errors.New("eval request was not delivered")

type evalProcess struct {
	proc agentcore.SandboxProcess
	in   io.WriteCloser

	// writeMu keeps one line per write. It can be held by a write the kernel is
	// not draining (a busy kernel, a full pipe), so nothing that must finish
	// (closing, killing) takes it.
	writeMu sync.Mutex
	frames  chan evalFrame
	errs    chan error
	done    chan struct{}
	closing chan struct{}

	stderr    cappedStringWriter
	closeOnce sync.Once
	nextID    uint64
	cleanup   func()
}

func newEvalProcess(proc agentcore.SandboxProcess, cleanup func()) *evalProcess {
	if cleanup == nil {
		cleanup = func() {}
	}
	p := &evalProcess{
		proc: proc, in: proc.Stdin(), frames: make(chan evalFrame, 64),
		errs: make(chan error, 1), done: make(chan struct{}), closing: make(chan struct{}), cleanup: cleanup,
	}
	p.stderr.limit = 8 * 1024
	go p.readFrames(proc.Stdout())
	go func() { _, _ = io.Copy(&p.stderr, proc.Stderr()) }()
	go func() {
		_, _ = proc.Wait()
		p.cleanup()
		close(p.done)
	}()
	return p
}

// exited reports that the kernel process has ended.
func (p *evalProcess) exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

func (p *evalProcess) readFrames(r io.Reader) {
	reader := bufio.NewReaderSize(r, 64<<10)
	fail := func(err error) {
		select {
		case p.errs <- err:
		default:
		}
	}
	for {
		line, head, size, over, err := readEvalLine(reader, maxEvalFrameBytes)
		if err != nil {
			// A last line without its newline is not a frame.
			if !errors.Is(err, io.EOF) {
				fail(fmt.Errorf("read eval protocol: %w", err))
			}
			return
		}
		var frame evalFrame
		if over {
			frame = oversizeFrame(head, size)
		} else if err := json.Unmarshal(line, &frame); err != nil {
			fail(fmt.Errorf("decode eval protocol: %w", err))
			return
		}
		select {
		case p.frames <- frame:
		case <-p.done:
			return
		}
	}
}

func (p *evalProcess) execute(ctx context.Context, code string, limit, maxBridgeCalls int, invoker agentcore.ToolInvoker) (evalCellResult, error) {
	if p.exited() {
		return evalCellResult{}, withEvalStderr(fmt.Errorf("%w: %w", errEvalNotSent, errEvalKernelClosed), p.stderr.String())
	}
	p.nextID++
	id := fmt.Sprintf("cell-%d", p.nextID)
	payload, err := json.Marshal(evalCellRequest{ID: id, Code: code})
	if err != nil {
		return evalCellResult{}, err
	}
	p.writeMu.Lock()
	_, err = p.in.Write(append(payload, '\n'))
	p.writeMu.Unlock()
	if err != nil {
		return evalCellResult{}, withEvalStderr(fmt.Errorf("write eval request: %w: %w", errEvalNotSent, err), p.stderr.String())
	}

	output := newBoundedEvalOutput(limit)
	result := evalCellResult{}
	// pictures holds the cell's pictures (displays and those its host tools
	// show) in the order they arrived, bounded by maxEvalImageBytes together.
	pictures := &evalPictures{budget: maxEvalImageBytes}
	var bridgeMu sync.Mutex
	var bridgeWG sync.WaitGroup
	bridgeResults := make([]evalBridgeResult, 0, maxBridgeCalls)
	bridgeCount := 0
	mergeBridgeResults := func() {
		bridgeMu.Lock()
		defer bridgeMu.Unlock()
		for _, bridged := range bridgeResults {
			result.Invocations = append(result.Invocations, bridged.invocations...)
			result.Extra = append(result.Extra, bridged.extra...)
			result.Terminate = result.Terminate || bridged.terminate
			result.External = result.External || bridged.external
		}
	}
	returnError := func(err error) (evalCellResult, error) {
		settled := make(chan struct{})
		go func() {
			bridgeWG.Wait()
			close(settled)
		}()
		select {
		case <-settled:
		case <-time.After(250 * time.Millisecond):
			// A host tool that ignores cancellation must not defeat the cell
			// deadline. Cooperative calls still get a short settlement window so
			// their traces survive in the outer durable outcome.
		}
		result.Output, result.Truncated = output.String()
		mergeBridgeResults()
		result.Parts = pictures.taken()
		return result, err
	}
	for {
		select {
		case <-ctx.Done():
			return returnError(ctx.Err())
		case err := <-p.errs:
			return returnError(withEvalStderr(err, p.stderr.String()))
		case <-p.done:
			return returnError(withEvalStderr(errEvalKernelClosed, p.stderr.String()))
		case frame := <-p.frames:
			if frame.ID != id {
				continue
			}
			switch frame.Type {
			case "tool_call":
				bridgeCount++
				slot := bridgeCount - 1
				bridgeMu.Lock()
				bridgeResults = append(bridgeResults, evalBridgeResult{})
				bridgeMu.Unlock()
				bridgeWG.Add(1)
				go p.handleToolCall(ctx, id, frame, slot, maxBridgeCalls, invoker, pictures, &bridgeMu, &bridgeResults, &bridgeWG)
			case "oversize":
				if frame.Name == "tool_call" && frame.RequestID != "" {
					// The kernel waits for an answer to a call too large to read.
					bridgeWG.Add(1)
					go func(requestID string) {
						defer bridgeWG.Done()
						p.writeToolResult(id, evalToolResultFrame{Type: "tool_result", RequestID: requestID, Kind: evalKindInvalid,
							Error: fmt.Sprintf("host tool call is %s bytes, over the %d-byte frame bound; pass less data (write it to a file and pass the path)", frame.Data, maxEvalFrameBytes)})
					}(frame.RequestID)
				} else {
					output.WriteSection("stderr", fmt.Sprintf("[a %s of %s bytes was dropped: it passed the %d MiB frame bound]\n", frame.Name, frame.Data, maxEvalFrameBytes>>20))
				}
			case "stdout":
				output.WriteSection("stdout", frame.Data)
			case "stderr":
				output.WriteSection("stderr", frame.Data)
			case "result":
				output.WriteSection("result", frame.Data)
			case "error":
				result.RuntimeError = true
				output.WriteSection("error", frame.Data)
			case "display":
				text := renderEvalDisplay(frame.Bundle, pictures)
				kind := "display"
				if frame.Status == "result" {
					kind = "result"
				}
				output.WriteSection(kind, text)
			case "done":
				result.ExecutionCount = frame.ExecutionCount
				result.Output, result.Truncated = output.String()
				mergeBridgeResults()
				result.Parts = pictures.taken()
				return result, nil
			}
		}
	}
}

func (p *evalProcess) handleToolCall(
	ctx context.Context,
	cellID string,
	frame evalFrame,
	slot, maxBridgeCalls int,
	invoker agentcore.ToolInvoker,
	pictures *evalPictures,
	mu *sync.Mutex,
	results *[]evalBridgeResult,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	response := evalToolResultFrame{Type: "tool_result", RequestID: frame.RequestID}
	setResult := func(out agentcore.ToolOutput) {
		mu.Lock()
		(*results)[slot] = evalBridgeResult{
			invocations: append([]agentcore.ToolInvocation(nil), out.Invocations...),
			extra:       append([]agentcore.Message(nil), out.AdditionalContexts...),
			terminate:   out.Terminate,
			external:    out.External,
		}
		mu.Unlock()
	}
	refuse := func(kind, reason string) {
		response.Kind, response.Error = kind, reason
	}
	recordBridgeBlock := func(reason string) {
		if invoker == nil || strings.TrimSpace(frame.Name) == "" {
			return
		}
		setResult(agentcore.ToolOutput{Invocations: []agentcore.ToolInvocation{{Trace: agentcore.ToolTrace{
			CallID: frame.RequestID, Tool: frame.Name, Args: string(frame.Arguments), Allowed: false, Reason: reason,
		}}}})
	}
	if frame.RequestID == "" || len(frame.RequestID) > 160 {
		refuse(evalKindInvalid, "invalid host tool request id")
	} else if slot >= maxBridgeCalls {
		refuse(evalKindLimit, fmt.Sprintf("eval host-tool call limit reached (%d per cell)", maxBridgeCalls))
		recordBridgeBlock(response.Error)
	} else if strings.TrimSpace(frame.Name) == "" || len(frame.Name) > 128 {
		refuse(evalKindInvalid, "invalid host tool name")
		recordBridgeBlock(response.Error)
	} else if invoker == nil {
		refuse(evalKindUnavailable, "host tool bridge is unavailable outside a live agent run")
	} else if err := ctx.Err(); err != nil {
		refuse(evalKindAborted, err.Error())
		recordBridgeBlock(response.Error)
	} else {
		arguments := frame.Arguments
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		if !json.Valid(arguments) {
			refuse(evalKindInvalid, "host tool arguments must be valid JSON")
		} else {
			out, err := invoker.InvokeTool(ctx, frame.Name, string(arguments))
			setResult(out)
			if err != nil {
				refuse(classifyEvalToolError(err, out), err.Error())
			} else if value, valueErr := evalBridgeValue(out, pictures); valueErr != nil {
				refuse(evalKindLimit, valueErr.Error())
			} else {
				response.OK = true
				response.Value = value
			}
		}
	}
	p.writeToolResult(cellID, response)
}

// classifyEvalToolError names why a nested call failed, for ToolError.kind.
func classifyEvalToolError(err error, out agentcore.ToolOutput) string {
	message := err.Error()
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return evalKindAborted
	case strings.Contains(message, "budget exhausted"):
		return evalKindBudget
	case strings.Contains(message, "needs human approval"), strings.Contains(message, "cannot park"):
		return evalKindApproval
	}
	if len(out.Invocations) > 0 && !out.Invocations[0].Trace.Allowed {
		return evalKindBlocked
	}
	return evalKindFailed
}

// evalBridgeValue is what a host tool call returns to the cell: always the
// tool's text (a string, so a cell can rely on the type), with a note appended
// when the tool showed pictures. The pictures themselves come to the model with
// the cell's result (as a direct call's would), not into the cell's variables,
// so a script that takes a screenshot of each state of a page shows every one
// of them. A text over maxEvalBridgeValue is refused rather than cut.
func evalBridgeValue(out agentcore.ToolOutput, pictures *evalPictures) (string, error) {
	text := out.Content
	if note := pictures.add(out.Parts); note != "" {
		if text != "" {
			text += "\n"
		}
		text += note
	}
	if len(text) > maxEvalBridgeValue {
		return "", fmt.Errorf("host tool result is %d bytes, over the %d-byte limit for one call inside a cell; ask the tool for less (a narrower range, a smaller limit, a filter)", len(text), maxEvalBridgeValue)
	}
	return text, nil
}

// evalPictures collects one cell's pictures, safe for the bridge goroutines
// and the frame loop together.
type evalPictures struct {
	mu     sync.Mutex
	budget int
	used   int
	parts  []agentcore.ContentPart
}

// take keeps one picture when it fits the cell's remaining bytes.
func (p *evalPictures) take(part agentcore.ContentPart, size int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if size > p.budget-p.used {
		return false
	}
	p.used += size
	p.parts = append(p.parts, part)
	return true
}

func (p *evalPictures) remaining() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.budget - p.used
}

func (p *evalPictures) taken() []agentcore.ContentPart {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.parts)
}

// add keeps the image parts of a host tool's result and says what became of
// them, "" when it had none.
func (p *evalPictures) add(parts []agentcore.ContentPart) string {
	shown, unshown := 0, 0
	for _, part := range parts {
		if part.Type != agentcore.ContentPartImage {
			continue
		}
		size := base64.StdEncoding.DecodedLen(len(part.Data))
		if p.take(part, size) {
			shown++
		} else {
			unshown++
		}
	}
	switch {
	case shown == 0 && unshown == 0:
		return ""
	case unshown == 0:
		return fmt.Sprintf("(%d picture(s) shown to you with this eval result)", shown)
	default:
		return fmt.Sprintf("(%d picture(s) shown to you with this eval result; %d more not shown: the pictures of one cell passed %d MB)", shown, unshown, maxEvalImageBytes>>20)
	}
}

func (p *evalProcess) writeToolResult(cellID string, response evalToolResultFrame) {
	payload, err := json.Marshal(response)
	if err != nil {
		payload, _ = json.Marshal(evalToolResultFrame{Type: "tool_result", RequestID: response.RequestID, Error: err.Error(), Kind: evalKindFailed})
	}
	// Cell ID is not needed by the runner to correlate a unique request ID, but
	// keeping it in the envelope lets future kernels reject cross-cell replies.
	var envelope map[string]any
	if json.Unmarshal(payload, &envelope) == nil {
		envelope["id"] = cellID
		payload, _ = json.Marshal(envelope)
	}
	select {
	case <-p.closing:
		return
	default:
	}
	// The write blocks while the kernel does not read (it is busy, or the pipe
	// is full). Closing the process kills it, which ends the write.
	p.writeMu.Lock()
	_, _ = p.in.Write(append(payload, '\n'))
	p.writeMu.Unlock()
}

func renderEvalDisplay(bundle map[string]json.RawMessage, pictures *evalPictures) string {
	var text string
	for _, mime := range []string{"text/markdown", "text/plain", "text/html", "image/svg+xml", "text/latex"} {
		raw, ok := bundle[mime]
		if !ok {
			continue
		}
		if json.Unmarshal(raw, &text) == nil && text != "" {
			if mime == "text/html" {
				text = "[HTML display]\n" + text
			} else if mime == "image/svg+xml" {
				text = "[SVG display]\n" + text
			} else if mime == "text/latex" {
				text = "[LaTeX display]\n" + text
			}
			break
		}
	}
	if raw, ok := bundle["application/json"]; ok {
		var value any
		if json.Unmarshal(raw, &value) == nil {
			if pretty, err := json.MarshalIndent(value, "", "  "); err == nil {
				text = string(pretty)
			}
		}
	}

	attached := 0
	appendNotice := func(notice string) {
		if text != "" {
			text += "\n"
		}
		text += notice
	}
	for _, mime := range []string{"image/png", "image/jpeg"} {
		raw, ok := bundle[mime]
		if !ok {
			continue
		}
		if pictures.remaining() <= 0 {
			appendNotice(fmt.Sprintf("[%s display omitted: the pictures of one cell passed %d MB]", mime, maxEvalImageBytes>>20))
			continue
		}
		var encoded string
		if json.Unmarshal(raw, &encoded) != nil || encoded == "" {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || !validEvalImage(mime, decoded) {
			appendNotice(fmt.Sprintf("[%s display omitted: invalid image data]", mime))
			continue
		}
		if !pictures.take(agentcore.ContentPart{Type: agentcore.ContentPartImage, MIMEType: mime, Data: encoded}, len(decoded)) {
			appendNotice(fmt.Sprintf("[%s display omitted: image exceeds the %d bytes left of the cell's %d MB of pictures]", mime, pictures.remaining(), maxEvalImageBytes>>20))
			continue
		}
		attached++
	}
	if attached > 0 {
		note := fmt.Sprintf("[%d rich image display(s) attached]", attached)
		if text == "" {
			text = note
		} else {
			text += "\n" + note
		}
	}
	return text
}

func validEvalImage(mime string, data []byte) bool {
	switch mime {
	case "image/png":
		return bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n"))
	case "image/jpeg":
		return len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff
	default:
		return false
	}
}

// closeProcess asks the kernel to exit and kills it if it has not within half
// a second. It never waits for writeMu: a write the kernel is not reading would
// hold it forever, and the kill is what ends that write.
func (p *evalProcess) closeProcess() { p.stop(false) }

// terminate kills the kernel at once, for one whose state is unknown (a cell
// timed out or the protocol broke): it is not going to read an exit request.
func (p *evalProcess) terminate() { p.stop(true) }

func (p *evalProcess) stop(kill bool) {
	p.closeOnce.Do(func() {
		close(p.closing)
		if !kill {
			go func() {
				if p.writeMu.TryLock() {
					_, _ = io.WriteString(p.in, "{\"type\":\"exit\"}\n")
					p.writeMu.Unlock()
				}
				_ = p.in.Close()
			}()
			select {
			case <-p.done:
				return
			case <-time.After(500 * time.Millisecond):
			}
		}
		_ = p.proc.Kill()
		_ = p.in.Close()
		<-p.done
	})
}

// close exists as a method value used by evalKernel without exposing the
// protocol process outside this package.
func (p *evalProcess) close() { p.closeProcess() }

type boundedEvalOutput struct {
	limit     int
	total     int
	headLimit int
	tailLimit int
	head      strings.Builder
	tail      []byte
	lastKind  string
	// sections counts the runs of one kind; a lone section needs no header.
	sections int
	// firstHeader is the first section's header, dropped again when it stays the
	// only one: stderr keeps its header, the others carry no information.
	firstHeader string
}

func newBoundedEvalOutput(limit int) *boundedEvalOutput {
	if limit < 1 {
		limit = defaultEvalMaxOutputBytes
	}
	return &boundedEvalOutput{limit: limit, headLimit: limit * 2 / 3, tailLimit: limit - limit*2/3}
}

func (b *boundedEvalOutput) WriteSection(kind, value string) {
	if value == "" {
		return
	}
	if b.lastKind != kind {
		if b.total > 0 {
			b.WriteString("\n")
		}
		header := "[" + kind + "]\n"
		if b.sections == 0 && kind != "stderr" {
			b.firstHeader = header
		}
		b.sections++
		b.WriteString(header)
		b.lastKind = kind
	}
	b.WriteString(value)
}

func (b *boundedEvalOutput) WriteString(value string) {
	if value == "" {
		return
	}
	data := []byte(value)
	b.total += len(data)
	if b.head.Len() < b.headLimit {
		n := b.headLimit - b.head.Len()
		if n > len(data) {
			n = len(data)
		}
		_, _ = b.head.Write(data[:n])
		data = data[n:]
	}
	if len(data) == 0 || b.tailLimit == 0 {
		return
	}
	if len(data) >= b.tailLimit {
		b.tail = append(b.tail[:0], data[len(data)-b.tailLimit:]...)
		return
	}
	if len(b.tail)+len(data) > b.tailLimit {
		drop := len(b.tail) + len(data) - b.tailLimit
		copy(b.tail, b.tail[drop:])
		b.tail = b.tail[:len(b.tail)-drop]
	}
	b.tail = append(b.tail, data...)
}

func (b *boundedEvalOutput) String() (string, bool) {
	head, total := b.head.String(), b.total
	if b.sections == 1 && b.firstHeader != "" && strings.HasPrefix(head, b.firstHeader) {
		head, total = head[len(b.firstHeader):], total-len(b.firstHeader)
	}
	if total <= b.limit {
		return head + string(b.tail), false
	}
	omitted := total - len(head) - len(b.tail)
	return fmt.Sprintf("%s\n[... %d bytes omitted: %s ...]\n%s", head, omitted, evalOmittedHint, b.tail), true
}

func withEvalStderr(err error, stderr string) error {
	stderr = strings.TrimSpace(stderr)
	if stderr == "" {
		return err
	}
	return fmt.Errorf("%w; kernel stderr: %s", err, stderr)
}
