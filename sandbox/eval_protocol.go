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
	"strings"
	"sync"
	"time"

	"github.com/2found/2ai/agentcore"
)

const maxEvalFrameBytes = 1024 * 1024

const (
	maxEvalImagesPerCell = 8
	maxEvalImageBytes    = 768 * 1024
	// A governed rich tool may return up to 768 KiB of decoded images, whose
	// base64 representation is roughly 1 MiB. Leave bounded JSON-envelope room.
	maxEvalBridgeValue = 1536 * 1024
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

type evalCellRequest struct {
	ID   string `json:"id"`
	Code string `json:"code"`
}

type evalCellResult struct {
	Output         string
	Parts          []agentcore.ContentPart
	ExecutionCount int
	RuntimeError   bool
	Truncated      bool
	Invocations    []agentcore.ToolInvocation
	Extra          []agentcore.Message
	Terminate      bool
}

type evalToolResultFrame struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id"`
	OK        bool   `json:"ok"`
	Value     any    `json:"value,omitempty"`
	Error     string `json:"error,omitempty"`
}

type evalBridgeResult struct {
	invocations []agentcore.ToolInvocation
	extra       []agentcore.Message
	terminate   bool
}

type evalProcess struct {
	proc agentcore.SandboxProcess
	in   io.WriteCloser

	writeMu sync.Mutex
	frames  chan evalFrame
	errs    chan error
	done    chan struct{}

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
		errs: make(chan error, 1), done: make(chan struct{}), cleanup: cleanup,
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

func (p *evalProcess) readFrames(r io.Reader) {
	reader := bufio.NewReader(r)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				select {
				case p.errs <- fmt.Errorf("read eval protocol: %w", err):
				default:
				}
			}
			return
		}
		if len(line) > maxEvalFrameBytes {
			select {
			case p.errs <- fmt.Errorf("eval protocol frame exceeds %d bytes", maxEvalFrameBytes):
			default:
			}
			return
		}
		var frame evalFrame
		if err := json.Unmarshal(line, &frame); err != nil {
			select {
			case p.errs <- fmt.Errorf("decode eval protocol: %w", err):
			default:
			}
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
		return evalCellResult{}, withEvalStderr(fmt.Errorf("write eval request: %w", err), p.stderr.String())
	}

	output := newBoundedEvalOutput(limit)
	result := evalCellResult{}
	imageBytes := 0
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
				go p.handleToolCall(ctx, id, frame, slot, maxBridgeCalls, invoker, &bridgeMu, &bridgeResults, &bridgeWG)
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
				text, parts, bytes := renderEvalDisplay(frame.Bundle, maxEvalImagesPerCell-len(result.Parts), maxEvalImageBytes-imageBytes)
				kind := "display"
				if frame.Status == "result" {
					kind = "result"
				}
				output.WriteSection(kind, text)
				result.Parts = append(result.Parts, parts...)
				imageBytes += bytes
			case "done":
				result.ExecutionCount = frame.ExecutionCount
				result.Output, result.Truncated = output.String()
				mergeBridgeResults()
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
		}
		mu.Unlock()
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
		response.Error = "invalid host tool request id"
	} else if slot >= maxBridgeCalls {
		response.Error = fmt.Sprintf("eval host-tool call limit reached (%d per cell)", maxBridgeCalls)
		recordBridgeBlock(response.Error)
	} else if strings.TrimSpace(frame.Name) == "" || len(frame.Name) > 128 {
		response.Error = "invalid host tool name"
		recordBridgeBlock(response.Error)
	} else if invoker == nil {
		response.Error = "host tool bridge is unavailable outside a live agent run"
	} else if err := ctx.Err(); err != nil {
		response.Error = err.Error()
		recordBridgeBlock(response.Error)
	} else {
		arguments := frame.Arguments
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		if !json.Valid(arguments) {
			response.Error = "host tool arguments must be valid JSON"
		} else {
			out, err := invoker.InvokeTool(ctx, frame.Name, string(arguments))
			setResult(out)
			if err != nil {
				response.Error = err.Error()
			} else if value, valueErr := evalBridgeValue(out); valueErr != nil {
				response.Error = valueErr.Error()
			} else {
				response.OK = true
				response.Value = value
			}
		}
	}
	p.writeToolResult(cellID, response)
}

func evalBridgeValue(out agentcore.ToolOutput) (any, error) {
	var value any
	if len(out.Parts) == 0 {
		value = out.Content
	} else {
		images := make([]map[string]string, 0, len(out.Parts))
		for _, part := range out.Parts {
			if part.Type != agentcore.ContentPartImage {
				continue
			}
			images = append(images, map[string]string{"mimeType": part.MIMEType, "data": part.Data})
		}
		value = map[string]any{"text": out.Content, "images": images}
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode host tool result: %w", err)
	}
	if len(encoded) > maxEvalBridgeValue {
		return nil, fmt.Errorf("host tool result exceeds %d bytes", maxEvalBridgeValue)
	}
	return value, nil
}

func (p *evalProcess) writeToolResult(cellID string, response evalToolResultFrame) {
	payload, err := json.Marshal(response)
	if err != nil {
		payload, _ = json.Marshal(evalToolResultFrame{Type: "tool_result", RequestID: response.RequestID, Error: err.Error()})
	}
	// Cell ID is not needed by the runner to correlate a unique request ID, but
	// keeping it in the envelope lets future kernels reject cross-cell replies.
	var envelope map[string]any
	if json.Unmarshal(payload, &envelope) == nil {
		envelope["id"] = cellID
		payload, _ = json.Marshal(envelope)
	}
	p.writeMu.Lock()
	_, _ = p.in.Write(append(payload, '\n'))
	p.writeMu.Unlock()
}

func renderEvalDisplay(bundle map[string]json.RawMessage, imageSlots, imageBudget int) (string, []agentcore.ContentPart, int) {
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

	parts := make([]agentcore.ContentPart, 0, 2)
	used := 0
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
		if imageSlots <= len(parts) {
			appendNotice(fmt.Sprintf("[%s display omitted: per-cell image limit reached]", mime))
			continue
		}
		if imageBudget-used <= 0 {
			appendNotice(fmt.Sprintf("[%s display omitted: cell image budget exhausted]", mime))
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
		if len(decoded) > imageBudget-used {
			appendNotice(fmt.Sprintf("[%s display omitted: image exceeds remaining %d-byte cell budget]", mime, imageBudget-used))
			continue
		}
		parts = append(parts, agentcore.ContentPart{
			Type: agentcore.ContentPartImage, MIMEType: mime, Data: encoded,
		})
		used += len(decoded)
	}
	if len(parts) > 0 {
		note := fmt.Sprintf("[%d rich image display(s) attached]", len(parts))
		if text == "" {
			text = note
		} else {
			text += "\n" + note
		}
	}
	return text, parts, used
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

func (p *evalProcess) closeProcess() {
	p.closeOnce.Do(func() {
		p.writeMu.Lock()
		_, _ = io.WriteString(p.in, "{\"type\":\"exit\"}\n")
		_ = p.in.Close()
		p.writeMu.Unlock()
		select {
		case <-p.done:
			return
		case <-time.After(500 * time.Millisecond):
			_ = p.proc.Kill()
			<-p.done
		}
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
		b.WriteString("[" + kind + "]\n")
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
	if b.total <= b.limit {
		return b.head.String() + string(b.tail), false
	}
	omitted := b.total - b.head.Len() - len(b.tail)
	return fmt.Sprintf("%s\n[... %d bytes omitted ...]\n%s", b.head.String(), omitted, b.tail), true
}

func withEvalStderr(err error, stderr string) error {
	stderr = strings.TrimSpace(stderr)
	if stderr == "" {
		return err
	}
	return fmt.Errorf("%w; kernel stderr: %s", err, stderr)
}
