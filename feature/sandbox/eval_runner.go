package sandbox

// pythonEvalRunner is intentionally self-contained and standard-library-only.
// Protocol frames use a duplicate of the original stdout descriptor; fd 1 and
// fd 2 are redirected to per-cell drain pipes before user code runs, so a print
// or child process cannot corrupt the NDJSON control stream or fill local disk.
const pythonEvalRunner = `
from __future__ import annotations

import ast
import asyncio
import base64
import builtins
import codecs
import concurrent.futures
import inspect
import io
import json
import os
import queue
import re
import select
import sys
import threading
import time
import traceback

_protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="backslashreplace", buffering=1)
# One thread owns stdin and routes what arrives: cell requests to the main loop,
# host tool answers to the call waiting for them. User code gets an empty stdin,
# so it can neither steal a protocol line nor wait on the terminal.
_stdin = sys.stdin
sys.stdin = io.StringIO("")
_write_lock = threading.Lock()
_out_lock = threading.Lock()
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_execution_count = 0
_result_name = "__agentray_cell_result__"
_current_request_id = None
_displayed_figure_ids = set()
_tool_sequence = 0
_seq_lock = threading.Lock()
_captures = []
_pending = {}
_pending_lock = threading.Lock()
_requests = queue.Queue()
_pool = None
_pool_lock = threading.Lock()
_TEXT_LIMIT = 262144
_IMAGE_LIMIT = 16777216
_TOOL_KINDS = ("blocked", "failed", "budget", "approval", "limit", "invalid", "unavailable", "aborted")

def _emit(value):
    line = json.dumps(value, ensure_ascii=False, default=repr)
    with _write_lock:
        _protocol.write(line + "\n")
        _protocol.flush()

def _emit_text(kind, request_id, text):
    if not text:
        return
    step = 8192
    for start in range(0, len(text), step):
        _emit({"type": kind, "id": request_id, "data": text[start:start + step]})

class _Capture:
    """Redirects one descriptor (1 or 2) into a pipe the cell's output is read
    from. Output frames, display frames and tool calls all leave under
    _out_lock, and a frame that is not output first drains the pipe, so the
    frames keep the order the cell produced them in."""

    def __init__(self, kind, request_id, target_fd):
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.dup2(write_fd, target_fd)
        os.close(write_fd)
        self.kind = kind
        self.id = request_id
        self.fd = read_fd
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.done = threading.Event()
        self.finished = False
        self.abandoned = False
        threading.Thread(target=self._drain, daemon=True).start()

    def pump(self):
        # Called with _out_lock held. True once the pipe is at its end.
        if self.finished:
            return True
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return False
            except OSError:
                data = b""
            if not data:
                _emit_text(self.kind, self.id, self.decoder.decode(b"", True))
                self.finished = True
                os.close(self.fd)
                return True
            _emit_text(self.kind, self.id, self.decoder.decode(data))

    def _drain(self):
        try:
            while not self.abandoned and not self.finished:
                ready, _, _ = select.select([self.fd], [], [], 0.25)
                if not ready:
                    continue
                with _out_lock:
                    if self.pump():
                        break
        except Exception:
            pass
        finally:
            with _out_lock:
                if not self.finished:
                    self.finished = True
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
            self.done.set()

def _emit_ordered(frame):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    with _out_lock:
        for capture in list(_captures):
            capture.pump()
        _emit(frame)

_REPR_MIMES = [
    ("_repr_html_", "text/html"),
    ("_repr_markdown_", "text/markdown"),
    ("_repr_svg_", "image/svg+xml"),
    ("_repr_png_", "image/png"),
    ("_repr_jpeg_", "image/jpeg"),
    ("_repr_json_", "application/json"),
    ("_repr_latex_", "text/latex"),
]

def _coerce_image(value):
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return str(value)

def _matplotlib_png(value):
    value_type = type(value)
    if value_type.__module__ != "matplotlib.figure" or value_type.__name__ != "Figure":
        return None
    savefig = getattr(value, "savefig", None)
    if not callable(savefig):
        return None
    try:
        buf = io.BytesIO()
        savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

def _mime_bundle(value, json_value=False):
    bundle = {}
    if json_value and (value is None or isinstance(value, (dict, list, tuple, str, int, float, bool))):
        bundle["application/json"] = value
    png = _matplotlib_png(value)
    if png is not None:
        bundle["image/png"] = png
    mimebundle = getattr(value, "_repr_mimebundle_", None)
    if callable(mimebundle):
        try:
            data = mimebundle()
            if isinstance(data, tuple):
                data = data[0]
            if isinstance(data, dict):
                bundle.update({str(k): v for k, v in data.items()})
        except Exception:
            pass
    for attr, mime in _REPR_MIMES:
        if mime in bundle:
            continue
        fn = getattr(value, attr, None)
        if not callable(fn):
            continue
        try:
            data = fn()
        except Exception:
            continue
        if data is None:
            continue
        bundle[mime] = _coerce_image(data) if mime in ("image/png", "image/jpeg") else data
    for mime in ("image/png", "image/jpeg"):
        if mime in bundle:
            bundle[mime] = _coerce_image(bundle[mime])
    if "text/plain" not in bundle:
        try:
            bundle["text/plain"] = repr(value)
        except Exception:
            bundle["text/plain"] = "<unrepresentable value>"
    return bundle

def _bound_text(value):
    text = value if isinstance(value, str) else str(value)
    if len(text) <= _TEXT_LIMIT // 4:
        return text
    raw = text.encode("utf-8", "replace")
    if len(raw) <= _TEXT_LIMIT:
        return text
    kept = raw[:_TEXT_LIMIT - 128].decode("utf-8", "ignore")
    omitted = len(raw) - len(kept.encode("utf-8"))
    return kept + "\n[display truncated: %d bytes omitted; print a narrower slice, or write the rest to a file]" % omitted

def _bound_bundle(bundle):
    # A display is bounded like the JavaScript runner's: a value too large to
    # show whole is cut with a note, never allowed to break the protocol.
    for mime in list(bundle):
        value = bundle[mime]
        if mime in ("image/png", "image/jpeg"):
            if not isinstance(value, str) or len(value) > _IMAGE_LIMIT:
                del bundle[mime]
                bundle["text/plain"] = _bound_text(bundle.get("text/plain", "")) + "\n[%s display omitted: over %d MiB]" % (mime, _IMAGE_LIMIT >> 20)
        elif mime == "application/json":
            try:
                encoded = json.dumps(value, ensure_ascii=False)
            except Exception:
                encoded = None
            if encoded is None or len(encoded) > _TEXT_LIMIT:
                del bundle[mime]
        else:
            bundle[mime] = _bound_text(value)
    return bundle

def _emit_display(value, kind="display"):
    if _current_request_id is None:
        return
    bundle = _bound_bundle(_mime_bundle(value, kind == "display"))
    if "image/png" in bundle and type(value).__module__ == "matplotlib.figure" and type(value).__name__ == "Figure":
        _displayed_figure_ids.add(id(value))
    _emit_ordered({"type": "display", "id": _current_request_id, "status": kind, "bundle": bundle})

def display(*values):
    for value in values:
        _emit_display(value)

class ToolError(RuntimeError):
    """A host tool call that did not succeed. kind: blocked (policy or schema
    refused it), failed (the tool errored), budget, approval, limit, invalid,
    unavailable or aborted."""

    def __init__(self, message, kind="failed", tool=None):
        super().__init__(message)
        self.kind = kind
        self.tool = tool

def _call_tool(name, args=None):
    global _tool_sequence
    cell_id = _current_request_id
    if cell_id is None:
        raise ToolError("host tools are only available while an eval cell is running", "unavailable")
    name = str(name or "").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", name) is None:
        raise ToolError("invalid host tool name", "invalid", name)
    if args is None:
        args = {}
    try:
        # Round-trip to reject cycles and values the Go host cannot validate as
        # tool arguments. Do not use _emit's repr fallback on this boundary.
        clean_args = json.loads(json.dumps(args, ensure_ascii=False))
    except Exception as error:
        raise ToolError("host tool arguments must be JSON serializable: " + str(error), "invalid", name) from error
    with _seq_lock:
        _tool_sequence += 1
        request_id = cell_id + "/tool-" + str(_tool_sequence)
    slot = [threading.Event(), None]
    with _pending_lock:
        _pending[request_id] = slot
    _emit_ordered({
        "type": "tool_call",
        "id": cell_id,
        "request_id": request_id,
        "name": name,
        "arguments": clean_args,
    })
    slot[0].wait()
    response = slot[1]
    if response is None:
        raise ToolError("eval kernel closed while waiting for host tool " + name, "unavailable", name)
    if str(response.get("id", "")) != cell_id:
        raise ToolError("host tool response belongs to a different eval cell", "failed", name)
    if not response.get("ok"):
        kind = str(response.get("kind") or "failed")
        raise ToolError(str(response.get("error") or "host tool call failed"), kind if kind in _TOOL_KINDS else "failed", name)
    value = response.get("value")
    return "" if value is None else str(value)

def tool(name, args=None):
    """Call a host tool: tool("read", {"path": "a.txt"}) returns its text."""
    return _call_tool(name, args)

def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="eval-tool")
        return _pool

async def atool(name, args=None):
    """Awaitable tool(); gather several to run host tools concurrently."""
    return await asyncio.get_running_loop().run_in_executor(_get_pool(), _call_tool, name, args)

def tool_all(calls):
    """Run host tool calls concurrently: tool_all([(name, args), ...]) returns
    their texts in order; a call that failed gives its ToolError in its place."""
    futures = []
    for call in calls:
        call = tuple(call)
        futures.append(_get_pool().submit(_call_tool, call[0], call[1] if len(call) > 1 else None))
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except ToolError as error:
            results.append(error)
    return results

def _wait_for_tool_calls():
    # A cell is not done while a call it started is unanswered: the host records
    # the call with the cell that made it.
    while True:
        with _pending_lock:
            if not _pending:
                return
        time.sleep(0.01)

def _read_stdin():
    try:
        for line in _stdin:
            try:
                message = json.loads(line)
            except Exception:
                continue
            kind = message.get("type")
            if kind == "tool_result":
                with _pending_lock:
                    slot = _pending.pop(str(message.get("request_id", "")), None)
                if slot is not None:
                    slot[1] = message
                    slot[0].set()
            elif kind == "exit":
                break
            else:
                _requests.put(message)
    except Exception:
        pass
    finally:
        _requests.put(None)
        with _pending_lock:
            slots = list(_pending.values())
            _pending.clear()
        for slot in slots:
            slot[0].set()

def _flush_matplotlib_figures():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    try:
        numbers = list(plt.get_fignums())
    except Exception:
        return
    for number in numbers:
        try:
            figure = plt.figure(number)
            if id(figure) not in _displayed_figure_ids:
                _emit_display(figure)
            plt.close(figure)
        except Exception:
            continue

def _no_input(*_args, **_kwargs):
    raise RuntimeError("interactive input is not supported by AgentRay eval")

builtins.input = _no_input
_user_ns = {
    "__name__": "__main__",
    "__doc__": None,
    "__builtins__": builtins,
    "display": display,
    "tool": tool,
    "atool": atool,
    "tool_all": tool_all,
    "ToolError": ToolError,
}

_RUNNER_CALLS = ("tool", "_call_tool", "atool", "tool_all")

def _format_exception():
    # The runner's own frames (its loop, the cell launcher, the tool bridge) say
    # nothing the cell's author can use: keep the cell's frames and below.
    try:
        formatted = traceback.TracebackException.from_exception(sys.exc_info()[1])
        seen = set()
        def clean(item):
            if item is None or id(item) in seen:
                return
            seen.add(id(item))
            frames = list(item.stack)
            first = next((n for n, f in enumerate(frames) if f.filename == "<cell>"), None)
            if first is not None:
                frames = frames[first:]
            else:
                frames = [f for f in frames if f.filename != "<string>"]
            while frames and frames[-1].filename == "<string>" and frames[-1].name in _RUNNER_CALLS:
                frames.pop()
            item.stack = traceback.StackSummary.from_list(frames)
            clean(item.__cause__)
            clean(item.__context__)
        clean(formatted)
        return "".join(formatted.format())
    except Exception:
        return traceback.format_exc()

def _compile_cell(source):
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    tree = compile(source, "<cell>", "exec", flags=flags | ast.PyCF_ONLY_AST, dont_inherit=True)
    has_result = bool(tree.body and isinstance(tree.body[-1], ast.Expr))
    if has_result:
        expr = tree.body[-1]
        target = ast.Name(id=_result_name, ctx=ast.Store())
        tree.body[-1] = ast.copy_location(ast.Assign(targets=[target], value=expr.value), expr)
        ast.fix_missing_locations(tree)
    return compile(tree, "<cell>", "exec", flags=flags, dont_inherit=True), has_result

def _run_cell(source):
    compiled, has_result = _compile_cell(source)
    pending = eval(compiled, _user_ns, _user_ns)
    if inspect.isawaitable(pending):
        _loop.run_until_complete(pending)
    if not has_result:
        return None, False
    value = _user_ns.pop(_result_name, None)
    _user_ns["_"] = value
    return value, True

threading.Thread(target=_read_stdin, daemon=True).start()

while True:
    _request = _requests.get()
    if _request is None or _request.get("type") == "exit":
        break
    _request_id = str(_request.get("id", ""))
    _current_request_id = _request_id
    _displayed_figure_ids.clear()
    _execution_count += 1
    _status = "ok"
    _error_text = ""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _captures[:] = [_Capture("stdout", _request_id, 1), _Capture("stderr", _request_id, 2)]
    try:
        _value, _has_result = _run_cell(str(_request.get("code", "")))
        if _has_result and _value is not None:
            _emit_display(_value, "result")
        _flush_matplotlib_figures()
    except BaseException:
        _status = "error"
        _error_text = _format_exception()
    _wait_for_tool_calls()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    with _out_lock:
        for _capture in _captures:
            _capture.pump()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    # Synchronous output drains before the done frame. A deliberately detached
    # child may retain the old pipe; do not let it pin the cell indefinitely.
    for _capture in _captures:
        if not _capture.done.wait(0.5):
            _capture.abandoned = True
    _captures[:] = []
    _emit_text("error", _request_id, _error_text)
    _emit({
        "type": "done",
        "id": _request_id,
        "status": _status,
        "execution_count": _execution_count,
    })
    _current_request_id = None

try:
    _pending_tasks = asyncio.all_tasks(_loop)
    for _task in _pending_tasks:
        _task.cancel()
    if _pending_tasks:
        _loop.run_until_complete(asyncio.gather(*_pending_tasks, return_exceptions=True))
    _loop.close()
except Exception:
    pass
`
