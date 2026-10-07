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
import io
import inspect
import json
import os
import re
import sys
import threading
import traceback

_protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="backslashreplace", buffering=1)
_write_lock = threading.Lock()
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_execution_count = 0
_result_name = "__agentray_cell_result__"
_current_request_id = None
_displayed_figure_ids = set()
_tool_sequence = 0

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

def _capture_fd(kind, request_id, target_fd):
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, target_fd)
    os.close(write_fd)
    done = threading.Event()
    def drain():
        try:
            while True:
                data = os.read(read_fd, 8192)
                if not data:
                    return
                _emit({"type": kind, "id": request_id, "data": data.decode("utf-8", "replace")})
        finally:
            os.close(read_fd)
            done.set()
    threading.Thread(target=drain, daemon=True).start()
    return done

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
        try:
            json.dumps(value, ensure_ascii=False)
            bundle["application/json"] = value
        except Exception:
            pass
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

def _emit_display(value, kind="display"):
    if _current_request_id is None:
        return
    bundle = _mime_bundle(value, kind == "display")
    if "image/png" in bundle and type(value).__module__ == "matplotlib.figure" and type(value).__name__ == "Figure":
        _displayed_figure_ids.add(id(value))
    _emit({"type": "display", "id": _current_request_id, "status": kind, "bundle": bundle})

def display(*values):
    for value in values:
        _emit_display(value)

def tool(name, args=None):
    global _tool_sequence
    if _current_request_id is None:
        raise RuntimeError("host tools are only available while an eval cell is running")
    name = str(name or "").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", name) is None:
        raise ValueError("invalid host tool name")
    if args is None:
        args = {}
    try:
        # Round-trip to reject cycles and values the Go host cannot validate as
        # tool arguments. Do not use _emit's repr fallback on this boundary.
        clean_args = json.loads(json.dumps(args, ensure_ascii=False))
    except Exception as error:
        raise TypeError("host tool arguments must be JSON serializable: " + str(error)) from error
    _tool_sequence += 1
    request_id = _current_request_id + "/tool-" + str(_tool_sequence)
    _emit({
        "type": "tool_call",
        "id": _current_request_id,
        "request_id": request_id,
        "name": name,
        "arguments": clean_args,
    })
    while True:
        response_line = sys.stdin.readline()
        if response_line == "":
            raise RuntimeError("eval kernel closed while waiting for host tool " + name)
        try:
            response = json.loads(response_line)
        except Exception:
            continue
        if response.get("type") == "exit":
            raise RuntimeError("eval kernel closed while waiting for host tool " + name)
        if response.get("type") != "tool_result" or str(response.get("request_id", "")) != request_id:
            continue
        if str(response.get("id", "")) != _current_request_id:
            raise RuntimeError("host tool response belongs to a different eval cell")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "host tool call failed"))
        return response.get("value")

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
}

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

for _line in sys.stdin:
    try:
        _request = json.loads(_line)
    except Exception:
        continue
    if _request.get("type") == "exit":
        break
    _request_id = str(_request.get("id", ""))
    _current_request_id = _request_id
    _displayed_figure_ids.clear()
    _execution_count += 1
    _status = "ok"
    _result_text = ""
    _error_text = ""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _stdout_done = _capture_fd("stdout", _request_id, 1)
    _stderr_done = _capture_fd("stderr", _request_id, 2)
    try:
        _value, _has_result = _run_cell(str(_request.get("code", "")))
        if _has_result and _value is not None:
            _emit_display(_value, "result")
        _flush_matplotlib_figures()
    except BaseException:
        _status = "error"
        _error_text = traceback.format_exc()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    # Synchronous output drains before the done frame. A deliberately detached
    # child may retain the old pipe; do not let it pin the cell indefinitely.
    _stdout_done.wait(0.5)
    _stderr_done.wait(0.5)
    _emit_text("result", _request_id, _result_text)
    _emit_text("error", _request_id, _error_text)
    _emit({
        "type": "done",
        "id": _request_id,
        "status": _status,
        "execution_count": _execution_count,
    })
    _current_request_id = None

try:
    _pending = asyncio.all_tasks(_loop)
    for _task in _pending:
        _task.cancel()
    if _pending:
        _loop.run_until_complete(asyncio.gather(*_pending, return_exceptions=True))
    _loop.close()
except Exception:
    pass
`
