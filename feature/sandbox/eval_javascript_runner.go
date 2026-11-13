package sandbox

// javascriptEvalRunner is a self-contained Node.js kernel. A REPLServer gives
// it persistent lexical bindings and Node's native top-level-await transform;
// AgentRay supplies framing, display(), bounded rich values, and lifecycle.
//
// POSIX/server launches reserve fd 3 for protocol traffic and redirect ordinary
// fd 1/2 away from it. That means console/process writes and even child
// processes using inherited stdio cannot spoof an NDJSON control frame. The
// trusted Windows-host fallback uses fd 1 and still intercepts the common
// console/process write paths.
const javascriptEvalRunner = `
"use strict";
const fs = require("node:fs");
const { registerHooks, stripTypeScriptTypes } = require("node:module");
const repl = require("node:repl");
const readline = require("node:readline");
const util = require("node:util");
const { AsyncLocalStorage } = require("node:async_hooks");
const { PassThrough } = require("node:stream");

const protocolFd = Number(process.env.AGENTRAY_EVAL_PROTOCOL_FD || "1");
const runs = new AsyncLocalStorage();
let executionCount = 0;
let bridgeSequence = 0;
const pendingBridges = new Map();
let liveRun = null;

// Attribute floating rejections to the active cell instead of letting Node's
// process-level unhandled-rejection policy kill the retained kernel after that
// cell already appeared successful.
process.on("unhandledRejection", reason => {
  // Node does not promise that AsyncLocalStorage context is preserved for the
  // process-level rejection event. Cells are serialized, so the explicit live
  // slot is the unambiguous owner.
  const run = liveRun;
  if (run) run.rejections.push(reason);
});

function writeProtocol(frame) {
  const bytes = Buffer.from(JSON.stringify(frame) + "\n", "utf8");
  let offset = 0;
  while (offset < bytes.length) offset += fs.writeSync(protocolFd, bytes, offset, bytes.length - offset);
}

function activeRun() {
  const run = runs.getStore();
  return run && run.active ? run : null;
}

function writeText(type, value) {
  const run = activeRun();
  if (!run || value === undefined || value === null) return;
  const text = Buffer.isBuffer(value) || value instanceof Uint8Array
    ? Buffer.from(value).toString("utf8")
    : String(value);
  for (let start = 0; start < text.length; start += 8192) {
    writeProtocol({ type, id: run.id, data: text.slice(start, start + 8192) });
  }
}

function redirectedWrite(type, chunk, encoding, callback) {
  writeText(type, chunk);
  const done = typeof encoding === "function" ? encoding : callback;
  if (typeof done === "function") queueMicrotask(done);
  return true;
}

process.stdout.write = (chunk, encoding, callback) => redirectedWrite("stdout", chunk, encoding, callback);
process.stderr.write = (chunk, encoding, callback) => redirectedWrite("stderr", chunk, encoding, callback);

function log(type, args) {
  writeText(type, util.formatWithOptions({ colors: false, depth: 6, maxArrayLength: 200 }, ...args) + "\n");
}

const consoleBridge = Object.create(console);
consoleBridge.log = (...args) => log("stdout", args);
consoleBridge.info = (...args) => log("stdout", args);
consoleBridge.debug = (...args) => log("stdout", args);
consoleBridge.warn = (...args) => log("stderr", args);
consoleBridge.error = (...args) => log("stderr", args);
consoleBridge.dir = value => log("stdout", [value]);

function imageBase64(data) {
  if (typeof data === "string") {
    if (data.length > 921600 || data.length === 0 || data.length % 4 !== 0 ||
        !/^[A-Za-z0-9+/]*={0,2}$/.test(data)) return null;
    return data;
  }
  let encoded = null;
  if (Buffer.isBuffer(data) || data instanceof Uint8Array) encoded = Buffer.from(data).toString("base64");
  else if (data instanceof ArrayBuffer) encoded = Buffer.from(data).toString("base64");
  else if (ArrayBuffer.isView(data)) encoded = Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString("base64");
  if (data && data.type === "Buffer" && Array.isArray(data.data)) {
    encoded = Buffer.from(data.data).toString("base64");
  }
  return encoded !== null && encoded.length <= 921600 ? encoded : null;
}

function boundedDisplayText(value) {
  const text = String(value);
  const bytes = Buffer.from(text, "utf8");
  if (bytes.length <= 262144) return text;
  return bytes.subarray(0, 262080).toString("utf8") + "\n[display truncated: " + (bytes.length - 262080) + " bytes omitted]";
}

function emitDisplay(value, status) {
  const run = activeRun();
  if (!run || value === undefined) return;
  let bundle;
  if (value && typeof value === "object" && value.type === "image" && typeof value.mimeType === "string") {
    const data = imageBase64(value.data);
    if (data === null || (value.mimeType !== "image/png" && value.mimeType !== "image/jpeg")) {
      writeText("stderr", "[display: invalid image payload omitted]\n");
      return;
    }
    bundle = { [value.mimeType]: data };
  } else if (value !== null && typeof value === "object") {
    try {
      const encoded = JSON.stringify(value);
      if (encoded !== undefined && Buffer.byteLength(encoded, "utf8") <= 262144) {
        bundle = { "application/json": value };
      } else {
        bundle = { "text/plain": "[JSON display omitted: value exceeds 262144 bytes]" };
      }
    } catch (_) {
      bundle = { "text/plain": boundedDisplayText(util.inspect(value, { depth: 6, maxArrayLength: 200, colors: false })) };
    }
  } else {
    bundle = { "text/plain": boundedDisplayText(value) };
  }
  writeProtocol({ type: "display", id: run.id, status, bundle });
}

const input = new PassThrough();
const output = new PassThrough();
output.resume();
const server = repl.start({ input, output, terminal: false, prompt: "", ignoreUndefined: true, useColors: false });
server.context.console = consoleBridge;
server.context.print = (...args) => log("stdout", args);
server.context.display = (...values) => values.forEach(value => emitDisplay(value, "display"));

function callHostTool(name, args = {}) {
  const run = activeRun();
  if (!run) return Promise.reject(new Error("host tools are only available while an eval cell is running"));
  name = String(name || "").trim();
  if (!/^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$/.test(name)) {
    return Promise.reject(new Error("invalid host tool name"));
  }
  let cleanArgs;
  try {
    const encoded = JSON.stringify(args === undefined ? {} : args);
    if (encoded === undefined) throw new Error("arguments are not JSON serializable");
    cleanArgs = JSON.parse(encoded);
  } catch (error) {
    return Promise.reject(new Error("host tool arguments must be JSON serializable: " + error.message));
  }
  const requestId = run.id + "/tool-" + (++bridgeSequence);
  let resolveBridge;
  let rejectBridge;
  const promise = new Promise((resolve, reject) => {
    resolveBridge = resolve;
    rejectBridge = reject;
  });
  // Attach a rejection observer immediately. The caller still observes the
  // same rejected promise when awaiting it, while an intentionally fire-and-
  // forget call cannot crash the retained Node process as an unhandled reject.
  promise.catch(() => {});
  run.bridges.add(promise);
  promise.then(
    () => run.bridges.delete(promise),
    () => run.bridges.delete(promise),
  );
  pendingBridges.set(requestId, { run, resolve: resolveBridge, reject: rejectBridge });
  try {
    writeProtocol({ type: "tool_call", id: run.id, request_id: requestId, name, arguments: cleanArgs });
  } catch (error) {
    pendingBridges.delete(requestId);
    rejectBridge(error);
  }
  return promise;
}

const toolProxy = new Proxy(callHostTool, {
  get(target, property) {
    if (property === "then") return undefined;
    if (property === Symbol.toStringTag) return "AgentRayToolBridge";
    if (typeof property !== "string") return Reflect.get(target, property);
    if (property in target) return Reflect.get(target, property);
    return args => callHostTool(property, args);
  },
});
server.context.tool = toolProxy;

// Node's built-in transformer keeps eval dependency-free while covering the
// TypeScript syntax agents commonly paste into scratch cells. Plain JavaScript
// never needs this API, so it remains usable on older operator-provided Node
// versions; TypeScript gets a precise version error instead of a vague parse
// failure. Transform mode also supports enums, namespaces, and parameter
// properties, unlike strip-only mode.
const looksLikeTypeScript =
  /(?:\bimport\s+type\b|\bexport\s+type\b|\b(?:import|export)\s*\{[^}\n]*\btype\s+\w|\binterface\s+\w|\btype\s+\w+\s*=|\b(?:enum|namespace)\s+\w|\b(?:public|private|protected|readonly|abstract|declare|override)\s+[$A-Za-z_]|\b(?:as|satisfies)\s+(?:string|number|boolean|any|unknown|object|[A-Z]|\bconst\b)|\??:\s*(?:string|number|boolean|any|unknown|void|never|object|[A-Z]\w*)\b|<\s*[A-Z]\w*\s*[,>]|[$A-Za-z_]\w*!\.)/;

function transformTypeScript(code) {
  if (!looksLikeTypeScript.test(code)) return code;
  if (typeof stripTypeScriptTypes !== "function") {
    throw new Error("TypeScript eval requires Node.js 22.13 or newer");
  }
  return stripTypeScriptTypes(code, { mode: "transform" });
}

// Node 22.15+ exposes synchronous loader hooks. Registering one in-process
// gives imported workspace .ts/.mts modules the same transform as cells,
// including non-erasable features such as enums and parameter properties. The
// hook is intentionally file/extension-scoped and delegates every other module
// to Node's normal loader, so package resolution and built-ins retain native
// semantics. Explicit extensions keep resolution deterministic.
if (typeof registerHooks === "function" && typeof stripTypeScriptTypes === "function") {
  registerHooks({
    load(url, context, nextLoad) {
      if (url.startsWith("file:")) {
        const pathname = new URL(url).pathname;
        if (pathname.endsWith(".ts") || pathname.endsWith(".mts")) {
          const source = fs.readFileSync(new URL(url), "utf8");
          return {
            format: "module",
            source: stripTypeScriptTypes(source, { mode: "transform", sourceUrl: url }),
            shortCircuit: true,
          };
        }
      }
      return nextLoad(url, context);
    },
  });
}

function skipQuoted(source, start, quote) {
  let i = start + 1;
  while (i < source.length) {
    if (source[i] === "\\") { i += 2; continue; }
    if (source[i] === quote) return i + 1;
    i += 1;
  }
  return source.length;
}

function skipLineComment(source, start) {
  let i = start + 2;
  while (i < source.length && source[i] !== "\n" && source[i] !== "\r") i += 1;
  return i;
}

function skipBlockComment(source, start) {
  const end = source.indexOf("*/", start + 2);
  return end === -1 ? source.length : end + 2;
}

function skipRegexLiteral(source, start) {
  let i = start + 1;
  let inClass = false;
  while (i < source.length) {
    const ch = source[i];
    if (ch === "\\") { i += 2; continue; }
    if (ch === "[") inClass = true;
    else if (ch === "]") inClass = false;
    else if (ch === "/" && !inClass) {
      i += 1;
      while (i < source.length && /[A-Za-z]/.test(source[i])) i += 1;
      return i;
    }
    i += 1;
  }
  return source.length;
}

function skipTemplateExpression(source, start) {
  let i = start;
  let braces = 1;
  let canEndExpression = false;
  while (i < source.length) {
    const ch = source[i];
    if (ch === "'" || ch === '"') { i = skipQuoted(source, i, ch); canEndExpression = true; continue; }
    if (source.charCodeAt(i) === 96) { i = skipTemplateLiteral(source, i); canEndExpression = true; continue; }
    if (ch === "/" && source[i + 1] === "/") { i = skipLineComment(source, i); continue; }
    if (ch === "/" && source[i + 1] === "*") { i = skipBlockComment(source, i); continue; }
    if (ch === "/" && !canEndExpression) { i = skipRegexLiteral(source, i); canEndExpression = true; continue; }
    if (ch === "{") { braces += 1; canEndExpression = false; i += 1; continue; }
    if (ch === "}") {
      braces -= 1;
      i += 1;
      if (braces === 0) return i;
      canEndExpression = true;
      continue;
    }
    if (/[$_A-Za-z0-9]/.test(ch)) canEndExpression = true;
    else if (!/\s/.test(ch)) canEndExpression = ch === ")" || ch === "]";
    i += 1;
  }
  return source.length;
}

function skipTemplateLiteral(source, start) {
  let i = start + 1;
  while (i < source.length) {
    if (source[i] === "\\") { i += 2; continue; }
    if (source.charCodeAt(i) === 96) return i + 1;
    if (source[i] === "$" && source[i + 1] === "{") {
      i = skipTemplateExpression(source, i + 2);
      continue;
    }
    i += 1;
  }
  return source.length;
}

const identifierPattern = /[$_\p{ID_Start}][$\u200C\u200D_\p{ID_Continue}]*/uy;
const expressionPrefixKeywords = new Set([
  "await", "case", "delete", "do", "else", "extends", "in", "instanceof",
  "new", "of", "return", "throw", "typeof", "void", "yield",
]);
const controlParenKeywords = new Set(["catch", "for", "if", "switch", "while", "with"]);

// This is intentionally a lexical scanner rather than a regex replacement:
// import-looking text in strings, templates, comments, and regular expressions
// must remain byte-for-byte user data. It emits only the token detail needed to
// recognize top-level static import declarations; Node remains the syntax
// authority for everything else.
function tokenizeForImports(source) {
  const tokens = [];
  let i = 0;
  let parens = 0;
  const parenKinds = [];
  let brackets = 0;
  let braces = 0;
  let canEndExpression = false;
  while (i < source.length) {
    const ch = source[i];
    if (/\s/.test(ch)) { i += 1; continue; }
    if (ch === "/" && source[i + 1] === "/") { i = skipLineComment(source, i); continue; }
    if (ch === "/" && source[i + 1] === "*") { i = skipBlockComment(source, i); continue; }
    const depth = { parens, brackets, braces };
    if (ch === "'" || ch === '"') {
      const end = skipQuoted(source, i, ch);
      tokens.push({ type: "string", value: source.slice(i, end), start: i, end, ...depth });
      i = end;
      canEndExpression = true;
      continue;
    }
    if (source.charCodeAt(i) === 96) {
      const end = skipTemplateLiteral(source, i);
      tokens.push({ type: "template", value: "", start: i, end, ...depth });
      i = end;
      canEndExpression = true;
      continue;
    }
    if (ch === "/" && !canEndExpression) {
      const end = skipRegexLiteral(source, i);
      tokens.push({ type: "regex", value: "", start: i, end, ...depth });
      i = end;
      canEndExpression = true;
      continue;
    }
    identifierPattern.lastIndex = i;
    const identifier = identifierPattern.exec(source);
    if (identifier) {
      const value = identifier[0];
      const end = identifierPattern.lastIndex;
      tokens.push({ type: "identifier", value, start: i, end, ...depth });
      i = end;
      canEndExpression = !expressionPrefixKeywords.has(value);
      continue;
    }
    if (/[0-9]/.test(ch)) {
      let end = i + 1;
      while (end < source.length && /[0-9A-Fa-f_xXoObBeE.n]/.test(source[end])) end += 1;
      tokens.push({ type: "number", value: source.slice(i, end), start: i, end, ...depth });
      i = end;
      canEndExpression = true;
      continue;
    }
    const two = source.slice(i, i + 2);
    const end = i + ((two === "++" || two === "--" || two === "=>" || two === "?.") ? 2 : 1);
    tokens.push({ type: "punct", value: source.slice(i, end), start: i, end, ...depth });
    let closedControlParen = false;
    if (ch === "(") {
      const previous = tokens[tokens.length - 2];
      const beforePrevious = tokens[tokens.length - 3];
      const control = controlParenKeywords.has(previous?.value) ||
        (previous?.value === "await" && beforePrevious?.value === "for");
      parenKinds.push(control);
      parens += 1;
    } else if (ch === ")") {
      closedControlParen = parenKinds.pop() === true;
      parens = Math.max(0, parens - 1);
    }
    else if (ch === "[") brackets += 1;
    else if (ch === "]") brackets = Math.max(0, brackets - 1);
    else if (ch === "{") braces += 1;
    else if (ch === "}") braces = Math.max(0, braces - 1);
    canEndExpression = (ch === ")" && !closedControlParen) || ch === "]" || ch === "}" || two === "++" || two === "--";
    i = end;
  }
  return tokens;
}

function parseNamedImports(tokens, index) {
  const pairs = [];
  if (tokens[index]?.value !== "{") return null;
  index += 1;
  while (index < tokens.length && tokens[index].value !== "}") {
    if (tokens[index].value === ",") { index += 1; continue; }
    const imported = tokens[index];
    if (imported.type !== "identifier" && imported.type !== "string") return null;
    index += 1;
    let local = imported;
    if (tokens[index]?.value === "as") {
      index += 1;
      local = tokens[index];
      if (!local || local.type !== "identifier") return null;
      index += 1;
    } else if (imported.type !== "identifier") {
      return null;
    }
    pairs.push({ imported: imported.value, local: local.value });
    if (tokens[index]?.value === ",") index += 1;
  }
  if (tokens[index]?.value !== "}") return null;
  return { pairs, index: index + 1 };
}

function parseStaticImport(source, tokens, startIndex) {
  let index = startIndex + 1;
  const first = tokens[index];
  if (!first || first.value === "(" || first.value === ".") return null;
  let defaultName = null;
  let namespaceName = null;
  let named = [];
  let sourceToken;

  if (first.type === "string") {
    sourceToken = first;
    index += 1;
  } else {
    if (first.type === "identifier" && first.value !== "from") {
      defaultName = first.value;
      index += 1;
      if (tokens[index]?.value === ",") index += 1;
    }
    if (tokens[index]?.value === "*") {
      if (tokens[index + 1]?.value !== "as" || tokens[index + 2]?.type !== "identifier") return null;
      namespaceName = tokens[index + 2].value;
      index += 3;
    } else if (tokens[index]?.value === "{") {
      const parsed = parseNamedImports(tokens, index);
      if (!parsed) return null;
      named = parsed.pairs;
      index = parsed.index;
    }
    if (tokens[index]?.value !== "from" || tokens[index + 1]?.type !== "string") return null;
    sourceToken = tokens[index + 1];
    index += 2;
  }

  let end = sourceToken.end;
  let options = "";
  if (tokens[index]?.value === "with" && tokens[index + 1]?.value === "{") {
    const open = tokens[index + 1];
    let cursor = index + 2;
    let level = 1;
    while (cursor < tokens.length && level > 0) {
      if (tokens[cursor].value === "{") level += 1;
      else if (tokens[cursor].value === "}") level -= 1;
      cursor += 1;
    }
    if (level !== 0) return null;
    const close = tokens[cursor - 1];
    options = ", { with: " + source.slice(open.start, close.end) + " }";
    end = close.end;
    index = cursor;
  }
  if (tokens[index]?.value === ";") {
    end = tokens[index].end;
    index += 1;
  }

  const call = "import(" + sourceToken.value + options + ")";
  let replacement;
  if (namespaceName) {
    replacement = "const " + namespaceName + " = await " + call + ";";
    if (defaultName) replacement += " const " + defaultName + " = " + namespaceName + ".default;";
  } else if (named.length > 0 || defaultName) {
    const properties = named.map(pair => pair.imported === pair.local ? pair.local : pair.imported + ": " + pair.local);
    if (defaultName) properties.unshift("default: " + defaultName);
    replacement = "const { " + properties.join(", ") + " } = await " + call + ";";
  } else {
    replacement = "await " + call + ";";
  }
  return { start: tokens[startIndex].start, end, replacement, nextIndex: index };
}

function rewriteStaticImports(code) {
  if (!code.includes("import")) return code;
  const tokens = tokenizeForImports(code);
  const edits = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type !== "identifier" || token.value !== "import" || token.parens || token.brackets || token.braces) continue;
    const next = tokens[index + 1];
    if (next?.value === "(" || next?.value === ".") continue;
    const edit = parseStaticImport(code, tokens, index);
    if (!edit) return code;
    edits.push(edit);
    index = edit.nextIndex - 1;
  }
  for (let index = edits.length - 1; index >= 0; index -= 1) {
    const edit = edits[index];
    code = code.slice(0, edit.start) + edit.replacement + code.slice(edit.end);
  }
  return code;
}

function prepareJavaScript(code) {
  return rewriteStaticImports(transformTypeScript(code));
}

function evaluate(code) {
	// REPLServer evaluates inside a Node domain. Rejecting a Promise directly
	// from its callback lets the domain consume that rejection before an outer
	// async catch can observe it, which would strand the cell without a done
	// frame. Runtime errors are emitted on that domain rather than passed to the
	// callback, so observe both paths and resolve an explicit result envelope.
	return new Promise(resolve => {
		let settled = false;
		const domain = server._domain;
		const finish = result => {
			if (settled) return;
			settled = true;
			resolve({ ...result, cleanup: () => domain?.removeListener("error", onDomainError) });
		};
		const onDomainError = error => {
			if (!settled) finish({ error, value: undefined });
			else if (liveRun) liveRun.rejections.push(error);
		};
		if (domain) domain.on("error", onDomainError);
		const invoke = () => server.eval(code, server.context, "<cell>", (error, value) => finish({ error, value }));
		if (domain) domain.run(invoke);
		else invoke();
	});
}

async function runCell(request) {
  const run = { id: String(request.id || ""), active: true, bridges: new Set(), rejections: [] };
  liveRun = run;
  executionCount += 1;
  let status = "ok";
  let cleanupEvaluation = () => {};
  await runs.run(run, async () => {
    try {
      const source = prepareJavaScript(String(request.code || ""));
      const result = await evaluate(source);
      cleanupEvaluation = result.cleanup;
      if (result.error) {
        status = "error";
        writeText("error", result.error.stack || String(result.error));
      } else if (result.value !== undefined) {
        emitDisplay(result.value, "result");
      }
    } catch (error) {
      status = "error";
      writeText("error", error && error.stack ? error.stack : String(error));
    } finally {
      // Keep the cell alive until every bridge request it started has received
      // a host response. This prevents an unawaited call from leaking into the
      // next retained cell or losing its audit trace.
      while (run.bridges.size > 0) {
        await Promise.allSettled(Array.from(run.bridges));
      }
      // Node reports unhandled rejections at the event-loop boundary. Give it
      // that boundary before finalizing so a floating rejection becomes a
      // cell error and the next retained cell stays healthy.
      await new Promise(resolve => setImmediate(resolve));
      cleanupEvaluation();
      if (run.rejections.length > 0) {
        status = "error";
        for (const reason of run.rejections) {
          writeText("error", reason && reason.stack ? reason.stack : String(reason));
        }
      }
      writeProtocol({ type: "done", id: run.id, status, execution_count: executionCount });
      run.active = false;
      if (liveRun === run) liveRun = null;
    }
  });
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let pending = Promise.resolve();
lines.on("line", line => {
  let request;
  try { request = JSON.parse(line); } catch (_) { return; }
  if (request.type === "tool_result") {
    const pendingBridge = pendingBridges.get(String(request.request_id || ""));
    if (!pendingBridge) return;
    pendingBridges.delete(String(request.request_id || ""));
    if (String(request.id || "") !== pendingBridge.run.id) {
      pendingBridge.reject(new Error("host tool response belongs to a different eval cell"));
    } else if (request.ok) {
      pendingBridge.resolve(request.value);
    } else {
      pendingBridge.reject(new Error(String(request.error || "host tool call failed")));
    }
    return;
  }
  pending = pending.then(async () => {
    if (request.type === "exit") {
      for (const bridge of pendingBridges.values()) bridge.reject(new Error("eval kernel is closing"));
      pendingBridges.clear();
      server.close();
      lines.close();
      return;
    }
    await runCell(request);
  }).catch(error => {
    const run = activeRun();
    if (run) writeProtocol({ type: "error", id: run.id, data: String(error) });
  });
});
`
