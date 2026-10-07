/** Model text -> tool calls, in both loop styles: the TS twin of `agent_loop_sim/parse.py`. */
import { TOOL_SPECS, type Obj } from "./data";
import { trim, validateArgs } from "./tools";

export function loadJson(text: string): [boolean, unknown] {
  try {
    return [true, JSON.parse(text)];
  } catch {
    return [false, null];
  }
}

const isPlainObject = (x: unknown): x is Obj =>
  typeof x === "object" && x !== null && !Array.isArray(x);

function checkCall(obj: unknown, allowed: string[]): [Obj | null, string | null] {
  if (!isPlainObject(obj) || typeof obj.name !== "string")
    return [null, 'a tool call must be a JSON object with a "name" string and an "arguments" object'];
  const name = obj.name as string;
  let args: unknown = Object.prototype.hasOwnProperty.call(obj, "arguments") ? obj.arguments : {};
  if (typeof args === "string") {
    const [ok, inner] = loadJson(args);
    if (ok) args = inner;
  }
  if (!allowed.includes(name)) return [null, `unknown tool '${name}'; the tools are: ${allowed.join(", ")}`];
  const why = validateArgs(TOOL_SPECS[name]!, args);
  if (why !== null) return [null, `invalid arguments for ${name}: ${why}`];
  return [{ name, args }, null];
}

export function parseNative(text: string, allowed: string[]): Obj {
  const calls: Obj[] = [];
  let pos = 0;
  for (;;) {
    const start = text.indexOf("<tool_call>", pos);
    if (start < 0) break;
    const end = text.indexOf("</tool_call>", start);
    if (end < 0) return { kind: "malformed", error: "a <tool_call> block is not closed with </tool_call>" };
    const body = trim(text.slice(start + "<tool_call>".length, end));
    const [ok, obj] = loadJson(body);
    if (!ok) return { kind: "malformed", error: "the text inside <tool_call></tool_call> is not valid JSON" };
    const [call, err] = checkCall(obj, allowed);
    if (err !== null) return { kind: "malformed", error: err };
    calls.push(call!);
    pos = end + "</tool_call>".length;
  }
  if (calls.length === 0) return { kind: "final", final: trim(text) };
  return { kind: "calls", calls };
}

export function parseReact(text: string, allowed: string[]): Obj {
  const obs = text.indexOf("Observation:");
  if (obs >= 0) text = text.slice(0, obs);
  const act = text.indexOf("Action:");
  const fin = text.indexOf("Final Answer:");
  if (fin >= 0 && (act < 0 || fin < act))
    return { kind: "final", final: trim(text.slice(fin + "Final Answer:".length)) };
  if (act < 0)
    return { kind: "malformed", error: "write either 'Action:' with 'Action Input:', or 'Final Answer:'" };
  const lineEnd = text.indexOf("\n", act);
  const name = trim(text.slice(act + "Action:".length, lineEnd >= 0 ? lineEnd : text.length));
  const inp = text.indexOf("Action Input:", act);
  if (inp < 0)
    return { kind: "malformed", error: "'Action:' must be followed by 'Action Input:' with a JSON object" };
  const raw = trim(text.slice(inp + "Action Input:".length));
  const [ok, args] = loadJson(raw);
  if (!ok) return { kind: "malformed", error: "the Action Input is not a valid JSON object" };
  const [call, err] = checkCall({ name, arguments: args }, allowed);
  if (err !== null) return { kind: "malformed", error: err };
  return { kind: "calls", calls: [call] };
}

export function parse(text: string, style: string, allowed: string[]): Obj {
  return style === "native" ? parseNative(text, allowed) : parseReact(text, allowed);
}
