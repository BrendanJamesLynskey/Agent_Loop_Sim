/** What the model sees: the TS twin of `agent_loop_sim/chat.py` (Qwen2.5's ChatML). */
import type { Obj } from "./data";
import { dumps } from "./jsonfmt";

export const TOOLS_HEADER =
  "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
  "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
export const TOOLS_FOOTER =
  "\n</tools>\n\nFor each function call, return a json object with function name and arguments " +
  "within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
  '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>';
export const GENERATION_PROMPT = "<|im_start|>assistant\n";

export function toolSchema(tool: Obj): Obj {
  return {
    type: "function",
    function: { name: tool.name, description: tool.description, parameters: tool.parameters },
  };
}

export function reactSystem(system: string, tools: Obj[]): string {
  const lines = [system, "", "You have access to the following tools:", ""];
  for (const t of tools) lines.push(`${t.name}: ${t.description} Arguments: ${dumps(t.parameters)}`);
  const names = tools.map((t) => t.name).join(", ");
  lines.push(
    "",
    "Use the following format:",
    "",
    "Thought: think about what to do next",
    `Action: the tool to use, one of [${names}]`,
    "Action Input: the tool's arguments as a JSON object",
    "Observation: the result of the tool (the system writes this)",
    "... (Thought/Action/Action Input/Observation can repeat)",
    "Thought: I now know the final answer",
    "Final Answer: the answer to the task",
  );
  return lines.join("\n");
}

/** [the system turn as rendered, the same turn without the tools]. */
export function systemSegment(system: string, tools: Obj[], style: string): [string, string] {
  const bare = "<|im_start|>system\n" + system + "<|im_end|>\n";
  if (style === "native" && tools.length > 0) {
    const body = tools.map((t) => "\n" + dumps(toolSchema(t))).join("");
    return ["<|im_start|>system\n" + system + TOOLS_HEADER + body + TOOLS_FOOTER + "<|im_end|>\n", bare];
  }
  if (style === "react") return ["<|im_start|>system\n" + reactSystem(system, tools) + "<|im_end|>\n", bare];
  return [bare, bare];
}

export function messageSegment(m: Obj): string {
  if (m.role === "tool")
    return "<|im_start|>user\n<tool_response>\n" + m.content + "\n</tool_response><|im_end|>\n";
  return "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
}

export function render(system: string, tools: Obj[], style: string, messages: Obj[]): string {
  const [seg] = systemSegment(system, tools, style);
  return seg + messages.map(messageSegment).join("") + GENERATION_PROMPT;
}
