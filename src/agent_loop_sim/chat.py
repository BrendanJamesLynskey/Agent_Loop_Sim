"""What the model actually sees: messages rendered into one prompt string.

The format is Qwen2.5's ChatML chat template (``tokenizer_config.json`` of
Qwen2.5-1.5B-Instruct), so the scripted runs, the replayed traces and the local model all
see the same bytes:

- **native** tool calling: the tools' JSON schemas go into the system turn inside
  ``<tools></tools>``; the model answers with ``<tool_call>{"name": …, "arguments": …}</tool_call>``;
  each result comes back in a user turn as ``<tool_response>…</tool_response>``;
- **react** (text) tool calling: no tools in the template; the system prompt describes the
  tools and the Thought / Action / Action Input / Observation format, and each result comes
  back as a user turn starting ``Observation:``.

Two choices differ from the template, both so a prompt splits into per-message segments
that each start with ``<|im_start|>`` (then the tokens of the parts add up exactly to the
tokens of the whole, because the tokenizer splits at special tokens first):

- an assistant turn is the model's raw output text, echoed back as it was generated
  (the template would re-render the parsed tool calls);
- each tool result is its own user turn (the template groups consecutive results into one).
"""
from __future__ import annotations

from typing import Any

from .jsonfmt import dumps

TOOLS_HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
GENERATION_PROMPT = "<|im_start|>assistant\n"


def tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """A tool as the template's ``tools`` list holds it (OpenAI function format)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def react_system(system: str, tools: list[dict[str, Any]]) -> str:
    """The system prompt of a ReAct (text-parsing) harness: tools described in prose."""
    lines = [system, "", "You have access to the following tools:", ""]
    for t in tools:
        lines.append(f"{t['name']}: {t['description']} Arguments: {dumps(t['parameters'])}")
    names = ", ".join(t["name"] for t in tools)
    lines += [
        "",
        "Use the following format:",
        "",
        "Thought: think about what to do next",
        f"Action: the tool to use, one of [{names}]",
        "Action Input: the tool's arguments as a JSON object",
        "Observation: the result of the tool (the system writes this)",
        "... (Thought/Action/Action Input/Observation can repeat)",
        "Thought: I now know the final answer",
        "Final Answer: the answer to the task",
    ]
    return "\n".join(lines)


def system_segment(system: str, tools: list[dict[str, Any]], style: str) -> tuple[str, str]:
    """(the system turn as rendered, the same turn without the tools) for one prompt.

    The difference in tokens between the two is what the tool definitions cost.
    """
    if style == "native" and tools:
        bare = "<|im_start|>system\n" + system + "<|im_end|>\n"
        body = "".join("\n" + dumps(tool_schema(t)) for t in tools)
        return "<|im_start|>system\n" + system + TOOLS_HEADER + body + TOOLS_FOOTER + "<|im_end|>\n", bare
    if style == "react":
        bare = "<|im_start|>system\n" + system + "<|im_end|>\n"
        return "<|im_start|>system\n" + react_system(system, tools) + "<|im_end|>\n", bare
    seg = "<|im_start|>system\n" + system + "<|im_end|>\n"
    return seg, seg


def message_segment(m: dict[str, Any]) -> str:
    """One non-system message as rendered (user / assistant / tool)."""
    role = m["role"]
    if role == "tool":
        return "<|im_start|>user\n<tool_response>\n" + m["content"] + "\n</tool_response><|im_end|>\n"
    return "<|im_start|>" + role + "\n" + m["content"] + "<|im_end|>\n"


def render(system: str, tools: list[dict[str, Any]], style: str, messages: list[dict[str, Any]]) -> str:
    """The whole prompt, ending with the generation prompt (``<|im_start|>assistant\\n``)."""
    seg, _ = system_segment(system, tools, style)
    return seg + "".join(message_segment(m) for m in messages) + GENERATION_PROMPT
