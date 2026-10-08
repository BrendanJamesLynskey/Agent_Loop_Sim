"""The MCP server the conformance recordings talk to, written with the official MCP Python SDK.

Run by ``scripts/record_sdk.py`` as a stdio subprocess. The engine's model of this server
(``agent_loop_sim.protocols.mcp.FIXTURE_SERVER``) must produce the same JSON-RPC messages.
It speaks both protocol eras the SDK supports: the handshake era (``initialize``, up to
2025-11-25) and the stateless 2026-07-28 revision, where a request that needs the user or a
model returns ``resultType: "input_required"`` instead of sending a request to the client.
"""
from __future__ import annotations

import warnings

import mcp_types as types
from mcp.server import MCPServer
from mcp.server.mcpserver import Context

warnings.simplefilter("ignore")  # the SDK marks sampling as deprecated in 2026-07-28 (SEP-2577)

MODERN = "2026-07-28"
mcp = MCPServer("fixture-server", version="1.0.0")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
async def read_doc(uri: str, ctx: Context) -> str:
    """Read a document by URI."""
    contents = list(await ctx.read_resource(uri))
    return str(contents[0].content)


@mcp.tool()
async def count(n: int, ctx: Context) -> str:
    """Count to n, reporting progress."""
    for i in range(1, n + 1):
        await ctx.report_progress(i, n)
    return f"counted to {n}"


CONFIRM_SCHEMA = {"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]}


@mcp.tool()
async def delete_file(path: str, ctx: Context) -> str:
    """Delete a file after the user confirms."""
    if ctx.protocol_version == MODERN:
        responses = ctx.input_responses
        if not responses:
            return types.InputRequiredResult(  # type: ignore[return-value]
                input_requests={
                    "confirm": types.ElicitRequest(
                        params=types.ElicitRequestFormParams(message=f"Delete {path}?", requested_schema=CONFIRM_SCHEMA)
                    )
                }
            )
        answer = responses["confirm"]
        ok = answer.action == "accept" and bool((answer.content or {}).get("confirm"))
    else:
        r = await ctx.session.elicit_form(f"Delete {path}?", CONFIRM_SCHEMA, related_request_id=ctx.request_id)
        ok = r.action == "accept" and bool((r.content or {}).get("confirm"))
    return f"deleted {path}" if ok else f"kept {path}"


def _sampling_request(text: str) -> types.CreateMessageRequestParams:
    return types.CreateMessageRequestParams(
        messages=[types.SamplingMessage(role="user", content=types.TextContent(type="text", text=f"Summarise: {text}"))],
        max_tokens=50,
    )


@mcp.tool()
async def summarise(text: str, ctx: Context) -> str:
    """Summarise text with the client's model."""
    if ctx.protocol_version == MODERN:
        responses = ctx.input_responses
        if not responses:
            return types.InputRequiredResult(  # type: ignore[return-value]
                input_requests={"summary": types.CreateMessageRequest(params=_sampling_request(text))}
            )
        content = responses["summary"].content
    else:
        p = _sampling_request(text)
        r = await ctx.session.create_message(p.messages, max_tokens=p.max_tokens, related_request_id=ctx.request_id)
        content = r.content
    return content.text if isinstance(content, types.TextContent) else "?"


@mcp.tool()
def fail(reason: str) -> str:
    """Always fails (to show a tool error)."""
    raise ValueError(reason)


@mcp.resource("docs://readme", mime_type="text/markdown")
def readme() -> str:
    """The project README."""
    return "# Demo\nA tiny project."


@mcp.prompt()
def review(code: str) -> str:
    """Review a piece of code."""
    return f"Please review:\n{code}"


if __name__ == "__main__":
    mcp.run()
