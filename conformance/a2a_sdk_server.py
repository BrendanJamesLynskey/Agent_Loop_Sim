"""The A2A agent the conformance recordings talk to, written with the official A2A Python SDK.

Run in-process by ``scripts/record_a2a_sdk.py`` (Starlette's test client: no socket is opened).
The engine's model of this agent (``agent_loop_sim.protocols.a2a.Agent``) must answer every
request with the same JSON-RPC messages and SSE events, apart from the IDs and timestamps the
SDK generates. A2A protocol version 1.0 (specification release 1.0.1).

The agent is a scripted "research agent" another agent delegates to:

- ``hello…``                 answers with a Message (no task);
- ``delete…``                rejects the task;
- ``summarise the meeting``  needs input (which meeting?), then summarises the answer;
- ``check my calendar``      needs the user to sign in (auth required), then answers;
- ``summarise <x>``          works and returns its summary as an artifact in two chunks.
"""
from __future__ import annotations

import uuid

from starlette.applications import Starlette

from a2a.helpers.proto_helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState

AGENT = "https://research.example.com"
RPC_PATH = "/a2a"

CARD = AgentCard(
    name="Research agent",
    description="Summarises documents and meetings for other agents.",
    supported_interfaces=[AgentInterface(url=AGENT + RPC_PATH, protocol_binding="JSONRPC", protocol_version="1.0")],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(id="summarise", name="Summarise", description="Summarise a document or a meeting.",
                   tags=["summary", "text"], examples=["summarise the design notes"]),
        AgentSkill(id="calendar", name="Calendar digest",
                   description="Today's meetings from the user's calendar (the user signs in first).",
                   tags=["calendar"], examples=["check my calendar"]),
    ],
)


def agent_text(u: TaskUpdater, text: str):
    return u.new_agent_message([new_text_part(text)])


async def finish(u: TaskUpdater, head: str, body: str) -> None:
    """The summary as one artifact streamed in two chunks, then completed."""
    aid = str(uuid.uuid4())
    await u.add_artifact([new_text_part(head)], artifact_id=aid, name="summary")
    await u.add_artifact([new_text_part(body)], artifact_id=aid, name="summary", append=True, last_chunk=True)
    await u.complete()


class ResearchAgent(AgentExecutor):
    async def execute(self, context: RequestContext, q) -> None:  # noqa: ANN001
        text = context.get_user_input()
        task = context.current_task
        if task is None:
            if text.startswith("hello"):
                await q.enqueue_event(new_text_message("Hello. Ask me to summarise something.",
                                                       context_id=context.context_id))
                return
            task = new_task_from_user_message(context.message)
            await q.enqueue_event(task)
            u = TaskUpdater(q, task.id, task.context_id)
            if text.startswith("delete"):
                await u.reject(agent_text(u, "I only summarise; I do not delete files."))
                return
            await u.start_work()
            if text == "summarise the meeting":
                await u.requires_input(agent_text(u, "Which meeting: Monday's or Tuesday's?"))
                return
            if text == "check my calendar":
                await u.requires_auth(agent_text(u, "Sign in to the calendar first: https://calendar.example.com/authorize"))
                return
            subject = text[len("summarise "):] if text.startswith("summarise ") else text
            await finish(u, "Summary of " + subject + ": ", "three decisions, two actions, no blockers.")
            return
        u = TaskUpdater(q, task.id, task.context_id)
        was = task.status.state
        await u.start_work()
        if was == TaskState.TASK_STATE_AUTH_REQUIRED:
            await finish(u, "Today: ", "design review at 10:00, one-to-one at 15:00.")
        else:
            await finish(u, "Summary of " + text + "'s meeting: ", "three decisions, two actions, no blockers.")

    async def cancel(self, context: RequestContext, q) -> None:  # noqa: ANN001
        await TaskUpdater(q, context.task_id, context.context_id).cancel()


def app() -> Starlette:
    handler = DefaultRequestHandlerV2(agent_executor=ResearchAgent(), task_store=InMemoryTaskStore(), agent_card=CARD)
    return Starlette(routes=create_agent_card_routes(CARD) + create_jsonrpc_routes(handler, RPC_PATH))
