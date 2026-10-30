"""LiveKit integration boundary for the deterministic BimaCheck workflow.

The workflow deliberately consumes a small session port instead of importing
LiveKit throughout the decision logic. A deployment adapter owns the concrete
AgentSession audio and data-channel calls; this module owns lifecycle,
cancellation, provider selection, and cleanup around CallWorkflow.

The local 1.8.10 package source is unavailable in the current broken virtual
environment, so the room/audio-track binding is intentionally represented by
the ``SessionPort`` protocol. A concrete adapter must be verified against the
installed SDK before it is used for a live call.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any, Callable, Mapping, Protocol

from agent.core import Judge
from agent.workflow import CallWorkflow, LiveKitGnaniConfig, WorkflowResult, normalize_tts_text


class SessionPort(Protocol):
    """Minimum transport surface required by the deterministic workflow.

    ``listen`` must return a committed final customer transcript. A concrete
    LiveKit adapter must separately surface late transcripts; this interface
    does not silently turn them into ordinary answers.
    """

    async def start(self, *, stt: Any, tts: Any) -> None: ...
    async def listen(self) -> str: ...
    async def speak(self, text: str) -> None: ...
    async def stop_playback(self) -> None: ...
    async def publish(self, event: Mapping[str, Any]) -> None: ...
    async def close(self) -> None: ...


class LiveKitAgentSessionPort:
    """Room-bound SessionPort backed by a LiveKit AgentSession."""

    def __init__(self, room: Any, *, event_topic: str = "bimacheck.events", session_factory: Callable[..., Any] | None = None, agent_factory: Callable[..., Any] | None = None):
        self.room = room
        self.event_topic = event_topic
        self._session_factory = session_factory
        self._agent_factory = agent_factory
        self._session: Any | None = None
        self._transcripts: asyncio.Queue[str | None] = asyncio.Queue()

    async def start(self, *, stt: Any, tts: Any) -> None:
        if self._session_factory is None or self._agent_factory is None:
            from livekit.agents import Agent, AgentSession
            session_factory, agent_factory = AgentSession, Agent
        else:
            session_factory, agent_factory = self._session_factory, self._agent_factory
        self._session = session_factory(stt=stt, tts=tts, allow_interruptions=True)

        @self._session.on("user_input_transcribed")
        def on_transcript(event: Any) -> None:
            if event.is_final:
                self._transcripts.put_nowait(event.transcript)

        @self._session.on("close")
        def on_close(_: Any) -> None:
            self._transcripts.put_nowait(None)

        await self._session.start(
            agent=agent_factory(instructions="BimaCheck controls the workflow; do not generate replies or change its order."),
            room=self.room,
        )

    async def listen(self) -> str:
        transcript = await self._transcripts.get()
        if transcript is None:
            raise RuntimeError("LiveKit AgentSession closed before a final transcript")
        return transcript

    async def speak(self, text: str) -> None:
        if self._session is None:
            raise RuntimeError("LiveKit AgentSession has not started")
        await self._session.say(text, allow_interruptions=True, add_to_chat_ctx=False)

    async def stop_playback(self) -> None:
        if self._session is not None:
            await self._session.interrupt(force=True)

    async def publish(self, event: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))
        await self.room.local_participant.publish_data(payload, reliable=True, topic=self.event_topic)

    async def close(self) -> None:
        if self._session is not None:
            self._session.shutdown(drain=False)
            self._session = None


def synthetic_call_data() -> tuple[dict[str, str], dict[str, Any]]:
    """Return the plan's fictional primary scenario for local development."""
    return ({"name": "Ramesh Kumar"}, {
        "product_name": "Suraksha Pension Plus", "premium_amount": 150000,
        "premium_term_years": 5, "policy_term_years": 20,
        "payout_start_after_years": 5, "product_type": "deferred annuity",
        "payout_recipient": "Sunita",
        "stop_paying_explanation": "policy may lapse or benefits may reduce",
        "declared_income": 600000,
    })


def create_livekit_server(call_data_loader: Callable[[Any], tuple[Mapping[str, Any], Mapping[str, Any]]] = lambda _ctx: synthetic_call_data()) -> Any:
    """Create the documented AgentServer and register the room entrypoint."""
    from livekit.agents import AgentServer, JobContext
    server = AgentServer()

    @server.rtc_session(agent_name="bimacheck")
    async def entrypoint(ctx: JobContext) -> None:
        customer, policy = call_data_loader(ctx)
        await run_workflow_session(
            LiveKitAgentSessionPort(ctx.room), customer, policy, on_started=ctx.connect
        )

    return server


def run_worker() -> None:
    """Start the worker through the LiveKit Agents CLI entry point."""
    from livekit.agents import cli
    cli.run_app(create_livekit_server())


class LiveKitSessionIO:
    """VoiceIO implementation with cancellation-safe playback ownership."""

    def __init__(self, session: SessionPort):
        self.session = session
        self._playback_task: asyncio.Task[None] | None = None

    async def speak(self, text: str) -> None:
        await self.stop_playback()
        normalized = normalize_tts_text(text)
        self._playback_task = asyncio.create_task(self.session.speak(normalized))
        try:
            await self._playback_task
        except asyncio.CancelledError:
            raise
        finally:
            self._playback_task = None

    async def listen(self) -> str:
        return await self.session.listen()

    async def stop_playback(self) -> None:
        task = self._playback_task
        self._playback_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self.session.stop_playback()

    async def publish(self, event: Mapping[str, Any]) -> None:
        await self.session.publish(dict(event))


async def run_workflow_session(
    session: SessionPort,
    customer: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    judge: Judge | None = None,
    on_started: Callable[[], Any] | None = None,
) -> WorkflowResult:
    """Start providers, run one deterministic call, and always close cleanly."""
    providers = LiveKitGnaniConfig.build()
    io = LiveKitSessionIO(session)
    try:
        await session.start(stt=providers["stt"], tts=providers["tts"])
        if on_started is not None:
            await on_started()
        return await CallWorkflow(io, customer, policy, judge=judge).run()
    finally:
        try:
            await io.stop_playback()
        finally:
            await session.close()


def provider_configuration() -> dict[str, Any]:
    """Construct the existing Gnani configuration without making a network call."""
    return LiveKitGnaniConfig.build()
