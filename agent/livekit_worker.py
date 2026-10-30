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
from contextlib import suppress
from typing import Any, Mapping, Protocol

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
) -> WorkflowResult:
    """Start providers, run one deterministic call, and always close cleanly."""
    providers = LiveKitGnaniConfig.build()
    io = LiveKitSessionIO(session)
    try:
        await session.start(stt=providers["stt"], tts=providers["tts"])
        return await CallWorkflow(io, customer, policy, judge=judge).run()
    finally:
        try:
            await io.stop_playback()
        finally:
            await session.close()


def provider_configuration() -> dict[str, Any]:
    """Construct the existing Gnani configuration without making a network call."""
    return LiveKitGnaniConfig.build()
