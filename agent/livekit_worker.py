"""LiveKit integration boundary for the deterministic BimaCheck workflow.

The workflow deliberately consumes a small session port instead of importing
LiveKit throughout the decision logic. A deployment adapter owns the concrete
AgentSession audio and data-channel calls; this module owns lifecycle,
cancellation, provider selection, and cleanup around CallWorkflow.

The room/audio-track binding uses the installed LiveKit Agents 1.8.10
AgentSession API. The SessionPort remains the seam that keeps provider calls
out of the deterministic workflow.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from typing import Any, Callable, Mapping, Protocol

from agent.compliance import VerificationEngine
from agent.core import Judge
from agent.workflow import CallWorkflow, LiveKitGnaniConfig, WorkflowResult, normalize_tts_text
from agent.session import SessionContext, SessionStatus
from agent.policy import load_policy


logger = logging.getLogger(__name__)


class SessionClosed(RuntimeError):
    """The room/session closed while the workflow was waiting for a turn."""


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


def initialize_gnani_plugin() -> Any:
    """Register Gnani before AgentServer dispatches any jobs."""
    return LiveKitGnaniConfig.initialize()


class LiveKitAgentSessionPort:
    """Room-bound SessionPort backed by a LiveKit AgentSession."""

    def __init__(self, room: Any, *, event_topic: str = "bimacheck.events", session_factory: Callable[..., Any] | None = None, agent_factory: Callable[..., Any] | None = None):
        self.room = room
        self.event_topic = event_topic
        self._session_factory = session_factory
        self._agent_factory = agent_factory
        self._session: Any | None = None
        self._transcripts: asyncio.Queue[str | None] = asyncio.Queue()
        self._active = False
        self._close_signal_sent = False
        self._turn_commit_tasks: set[asyncio.Task[None]] = set()

    async def start(self, *, stt: Any, tts: Any) -> None:
        if self._session_factory is None or self._agent_factory is None:
            from livekit.agents import Agent, AgentSession
            session_factory, agent_factory = AgentSession, Agent
        else:
            session_factory, agent_factory = self._session_factory, self._agent_factory
        self._session = session_factory(stt=stt, tts=tts, allow_interruptions=True)

        @self._session.on("user_input_transcribed")
        def on_transcript(event: Any) -> None:
            logger.info("final_transcript_event_received", extra={"is_final": bool(event.is_final), "text_length": len(event.transcript)})
            if event.is_final:
                task = asyncio.create_task(self._commit_then_queue(event.transcript))
                self._turn_commit_tasks.add(task)
                task.add_done_callback(self._turn_commit_tasks.discard)

        @self._session.on("close")
        def on_close(event: Any) -> None:
            logger.info("livekit_session_close_event", extra={"reason": getattr(event, "reason", "unknown")})
            self._active = False
            if not self._close_signal_sent:
                self._close_signal_sent = True
                self._transcripts.put_nowait(None)

        await self._session.start(
            agent=agent_factory(instructions="BimaCheck controls the workflow; do not generate replies or change its order."),
            room=self.room,
        )
        self._active = True
        self._close_signal_sent = False

    async def listen(self) -> str:
        transcript = await self._transcripts.get()
        if transcript is None:
            logger.info("transcript_queue_interrupted_by_session_close")
            raise SessionClosed("LiveKit AgentSession closed before a final transcript")
        logger.info("transcript_retrieved_from_queue", extra={"text_length": len(transcript)})
        return transcript

    async def _commit_then_queue(self, transcript: str) -> None:
        """Publish a final STT result only after LiveKit commits its user turn.

        In Agents 1.8.10 ``user_input_transcribed`` is emitted from the final
        STT callback before end-of-turn processing.  ``commit_user_turn`` is a
        public API and its future resolves after the current turn has been
        committed.  ``skip_reply`` is required because BimaCheck owns replies.
        The zero timeout avoids adding a second STT wait when the final event
        is already available.
        """
        session = self._session
        commit_user_turn = getattr(session, "commit_user_turn", None)
        if callable(commit_user_turn):
            committed = commit_user_turn(transcript_timeout=0.0, skip_reply=True)
            committed_transcript = await committed
            logger.info("user_turn_committed_before_queue", extra={"text_length": len(transcript)})
            if not committed_transcript or not committed_transcript.strip():
                logger.info("empty_committed_turn_ignored")
                return
            transcript = committed_transcript
        self._transcripts.put_nowait(transcript)
        logger.info("transcript_added_to_queue", extra={"queue_size": self._transcripts.qsize()})

    async def speak(self, text: str) -> None:
        if self._session is None or not self._active:
            raise RuntimeError("LiveKit AgentSession has not started")
        normalized = normalize_tts_text(text)
        handle = self._session.say(normalized, allow_interruptions=True, add_to_chat_ctx=False)
        logger.info("speech_scheduled", extra={"text_length": len(normalized), "scheduled": getattr(handle, "scheduled", None)})

        def on_done(done_handle: Any) -> None:
            interrupted = bool(getattr(done_handle, "interrupted", False))
            error = None
            with suppress(Exception):
                error = done_handle.exception()
            logger.info(
                "speech_interrupted" if interrupted else "audio_playback_completed",
                extra={"text_length": len(normalized), "interrupted": interrupted, "error_type": type(error).__name__ if error else None},
            )

        handle.add_done_callback(on_done)
        generation_task = asyncio.create_task(self._observe_generation_start(handle))
        try:
            await handle
            error = handle.exception()
            if error is not None:
                raise error
        except asyncio.CancelledError:
            logger.info("speech_cancelled", extra={"text_length": len(normalized)})
            with suppress(Exception):
                if not handle.done():
                    handle.interrupt(force=True)
                    await handle
            raise
        finally:
            with suppress(asyncio.CancelledError):
                await generation_task

    async def _observe_generation_start(self, handle: Any) -> None:
        """Observe the 1.8.10 handle's internal generation future for diagnostics."""
        while not handle.done() and not getattr(handle, "_generations", ()):
            await asyncio.sleep(0)
        if getattr(handle, "_generations", ()):
            logger.info("audio_generation_started")

    async def stop_playback(self) -> None:
        if self._session is not None and self._active:
            await self._session.interrupt(force=True)
        else:
            logger.info("playback_stop_skipped_session_closed")

    async def publish(self, event: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))
        await self.room.local_participant.publish_data(payload, reliable=True, topic=self.event_topic)

    async def close(self) -> None:
        for task in tuple(self._turn_commit_tasks):
            task.cancel()
        for task in tuple(self._turn_commit_tasks):
            with suppress(asyncio.CancelledError):
                await task
        self._turn_commit_tasks.clear()
        if self._session is not None:
            if self._active:
                logger.info("session_shutdown_started", extra={"reason": "application_cleanup"})
                self._active = False
                self._session.shutdown(drain=False)
            else:
                logger.info("session_shutdown_skipped_already_closed")
            self._session = None


def synthetic_call_data() -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Return the plan's fictional primary scenario for local development."""
    customer = {"name": "Vansh", "name_variants": ("वंश",), "declared_income": 600000}
    return customer, load_policy().with_customer_context(customer)


def create_livekit_server(call_data_loader: Callable[[Any], tuple[Mapping[str, Any], Mapping[str, Any]]] = lambda _ctx: synthetic_call_data()) -> Any:
    """Create the documented AgentServer and register the room entrypoint."""
    from livekit.agents import AgentServer, JobContext
    initialize_gnani_plugin()
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
    session_context: SessionContext | None = None,
) -> WorkflowResult:
    """Start providers, run one deterministic call, and always close cleanly."""
    call_session = session_context or SessionContext()
    io = LiveKitSessionIO(session)
    try:
        if call_session.status == SessionStatus.CREATED:
            call_session.start()
        providers = LiveKitGnaniConfig.build()
        await session.start(stt=providers["stt"], tts=providers["tts"])
        if on_started is not None:
            await on_started()
        logger.info("workflow_started")
        result = await CallWorkflow(io, customer, policy, judge=judge, session_context=call_session).run()
        logger.info("workflow_returned", extra={"verdict": result.verdict, "hold_status": result.hold_status})
        return result
    except asyncio.CancelledError:
        logger.warning("workflow_cancelled", exc_info=True)
        raise
    except SessionClosed:
        logger.info("workflow_ended_with_session_disconnect")
        call_session.close()
        report = VerificationEngine().calculate_verdict(session=call_session)
        return WorkflowResult(report.verdict, report.hold_status)
    except Exception:
        logger.exception("workflow_raised_exception")
        raise
    finally:
        try:
            await io.stop_playback()
        except Exception:
            logger.exception("playback_cleanup_failed")
        finally:
            try:
                await session.close()
            except Exception:
                logger.exception("session_cleanup_failed")
            if not call_session.completed:
                call_session.close()


def provider_configuration() -> dict[str, Any]:
    """Construct the existing Gnani configuration without making a network call."""
    return LiveKitGnaniConfig.build()
