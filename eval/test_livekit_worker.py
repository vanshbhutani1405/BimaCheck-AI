import asyncio
import unittest
from unittest.mock import patch

from agent.livekit_worker import LiveKitAgentSessionPort, LiveKitSessionIO, SessionClosed, run_workflow_session


class FakeSession:
    def __init__(self):
        self.started = None
        self.replies = iter(["haan"])
        self.spoken = []
        self.events = []
        self.stop_calls = 0
        self.closed = 0

    async def start(self, *, stt, tts):
        self.started = (stt, tts)

    async def listen(self):
        return next(self.replies)

    async def speak(self, text):
        self.spoken.append(text)

    async def stop_playback(self):
        self.stop_calls += 1

    async def publish(self, event):
        self.events.append(dict(event))

    async def close(self):
        self.closed += 1


class FakeRoom:
    class Participant:
        def __init__(self): self.messages = []
        async def publish_data(self, payload, *, reliable, topic):
            self.messages.append((payload, reliable, topic))
    def __init__(self): self.local_participant = self.Participant()


class FakeAgentSession:
    def __init__(self, **kwargs):
        self.kwargs, self.handlers, self.started = kwargs, {}, None
        self.said, self.interrupted, self.shutdown_args = [], [], None
    def on(self, event):
        def register(handler): self.handlers[event] = handler; return handler
        return register
    async def start(self, **kwargs): self.started = kwargs
    def say(self, text, **kwargs):
        handle = FakeSpeechHandle()
        self.said.append((text, kwargs, handle))
        return handle
    async def interrupt(self, **kwargs): self.interrupted.append(kwargs)
    def shutdown(self, **kwargs): self.shutdown_args = kwargs


class DelayedCommitAgentSession(FakeAgentSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commit_calls = []
        self.commit_future = None

    def commit_user_turn(self, **kwargs):
        self.commit_calls.append(kwargs)
        self.commit_future = asyncio.get_running_loop().create_future()
        return self.commit_future


class FakeSpeechHandle:
    def __init__(self):
        self.scheduled = True
        self.interrupted = False
        self._generations = [object()]
        self._done = False
        self._callbacks = []

    def done(self): return self._done
    def exception(self): return None
    def add_done_callback(self, callback):
        self._callbacks.append(callback)

    def __await__(self):
        async def wait():
            self._done = True
            for callback in self._callbacks:
                callback(self)
            return self
        return wait().__await__()


class FakeAgent:
    def __init__(self, **kwargs): self.kwargs = kwargs


class WorkerTests(unittest.TestCase):
    def test_consent_refusal_starts_and_closes_session(self):
        session = FakeSession()
        session.replies = iter(["nahi"])
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            result = asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))
        self.assertEqual(session.started, ("stt", "tts"))
        self.assertEqual(result.verdict, "red")
        self.assertEqual(result.hold_status, "hold")
        self.assertEqual(session.closed, 1)
        self.assertEqual(session.events[-1]["type"], "verdict")

    def test_speak_normalizes_text_and_interrupts_previous_playback(self):
        session = FakeSession()
        io = LiveKitSessionIO(session)
        asyncio.run(io.speak("  Namaste   ji "))
        asyncio.run(io.stop_playback())
        self.assertEqual(session.spoken, ["Namaste ji"])
        self.assertGreaterEqual(session.stop_calls, 2)

    def test_cleanup_runs_when_workflow_listen_fails(self):
        session = FakeSession()

        async def broken_listen():
            raise RuntimeError("disconnected")

        session.listen = broken_listen
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))
        self.assertEqual(session.closed, 1)

    def test_cleanup_runs_when_session_start_fails(self):
        session = FakeSession()

        async def broken_start(*, stt, tts):
            raise RuntimeError("provider unavailable")

        session.start = broken_start
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))
        self.assertEqual(session.closed, 1)

    def test_room_bound_port_uses_agent_session_and_data_channel(self):
        room = FakeRoom()
        port = LiveKitAgentSessionPort(room, session_factory=FakeAgentSession, agent_factory=FakeAgent)

        async def exercise():
            await port.start(stt="prisma", tts="timbre")
            session = port._session
            session.handlers["user_input_transcribed"](type("Event", (), {"is_final": True, "transcript": "bees saal"})())
            self.assertEqual(await port.listen(), "bees saal")
            await port.speak("  Namaste   ji ")
            await port.stop_playback()
            await port.publish({"type": "transcript", "text": "bees saal"})
            await port.close()
            return session

        with self.assertLogs("agent.livekit_worker", level="INFO") as logs:
            session = asyncio.run(exercise())
        self.assertEqual(session.started["room"], room)
        self.assertEqual(session.said[0][0], "Namaste ji")
        self.assertEqual(session.interrupted, [{"force": True}])
        self.assertEqual(session.shutdown_args, {"drain": False})
        self.assertIn('"type":"transcript"', room.local_participant.messages[0][0])
        self.assertTrue(any("audio_playback_completed" in line for line in logs.output))

    def test_participant_disconnect_unblocks_listen_as_handled_outcome(self):
        room = FakeRoom()
        port = LiveKitAgentSessionPort(room, session_factory=FakeAgentSession, agent_factory=FakeAgent)

        async def exercise():
            await port.start(stt="prisma", tts="timbre")
            port._session.handlers["close"](type("Event", (), {"reason": "CLIENT_INITIATED"})())
            with self.assertRaises(SessionClosed):
                await port.listen()
            await port.stop_playback()
            await port.close()

        asyncio.run(exercise())
        self.assertEqual(port._session, None)

    def test_cleanup_after_session_close_does_not_interrupt_or_shutdown_again(self):
        room = FakeRoom()
        port = LiveKitAgentSessionPort(room, session_factory=FakeAgentSession, agent_factory=FakeAgent)

        async def exercise():
            await port.start(stt="prisma", tts="timbre")
            session = port._session
            session.handlers["close"](type("Event", (), {"reason": "CLIENT_INITIATED"})())
            await port.stop_playback()
            await port.close()
            return session

        session = asyncio.run(exercise())
        self.assertEqual(session.interrupted, [])
        self.assertIsNone(session.shutdown_args)

    def test_final_transcript_waits_for_turn_commit_before_queueing(self):
        room = FakeRoom()
        port = LiveKitAgentSessionPort(
            room,
            session_factory=DelayedCommitAgentSession,
            agent_factory=FakeAgent,
        )

        async def exercise():
            await port.start(stt="prisma", tts="timbre")
            session = port._session
            session.handlers["user_input_transcribed"](
                type("Event", (), {"is_final": True, "transcript": "haan"})()
            )
            await asyncio.sleep(0)
            self.assertEqual(session.commit_calls, [{"transcript_timeout": 0.0, "skip_reply": True}])
            self.assertFalse(port._transcripts.qsize())
            listen_task = asyncio.create_task(port.listen())
            await asyncio.sleep(0)
            self.assertFalse(listen_task.done())
            session.commit_future.set_result("haan")
            self.assertEqual(await listen_task, "haan")
            await port.close()

        asyncio.run(exercise())

    def test_empty_late_commit_cannot_become_identity_answer(self):
        room = FakeRoom()
        port = LiveKitAgentSessionPort(
            room,
            session_factory=DelayedCommitAgentSession,
            agent_factory=FakeAgent,
        )

        async def exercise():
            await port.start(stt="prisma", tts="timbre")
            session = port._session

            session.handlers["user_input_transcribed"](
                type("Event", (), {"is_final": True, "transcript": "हाँ"})()
            )
            await asyncio.sleep(0)
            session.commit_future.set_result("हाँ")
            self.assertEqual(await port.listen(), "हाँ")

            # A late final event from the consent turn must not be delivered
            # as the answer to the newly spoken identity prompt.
            session.handlers["user_input_transcribed"](
                type("Event", (), {"is_final": True, "transcript": "जी हाँ"})()
            )
            await asyncio.sleep(0)
            session.commit_future.set_result("")
            await asyncio.sleep(0)
            identity_task = asyncio.create_task(port.listen())
            await asyncio.sleep(0)
            self.assertFalse(identity_task.done())

            session.handlers["user_input_transcribed"](
                type("Event", (), {"is_final": True, "transcript": "वंश"})()
            )
            await asyncio.sleep(0)
            session.commit_future.set_result("वंश")
            self.assertEqual(await identity_task, "वंश")
            await port.close()

        asyncio.run(exercise())

    def test_final_response_completes_before_workflow_cleanup(self):
        session = FakeSession()
        session.replies = iter(["nahi"])
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            result = asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))
        self.assertEqual(result.verdict, "red")
        self.assertEqual(session.spoken[-1], "Dhanyavaad. Is call ko human review ke liye bheja ja raha hai.")
        self.assertEqual(session.closed, 1)

    def test_runner_handles_disconnect_without_unexpected_exception(self):
        session = FakeSession()

        async def disconnected_listen():
            raise SessionClosed("participant disconnected")

        session.listen = disconnected_listen
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            result = asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))
        self.assertEqual((result.verdict, result.hold_status), ("amber", "pending_review"))
        self.assertEqual(session.closed, 1)

    def test_cleanup_failure_does_not_mask_workflow_exception(self):
        session = FakeSession()

        async def broken_listen():
            raise RuntimeError("original workflow failure")

        async def broken_close():
            raise RuntimeError("cleanup failure")

        session.listen = broken_listen
        session.close = broken_close
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            with self.assertRaisesRegex(RuntimeError, "original workflow failure"):
                asyncio.run(run_workflow_session(session, {"name": "Asha"}, {}))

    def test_hindi_name_variant_reaches_next_workflow_prompt(self):
        session = FakeSession()
        session.replies = iter(["जी हाँ मैं सहमत हूँ", "जी मेरा नाम वंश है", "नहीं"])
        with patch("agent.livekit_worker.LiveKitGnaniConfig.build", return_value={"stt": "stt", "tts": "tts"}):
            result = asyncio.run(run_workflow_session(
                session,
                {"name": "Vansh", "name_variants": ("वंश",)},
                {"product_name": "Synthetic Plan", "premium_amount": 1, "premium_term_years": 1, "policy_term_years": 1},
            ))
        self.assertTrue(any("Synthetic Plan" in text for text in session.spoken))
        self.assertEqual(result.verdict, "red")
        self.assertFalse(result.consent_refused)


if __name__ == "__main__":
    unittest.main()
