import asyncio
import unittest
from unittest.mock import patch

from agent.livekit_worker import LiveKitAgentSessionPort, LiveKitSessionIO, run_workflow_session


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
    async def say(self, text, **kwargs): self.said.append((text, kwargs))
    async def interrupt(self, **kwargs): self.interrupted.append(kwargs)
    def shutdown(self, **kwargs): self.shutdown_args = kwargs


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

        session = asyncio.run(exercise())
        self.assertEqual(session.started["room"], room)
        self.assertEqual(session.said[0][0], "Namaste ji")
        self.assertEqual(session.interrupted, [{"force": True}])
        self.assertEqual(session.shutdown_args, {"drain": False})
        self.assertIn('"type":"transcript"', room.local_participant.messages[0][0])


if __name__ == "__main__":
    unittest.main()
