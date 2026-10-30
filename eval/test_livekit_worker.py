import asyncio
import unittest
from unittest.mock import patch

from agent.livekit_worker import LiveKitSessionIO, run_workflow_session


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


if __name__ == "__main__":
    unittest.main()
