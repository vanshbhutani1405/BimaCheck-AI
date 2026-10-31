import sys
import threading
import types
import unittest
from unittest.mock import patch

from agent.livekit_worker import create_livekit_server, initialize_gnani_plugin
from agent.workflow import LiveKitGnaniConfig


class FakeGnani:
    class STT:
        def __init__(self, **kwargs): self.kwargs = kwargs

    class TTS:
        def __init__(self, **kwargs): self.kwargs = kwargs


class GnaniRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.previous_module = LiveKitGnaniConfig._gnani_module
        LiveKitGnaniConfig._gnani_module = None

    def tearDown(self):
        LiveKitGnaniConfig._gnani_module = self.previous_module

    def test_initialization_is_main_thread_and_idempotent(self):
        with patch("agent.workflow.importlib.import_module", return_value=FakeGnani) as importer:
            self.assertIs(initialize_gnani_plugin(), FakeGnani)
            self.assertIs(initialize_gnani_plugin(), FakeGnani)
        importer.assert_called_once_with("livekit.plugins.gnani")

    def test_background_initialization_is_rejected_before_import(self):
        errors = []
        with patch("agent.workflow.importlib.import_module") as importer:
            worker = threading.Thread(
                target=lambda: self._capture_error(errors, initialize_gnani_plugin)
            )
            worker.start()
            worker.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("main thread", str(errors[0]))
        importer.assert_not_called()

    @staticmethod
    def _capture_error(errors, function):
        try:
            function()
        except RuntimeError as error:
            errors.append(error)

    def test_build_only_constructs_already_initialized_providers(self):
        LiveKitGnaniConfig._gnani_module = FakeGnani
        with patch("agent.workflow.importlib.import_module") as importer:
            providers = LiveKitGnaniConfig.build()
        importer.assert_not_called()
        self.assertEqual(providers["stt"].kwargs, {"language": "hi-IN", "use_streaming": True})
        self.assertEqual(providers["tts"].kwargs["model"], "timbre-v2.5")

    def test_server_registration_follows_main_thread_initialization(self):
        calls = []

        class FakeServer:
            def rtc_session(self, **_kwargs):
                def register(function):
                    calls.append("server_registered")
                    return function
                return register

        fake_agents = types.ModuleType("livekit.agents")
        fake_agents.AgentServer = FakeServer
        fake_agents.JobContext = object
        fake_livekit = types.ModuleType("livekit")
        fake_livekit.agents = fake_agents
        with patch.dict(sys.modules, {"livekit": fake_livekit, "livekit.agents": fake_agents}):
            with patch("agent.livekit_worker.initialize_gnani_plugin", side_effect=lambda: calls.append("gnani_initialized")):
                create_livekit_server()
        self.assertEqual(calls, ["gnani_initialized", "server_registered"])


if __name__ == "__main__":
    unittest.main()
