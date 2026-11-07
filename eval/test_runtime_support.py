import unittest

from agent.runtime_support import AudioCache, TurnTimer, audio_cache_key


class RuntimeSupportTests(unittest.TestCase):
    def test_audio_cache_key_normalizes_text_and_separates_synthesis_options(self):
        common = {"voice": "voice-a", "model": "timbre-v2.5", "language": "hi-IN"}
        self.assertEqual(
            audio_cache_key("  Namaste   ji ", **common),
            audio_cache_key("Namaste ji", **common),
        )
        self.assertNotEqual(
            audio_cache_key("Namaste ji", **common),
            audio_cache_key("Namaste ji", voice="voice-b", model="timbre-v2.5", language="hi-IN"),
        )

    def test_audio_cache_only_accepts_approved_audio_and_returns_copies(self):
        cache = AudioCache()
        with self.assertRaises(ValueError):
            cache.put_approved("line", b"audio", listener_verdict="wrong")
        cache.put_approved("line", b"audio", listener_verdict="ok")
        self.assertEqual(cache.get("line"), b"audio")
        self.assertIsNone(cache.get("missing"))

    def test_turn_timing_reports_speech_to_transcript_and_first_audio(self):
        marks = iter((10.0, 10.2, 10.5))
        timer = TurnTimer(clock=lambda: next(marks))
        timer.transcript_final()
        measured = timer.first_audio(cached=False)
        self.assertAlmostEqual(measured.transcript_final_ms, 200.0)
        self.assertAlmostEqual(measured.response_first_audio_ms, 500.0)
        self.assertFalse(measured.cached_playback)

    def test_missing_final_transcript_is_reported_as_missing_not_zero_latency(self):
        marks = iter((4.0, 4.6))
        timer = TurnTimer(clock=lambda: next(marks))
        measured = timer.first_audio(cached=True)
        self.assertIsNone(measured.transcript_final_ms)
        self.assertAlmostEqual(measured.response_first_audio_ms, 600.0)
        self.assertTrue(measured.cached_playback)


if __name__ == "__main__":
    unittest.main()
