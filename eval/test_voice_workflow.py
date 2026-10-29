import asyncio
import unittest

from agent.core import CHECK_ORDER, Stage
from agent.workflow import CallWorkflow, LiveKitIO, normalize_tts_text


class FakeIO:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.spoken = []
        self.events = []
        self.stopped = 0

    async def speak(self, text): self.spoken.append(text)
    async def listen(self): return next(self.replies)
    async def stop_playback(self): self.stopped += 1
    async def publish(self, event): self.events.append(dict(event))


def valid_replies(name="Asha"):
    # The three judge-only checks have no provider in these workflow tests,
    # so each receives its one required retry answer after the first response.
    return ["haan", name, "haan", "1.5 lakh", "5 saal", "20 saal", "pension plan", "pension plan retry", "Asha", "Asha retry", "5 saal", "policy jaari rahegi", "continue retry", "yes"]


class WorkflowTests(unittest.TestCase):
    policy = {
        "product_name": "Suraksha Pension Plus", "premium_amount": 150000,
        "premium_term_years": 5, "policy_term_years": 20,
        "payout_start_after_years": 5, "product_type": "deferred annuity",
        "payout_recipient": "Asha", "declared_income": 600000,
    }

    def test_complete_progression_fixed_prompts_and_event_shapes(self):
        io = FakeIO(valid_replies())
        result = asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(result.verdict, "amber")  # unavailable free-text judge remains unclear
        questions = [line for line in io.spoken if line in {__import__('agent.workflow', fromlist=['PROMPTS']).PROMPTS[k] for k in CHECK_ORDER}]
        self.assertEqual(questions, [__import__('agent.workflow', fromlist=['PROMPTS']).PROMPTS[k] for k in CHECK_ORDER])
        self.assertEqual([e["type"] for e in io.events].count("transcript"), 14)
        self.assertEqual([e["type"] for e in io.events].count("check_update"), len(CHECK_ORDER))
        self.assertEqual(sum("ek baar phir" in line for line in io.spoken), 3)
        self.assertEqual(sum(result.checks[key].retries for key in CHECK_ORDER), 3)
        for event in io.events:
            self.assertIn(event["type"], {"transcript", "check_update", "flag", "verdict"})
        self.assertEqual(io.events[-1]["type"], "verdict")

    def test_consent_refusal_routes_human_and_stops(self):
        io = FakeIO(["nahi"])
        result = asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(result.verdict, "red")
        self.assertEqual(result.hold_status, "hold")
        self.assertTrue(result.consent_refused)
        self.assertEqual(list(io.replies), [])
        self.assertEqual(io.events[-1]["type"], "verdict")

    def test_one_retry_and_flags_are_emitted(self):
        replies = ["haan", "Asha", "haan", "wrong", "1.5 lakh", "5 saal", "20 saal", "pension FD", "retry product", "Asha", "retry recipient", "5 saal", "stop", "retry stop", "yes"]
        io = FakeIO(replies)
        result = asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(result.checks["premium_amount"].retries, 1)
        self.assertLessEqual(result.checks["premium_amount"].retries, 1)
        self.assertEqual(len([e for e in io.events if e["type"] == "transcript"]), 15)
        self.assertTrue(all(check.retries <= 1 for check in result.checks.values()))
        self.assertTrue(any(e["type"] == "flag" and e["flag"] == "FD_CONFUSION" for e in io.events))
        self.assertNotEqual(result.verdict, "green")

    def test_barge_in_adapter_contract_and_normalization(self):
        calls = []
        async def invoke(): calls.append("called")
        io = LiveKitIO(speak=invoke, listen=lambda: invoke(), stop_playback=invoke, publish=lambda _: invoke())
        asyncio.run(io.stop_playback())
        self.assertEqual(calls, ["called"])
        self.assertEqual(normalize_tts_text("  Namaste   ji "), "Namaste ji")


if __name__ == "__main__":
    unittest.main()
