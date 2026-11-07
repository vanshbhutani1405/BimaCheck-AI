import asyncio
import unittest

from agent.core import CHECK_ORDER, Stage
from agent.workflow import (
    CallWorkflow,
    LiveKitIO,
    PROMPTS,
    format_policy_value,
    normalize_tts_text,
    policy_question_response,
)


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
        self.assertTrue(any("dedh lakh" in line for line in io.spoken))
        self.assertNotEqual(result.verdict, "green")

    def test_barge_in_adapter_contract_and_normalization(self):
        calls = []
        async def invoke(): calls.append("called")
        io = LiveKitIO(speak=invoke, listen=lambda: invoke(), stop_playback=invoke, publish=lambda _: invoke())
        asyncio.run(io.stop_playback())
        self.assertEqual(calls, ["called"])
        self.assertEqual(normalize_tts_text("  Namaste   ji "), "Namaste ji")

    def test_policy_value_formatting_is_spoken_friendly(self):
        self.assertEqual(format_policy_value(150000, kind="amount"), "dedh lakh")
        self.assertEqual(format_policy_value(250000, kind="amount"), "dhai lakh")
        self.assertEqual(format_policy_value(5, kind="duration"), "paanch saal")
        self.assertEqual(format_policy_value(20, kind="duration"), "bees saal")
        self.assertEqual(format_policy_value(30, kind="duration"), "tees saal")

    def test_facts_prompt_passed_to_tts_uses_spoken_policy_values(self):
        io = FakeIO(["haan", "Asha", "nahi"])
        asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(
            io.spoken[2],
            "Aapki policy Suraksha Pension Plus hai. Har saal ka premium dedh lakh rupaye hai, jo paanch saal tak bharna hai. Policy bees saal tak chalegi. Kya yeh jaankari sahi hai?",
        )
        self.assertNotIn("150000", io.spoken[2])
        self.assertNotIn("20 saal", io.spoken[2])

    def test_supported_policy_questions_use_only_policy_data(self):
        questions = {
            "Yeh FD hai kya?": "Nahi, yeh fixed deposit nahi hai. Yeh deferred annuity hai.",
            "Premium kitna dena hai?": "Record ke mutabik har saal ka premium dedh lakh rupaye hai.",
            "Premium kitne saal bharna hai?": "Premium har saal paanch saal tak bharna hai.",
            "Policy kitne saal chalegi?": "Policy bees saal tak chalegi.",
            "Pension kab shuru hogi?": "Pension paanch saal baad shuru hogi.",
            "Paisa kisko milega?": "Policy record ke mutabik payout Asha ko milega.",
        }
        for question, expected in questions.items():
            with self.subTest(question=question):
                self.assertEqual(policy_question_response(question, self.policy), expected)

    def test_policy_question_repeats_exact_pending_prompt_without_state_change(self):
        io = FakeIO(["Premium kitna dena hai?", "dedh lakh"])
        workflow = CallWorkflow(io, {"name": "Asha"}, self.policy)
        pending = PROMPTS["premium_amount"]

        answer = asyncio.run(workflow._answer_for_prompt(pending))

        self.assertEqual(answer, "dedh lakh")
        self.assertEqual(io.spoken, [
            "Record ke mutabik har saal ka premium dedh lakh rupaye hai.",
            pending,
        ])
        self.assertEqual(workflow.flow.stage, Stage.CONNECT)
        self.assertEqual(workflow.flow.check_index, 0)
        self.assertEqual(workflow.checks, {})
        self.assertEqual(len([event for event in io.events if event["type"] == "transcript"]), 2)

    def test_policy_question_does_not_consume_retry_in_full_check(self):
        replies = [
            "haan", "Asha", "haan", "Premium kitna dena hai?", "1.5 lakh",
            "5 saal", "20 saal", "pension plan", "pension plan retry",
            "Asha", "Asha retry", "5 saal", "policy jaari rahegi",
            "continue retry", "yes",
        ]
        io = FakeIO(replies)
        result = asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(result.checks["premium_amount"].retries, 0)
        self.assertEqual(result.checks["premium_amount"].verdict.value, "pass")
        self.assertEqual(io.spoken[5], PROMPTS["premium_amount"])

    def test_natural_number_answer_passes_without_retry(self):
        replies = [
            "haan", "Asha", "haan", "mera premium dedh lakh hai",
            "paanch saal tak dena hai", "policy bees saal chalegi",
            "pension plan", "pension plan retry", "Asha", "Asha retry",
            "5 saal", "policy jaari rahegi", "continue retry", "yes",
        ]
        io = FakeIO(replies)
        result = asyncio.run(CallWorkflow(io, {"name": "Asha"}, self.policy).run())
        self.assertEqual(result.checks["premium_amount"].verdict.value, "pass")
        self.assertEqual(result.checks["premium_amount"].retries, 0)

    def test_consent_and_identity_use_separate_transcripts(self):
        io = FakeIO(["हाँ", "वंश", "नहीं"])
        result = asyncio.run(
            CallWorkflow(
                io,
                {"name": "Vansh", "name_variants": ("वंश",)},
                self.policy,
            ).run()
        )
        self.assertFalse(result.consent_refused)
        self.assertTrue(any("Suraksha Pension Plus" in line for line in io.spoken))
        self.assertEqual(result.verdict, "red")

    def test_identity_accepts_only_declared_roman_or_devanagari_variant(self):
        for identity in ("Vansh", "वंश"):
            with self.subTest(identity=identity):
                io = FakeIO(["हाँ", identity, "नहीं"])
                result = asyncio.run(
                    CallWorkflow(
                        io,
                        {"name": "Vansh", "name_variants": ("वंश",)},
                        self.policy,
                    ).run()
                )
                self.assertFalse(result.consent_refused)
                self.assertTrue(any("Suraksha Pension Plus" in line for line in io.spoken))

    def test_unsupported_policy_question_is_not_answered_or_advanced(self):
        question = "Surrender value kitni milegi?"
        self.assertIsNone(policy_question_response(question, self.policy))
        io = FakeIO([question])
        workflow = CallWorkflow(io, {"name": "Asha"}, self.policy)
        answer = asyncio.run(workflow._answer_for_prompt(PROMPTS["premium_amount"]))
        self.assertEqual(answer, question)
        self.assertEqual(io.spoken, [])
        self.assertEqual(workflow.flow.stage, Stage.CONNECT)
        self.assertEqual(workflow.flow.check_index, 0)


if __name__ == "__main__":
    unittest.main()
