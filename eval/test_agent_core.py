import asyncio
import json
import unittest

from agent.core import (
    CHECK_ORDER, CheckResult, CheckVerdict, CallFlow, Flag, Judge, JudgeResult, Stage,
    VerdictInput, compute_verdict, detect_flags, evaluate_check,
    evaluate_with_retry, parse_judge_response,
    parse_number,
)


class NumberParserTests(unittest.TestCase):
    def test_supported_utterances(self):
        cases = [
            ("0", 0), ("1", 1), ("1.5", 1.5), ("ek", 1), ("एक", 1),
            ("do", 2), ("दो", 2), ("teen", 3), ("तीन", 3),
            ("char", 4), ("chaar", 4), ("चार", 4), ("paanch", 5), ("पाँच", 5),
            ("cheh", 6), ("छह", 6), ("saat", 7), ("सात", 7), ("aath", 8), ("आठ", 8),
            ("nau", 9), ("नौ", 9), ("das", 10), ("gyarah", 11), ("barah", 12),
            ("terah", 13), ("chaudah", 14), ("pandrah", 15), ("solah", 16),
            ("satrah", 17), ("atharah", 18), ("unnis", 19), ("bees", 20),
            ("tees", 30), ("chalis", 40), ("paintalis", 45), ("pachaas", 50),
            ("saath", 60), ("sattar", 70), ("assi", 80), ("nabbe", 90),
            ("dedh", 1.5), ("डेढ़", 1.5), ("dhai", 2.5), ("ढाई", 2.5),
            ("sawa", 1.25), ("सवा", 1.25), ("hazaar", None), ("ek hazaar", 1000),
            ("do hazaar", 2000), ("paanch lakh", 500000), ("सात लाख", 700000),
            ("ek crore", 10000000), ("dedh lakh", 150000), ("dhai lakh", 250000),
            ("sawa lakh", 125000), ("1 lakh", 100000), ("1.5 lakh", 150000),
            ("1 lakh 50 hazaar", 150000), ("do lakh teen hazaar", 203000),
            ("bees saal", 20), ("paintalis", 45), ("पैंतालीस", 45),
        ]
        self.assertGreaterEqual(len(cases), 60)
        for utterance, expected in cases:
            with self.subTest(utterance=utterance):
                if expected is None:
                    with self.assertRaises(ValueError):
                        parse_number(utterance)
                else:
                    self.assertEqual(parse_number(utterance), expected)

    def test_ambiguous_or_invalid(self):
        for utterance in ("", "kal", "dedh bees", "lakh", "1 twenty", "bees bees"):
            with self.subTest(utterance=utterance), self.assertRaises(ValueError):
                parse_number(utterance)


class FlagTests(unittest.TestCase):
    def test_positives_roman_and_devanagari(self):
        samples = {
            Flag.FD_CONFUSION: ["FD jaisa hai", "fixed deposit", "एक ही बार", "फिक्स्ड डिपॉजिट"],
            Flag.GUARANTEE_CLAIM: ["guaranteed return", "pakka return", "पक्का रिटर्न"],
            Flag.SIGN_PRESSURE: ["bas sign kar do", "bete ne bola", "बस साइन कर दो"],
            Flag.THIRD_PARTY: ["main unka beta bol raha", "मैं उनकी बेटी बोल रही"],
        }
        for flag, utterances in samples.items():
            for utterance in utterances:
                with self.subTest(flag=flag, utterance=utterance):
                    self.assertIn(flag, {f for f, _ in detect_flags(utterance)})

    def test_negative(self):
        for utterance in ("premium paanch saal bharna hai", "policy mujhe samajh aa gayi", "namaste ji"):
            self.assertEqual(detect_flags(utterance), [])


class FlowTests(unittest.TestCase):
    def test_fixed_order_and_consent(self):
        flow = CallFlow()
        self.assertEqual(flow.advance("connected"), Stage.CONSENT)
        self.assertEqual(flow.advance("consent", accepted=True), Stage.IDENTITY)
        self.assertEqual(flow.advance("identity", accepted=True), Stage.FACT_CONFIRMATION)
        self.assertEqual(flow.advance("facts_confirmed"), Stage.COMPREHENSION)
        self.assertEqual(flow.current_check, CHECK_ORDER[0])
        for expected in CHECK_ORDER:
            self.assertEqual(flow.current_check, expected)
            flow.advance("check_passed")
        self.assertEqual(flow.stage, Stage.WRAP_UP)

    def test_refusal_ends_call_and_bad_transition_rejected(self):
        flow = CallFlow()
        with self.assertRaises(ValueError):
            flow.advance("identity", accepted=True)
        flow.advance("connected")
        self.assertEqual(flow.advance("consent", accepted=False), Stage.ENDED)


class JudgeAndVerdictTests(unittest.TestCase):
    def test_labelled_judge_fixtures(self):
        fixtures = [
            ('{"verdict":"correct","quote":"pension","reason":"matches"}', "Pension milegi", "correct"),
            ('{"verdict":"incorrect","quote":"FD","reason":"wrong product"}', "FD hai", "incorrect"),
            ('{"verdict":"unclear","quote":"shayad","reason":"uncertain"}', "shayad", "unclear"),
        ]
        for raw, answer, verdict in fixtures:
            self.assertEqual(parse_judge_response(raw, answer).verdict, verdict)
        for raw in ("{}", "not json", json.dumps({"verdict":"correct","quote":"invented","reason":"ok"})):
            with self.assertRaises(ValueError):
                parse_judge_response(raw, "answer")

    def test_timeout_and_malformed_are_unclear(self):
        async def request(*_):
            return "oops"
        judge = Judge(request)
        self.assertEqual(asyncio.run(judge.judge("product_type", "annuity", "FD")).verdict, "unclear")
        async def slow(*_):
            await asyncio.sleep(0.02)
            return '{}'
        self.assertEqual(asyncio.run(Judge(slow, timeout=0.001).judge("stop_paying", "", "idk")).verdict, "unclear")

    def test_async_judge_result_is_awaited_and_evaluated(self):
        async def request(key, expected, answer):
            self.assertEqual((key, expected, answer), ("product_type", "deferred annuity", "pension plan"))
            return json.dumps({"verdict": "correct", "quote": "pension plan", "reason": "matches"})
        policy = {"product_type": "deferred annuity"}
        async def run():
            result, prompt = await evaluate_with_retry(
                "product_type", "pension plan", policy,
                re_explanation="Please explain the plan type.",
                get_retry=lambda: self.fail("passing first answer must not retry"),
                judge=Judge(request),
            )
            return result, prompt
        result, prompt = asyncio.run(run())
        self.assertEqual(result.verdict, CheckVerdict.PASS)
        self.assertIsNone(prompt)

    def test_invalid_judge_results_never_pass(self):
        policy = {"product_type": "annuity"}
        for jr in (JudgeResult("unclear", "", "timeout"), JudgeResult("maybe", "x", "invalid verdict"), JudgeResult("correct", "fabricated", "bad quote"), JudgeResult("correct", "", "missing quote")):
            self.assertNotEqual(evaluate_check("product_type", "FD", policy, judge_result=jr).verdict, CheckVerdict.PASS)

    def test_retry_occurs_once_and_returns_prompt_not_playback_claim(self):
        calls = []
        async def request_retry():
            calls.append("retry")
            return "unknown"
        policy = {"premium_term_years": 5}
        result, prompt = asyncio.run(evaluate_with_retry(
            "premium_term", "one year", policy,
            re_explanation="The record says five years.", get_retry=request_retry,
        ))
        self.assertEqual(calls, ["retry"])
        self.assertEqual(result.retries, 1)
        self.assertEqual(result.verdict, CheckVerdict.UNCLEAR)
        self.assertEqual(prompt, "The record says five years.")

    def test_non_green_safety(self):
        for checks, flags, unresolved, consent, low, failure, expected in [
            ({}, set(), set(), False, False, False, "amber"),
            ({"premium_amount": CheckResult("premium_amount", CheckVerdict.FAIL, "wrong", True)}, set(), set(), False, False, False, "red"),
            ({}, {Flag.FD_CONFUSION}, set(), False, False, False, "red"),
            ({}, set(), {Flag.THIRD_PARTY}, False, False, False, "red"),
            ({}, set(), set(), True, False, False, "red"),
            ({}, set(), set(), False, True, False, "amber"),
            ({}, set(), set(), False, False, True, "red"),
        ]:
            result, _ = compute_verdict(VerdictInput(checks, flags, unresolved, consent, low, failure))
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
