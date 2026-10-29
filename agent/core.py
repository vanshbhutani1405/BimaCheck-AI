"""Pure, deterministic verification rules for BimaCheck AI."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping


class Stage(str, Enum):
    CONNECT = "connect"
    CONSENT = "consent"
    IDENTITY = "identity"
    FACT_CONFIRMATION = "fact_confirmation"
    COMPREHENSION = "comprehension"
    WRAP_UP = "wrap_up"
    ENDED = "ended"


CHECK_ORDER = ("premium_amount", "premium_term", "policy_term", "product_type", "payout_recipient", "payout_timing", "stop_paying", "affordability")
MANDATORY_CHECKS = frozenset(CHECK_ORDER[:-1])
JUDGE_CHECKS = frozenset({"product_type", "payout_recipient", "stop_paying"})


@dataclass
class CallFlow:
    stage: Stage = Stage.CONNECT
    check_index: int = 0
    consent_given: bool | None = None
    identity_confirmed: bool | None = None

    @property
    def current_check(self) -> str | None:
        if self.stage != Stage.COMPREHENSION:
            return None
        return CHECK_ORDER[self.check_index] if self.check_index < len(CHECK_ORDER) else None

    def advance(self, event: str, *, accepted: bool | None = None) -> Stage:
        transitions = {
            (Stage.CONNECT, "connected"): Stage.CONSENT,
            (Stage.CONSENT, "consent"): Stage.IDENTITY if accepted else Stage.ENDED,
            (Stage.IDENTITY, "identity"): Stage.FACT_CONFIRMATION if accepted else Stage.ENDED,
            (Stage.FACT_CONFIRMATION, "facts_confirmed"): Stage.COMPREHENSION,
        }
        if self.stage == Stage.COMPREHENSION and event == "check_passed":
            if self.current_check is None:
                raise ValueError("all checks are already complete")
            self.check_index += 1
            if self.current_check is None:
                self.stage = Stage.WRAP_UP
            return self.stage
        next_stage = transitions.get((self.stage, event))
        if next_stage is None:
            raise ValueError(f"invalid event {event!r} during {self.stage.value}")
        if self.stage == Stage.CONSENT:
            self.consent_given = bool(accepted)
        if self.stage == Stage.IDENTITY:
            self.identity_confirmed = bool(accepted)
        self.stage = next_stage
        return self.stage


_ONES = {"shunya": 0, "zero": 0, "ek": 1, "एक": 1, "do": 2, "दो": 2, "teen": 3, "तीन": 3, "char": 4, "चार": 4, "chaar": 4, "paanch": 5, "पांच": 5, "पाँच": 5, "cheh": 6, "छह": 6, "saat": 7, "सात": 7, "aath": 8, "आठ": 8, "nau": 9, "नौ": 9}
_TENS = {"das": 10, "दस": 10, "gyarah": 11, "ग्यारह": 11, "barah": 12, "बारह": 12, "terah": 13, "तेरह": 13, "chaudah": 14, "चौदह": 14, "pandrah": 15, "पंद्रह": 15, "solah": 16, "सोलह": 16, "satrah": 17, "सत्रह": 17, "atharah": 18, "अठारह": 18, "unnis": 19, "उन्नीस": 19, "bees": 20, "बीस": 20, "tees": 30, "तीस": 30, "chalis": 40, "चालीस": 40, "paintalis": 45, "पैंतालीस": 45, "pachaas": 50, "पचास": 50, "saath": 60, "साठ": 60, "sattar": 70, "सत्तर": 70, "assi": 80, "अस्सी": 80, "nabbe": 90, "नब्बे": 90}
_SCALE = {"hazaar": 1_000, "हज़ार": 1_000, "हजार": 1_000, "lakh": 100_000, "लाख": 100_000, "crore": 10_000_000, "करोड़": 10_000_000}
_FRACTIONS = {"dedh": 1.5, "डेढ़": 1.5, "dhai": 2.5, "ढाई": 2.5, "sawa": 1.25, "सवा": 1.25}


class NumberParseError(ValueError):
    pass


def parse_number(text: str) -> int | float:
    """Parse a supported Hindi/Hinglish number expression; reject ambiguity."""
    raw = text.strip().lower().replace(",", "")
    if not raw:
        raise NumberParseError("empty number")
    raw = re.sub(r"\s+", " ", raw)
    words = raw.split()
    if words[-1] in {"saal", "साल", "varsh", "वर्ष", "rupees", "रुपये", "रुपया"}:
        words.pop()
    if not words:
        raise NumberParseError("number is missing")
    if len(words) == 1 and re.fullmatch(r"\d+(?:\.\d+)?", words[0]):
        value = float(words[0]) if "." in words[0] else int(words[0])
        return int(value) if isinstance(value, float) and value.is_integer() else value

    total = 0.0
    quantity = 0.0
    has_quantity = False
    saw_magnitude = False
    previous_scale = float("inf")
    for word in words:
        if word in _SCALE:
            scale = _SCALE[word]
            if not has_quantity or scale >= previous_scale:
                raise NumberParseError("incomplete or repeated magnitude")
            total += quantity * scale
            quantity = 0.0
            has_quantity = False
            saw_magnitude = True
            previous_scale = scale
            continue
        if has_quantity:
            raise NumberParseError("ambiguous adjacent numeric quantities")
        if word in _FRACTIONS:
            quantity = _FRACTIONS[word]
        elif re.fullmatch(r"\d+(?:\.\d+)?", word):
            quantity = float(word)
        elif word in _ONES:
            quantity = _ONES[word]
        elif word in _TENS:
            quantity = _TENS[word]
        else:
            raise NumberParseError(f"unsupported or ambiguous token: {word!r}")
        has_quantity = True
    if not has_quantity and not saw_magnitude:
        raise NumberParseError("incomplete magnitude")
    result = total + (quantity if has_quantity else 0)
    return int(result) if float(result).is_integer() else result


class Flag(str, Enum):
    FD_CONFUSION = "FD_CONFUSION"
    GUARANTEE_CLAIM = "GUARANTEE_CLAIM"
    SIGN_PRESSURE = "SIGN_PRESSURE"
    THIRD_PARTY = "THIRD_PARTY"


DEFAULT_FLAG_RULES: Mapping[Flag, tuple[str, ...]] = {
    Flag.FD_CONFUSION: (r"\bfd\b", "fixed deposit", "ek hi baar", "एक ही बार", "फिक्स्ड डिपॉजिट"),
    Flag.GUARANTEE_CLAIM: ("guaranteed", "pakka return", "bank wale ne bola", "गारंटीड", "पक्का रिटर्न", "बैंक वाले ने बोला"),
    Flag.SIGN_PRESSURE: ("bas sign kar do", "bete ne bola", "agent ne bola kar do", "बस साइन कर दो", "बेटे ने बोला", "एजेंट ने बोला कर दो"),
    Flag.THIRD_PARTY: ("main unka beta bol raha", "main unki beti bol rahi", "main policyholder nahi", "मैं उनका बेटा बोल रहा", "मैं उनकी बेटी बोल रही", "मैं पॉलिसीधारक नहीं"),
}


def detect_flags(text: str, rules: Mapping[Flag, tuple[str, ...]] = DEFAULT_FLAG_RULES) -> list[tuple[Flag, str]]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    found = []
    for flag, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                found.append((flag, text))
                break
    return found


class CheckVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCLEAR = "unclear"
    PASS_AFTER_RETRY = "pass_after_retry"


@dataclass
class CheckResult:
    key: str
    verdict: CheckVerdict
    heard: str
    mandatory: bool
    retries: int = 0
    reason: str = ""


def evaluate_check(key: str, answer: str, policy: Mapping[str, Any], *, judge_result: "JudgeResult | None" = None) -> CheckResult:
    mandatory = key in MANDATORY_CHECKS
    if not answer.strip():
        return CheckResult(key, CheckVerdict.UNCLEAR, answer, mandatory, reason="empty answer")
    fields = {"premium_amount": "premium_amount", "premium_term": "premium_term_years", "policy_term": "policy_term_years", "payout_timing": "payout_start_after_years"}
    if key in fields:
        try:
            heard = parse_number(answer)
            expected = policy[fields[key]]
            if heard == expected:
                return CheckResult(key, CheckVerdict.PASS, answer, mandatory)
            return CheckResult(key, CheckVerdict.FAIL, answer, mandatory, reason=f"expected {expected}")
        except (NumberParseError, KeyError, TypeError):
            return CheckResult(key, CheckVerdict.UNCLEAR, answer, mandatory, reason="could not compare numeric answer")
    if key == "affordability":
        try:
            income = float(policy["declared_income"])
            premium = float(policy["premium_amount"])
            return CheckResult(key, CheckVerdict.PASS if premium <= income else CheckVerdict.FAIL, answer, False, reason="affordability warning" if premium > income else "")
        except (KeyError, TypeError, ValueError):
            return CheckResult(key, CheckVerdict.UNCLEAR, answer, False, reason="required policy data unavailable")
    if key not in JUDGE_CHECKS or judge_result is None:
        return CheckResult(key, CheckVerdict.UNCLEAR, answer, mandatory, reason="validated judge result required")
    verdicts = {"correct": CheckVerdict.PASS, "incorrect": CheckVerdict.FAIL, "unclear": CheckVerdict.UNCLEAR}
    verdict = verdicts.get(judge_result.verdict, CheckVerdict.UNCLEAR)
    if verdict == CheckVerdict.PASS and (not judge_result.quote or judge_result.quote not in answer):
        verdict = CheckVerdict.UNCLEAR
    return CheckResult(key, verdict, answer, mandatory, reason=judge_result.reason)


async def evaluate_with_retry(key: str, first_answer: str, policy: Mapping[str, Any], *, re_explanation: str, get_retry: Callable[[], Awaitable[str]], judge: "Judge | None" = None) -> tuple[CheckResult, str | None]:
    async def evaluate(answer: str) -> CheckResult:
        if key in JUDGE_CHECKS:
            if judge is None:
                return evaluate_check(key, answer, policy)
            expected_key = {"product_type": "product_type", "payout_recipient": "payout_recipient", "stop_paying": "stop_paying_explanation"}[key]
            try:
                judge_result = await judge.judge(key, str(policy[expected_key]), answer)
            except (KeyError, ValueError, TypeError):
                judge_result = JudgeResult("unclear", "", "judge input unavailable")
            return evaluate_check(key, answer, policy, judge_result=judge_result)
        return evaluate_check(key, answer, policy)

    result = await evaluate(first_answer)
    if result.verdict == CheckVerdict.PASS:
        return result, None
    retry_answer = await get_retry()
    retry = await evaluate(retry_answer)
    retry.retries = 1
    if retry.verdict == CheckVerdict.PASS:
        retry.verdict = CheckVerdict.PASS_AFTER_RETRY
    return retry, re_explanation


@dataclass(frozen=True)
class JudgeResult:
    verdict: str
    quote: str
    reason: str


def parse_judge_response(raw: str, answer: str) -> JudgeResult:
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("judge response is not valid JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"verdict", "quote", "reason"}:
        raise ValueError("judge JSON must contain exactly verdict, quote, reason")
    if obj["verdict"] not in {"correct", "incorrect", "unclear"} or not all(isinstance(obj[k], str) for k in ("quote", "reason")):
        raise ValueError("invalid judge fields")
    if obj["quote"] and obj["quote"] not in answer:
        raise ValueError("judge quote must be copied from the answer")
    return JudgeResult(**obj)


class Judge:
    """Thin async boundary. Caller supplies provider function; no provider SDK here."""
    def __init__(self, request: Callable[[str, str, str], Awaitable[str]], timeout: float = 1.5):
        self.request, self.timeout = request, timeout

    async def judge(self, key: str, expected: str, answer: str) -> JudgeResult:
        if key not in JUDGE_CHECKS:
            raise ValueError("judge is restricted to designated free-text checks")
        try:
            raw = await asyncio.wait_for(self.request(key, expected, answer), timeout=self.timeout)
            return parse_judge_response(raw, answer)
        except (asyncio.TimeoutError, ValueError, TypeError):
            return JudgeResult("unclear", "", "judge timeout or invalid response")

@dataclass
class VerdictInput:
    checks: Mapping[str, CheckResult]
    flags: set[Flag] = field(default_factory=set)
    unresolved_flags: set[Flag] = field(default_factory=set)
    consent_refused: bool = False
    low_confidence: bool = False
    unresolved_failure: bool = False


def compute_verdict(data: VerdictInput) -> tuple[str, str]:
    if data.consent_refused or data.unresolved_flags or any(data.checks.get(k, CheckResult(k, CheckVerdict.UNCLEAR, "", True)).verdict == CheckVerdict.FAIL for k in MANDATORY_CHECKS) or data.unresolved_failure or data.flags:
        return "red", "hold"
    if any(data.checks.get(k, CheckResult(k, CheckVerdict.UNCLEAR, "", True)).verdict in {CheckVerdict.UNCLEAR, CheckVerdict.PASS_AFTER_RETRY} for k in MANDATORY_CHECKS) or data.low_confidence:
        return "amber", "pending_review"
    return "green", "clear"
