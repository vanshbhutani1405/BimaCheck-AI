"""Provider-independent per-call conversation orchestration.

Transport implementations provide async listen/speak/stop and event publishing;
the core CallFlow and evaluation functions retain authority over decisions.
"""
from __future__ import annotations

import importlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from agent.core import (
    CHECK_ORDER, MANDATORY_CHECKS, CallFlow, CheckResult, CheckVerdict, Flag,
    Judge, Stage, VerdictInput, compute_verdict, detect_flags,
    evaluate_with_retry,
)
from agent.session import ConsentStatus, SessionContext, SessionStatus


logger = logging.getLogger(__name__)


PROMPTS = {
    "disclosure": "Namaste. Main BimaCheck AI ka automated assistant hoon. Yeh call record ki jaayegi. Kya aap is baat-cheet ko jaari rakhna chahenge?",
    "identity": "Kripya apna naam batayein, taaki main policyholder ki pehchaan ki pushti kar sakoon.",
    "facts": "Aapki policy {product_name} hai. Har saal ka premium {premium_amount} rupaye hai, jo {premium_term_years} tak bharna hai. Policy {policy_term_years} tak chalegi. Kya yeh jaankari sahi hai?",
    "premium_amount": "Har saal aapko kitni rakam premium ke roop mein deni hai?",
    "premium_term": "Aapko premium kitne saal tak bharna hai?",
    "policy_term": "Policy kitne saal tak chalegi?",
    "product_type": "Kya yeh fixed deposit hai, ya pension policy?",
    "payout_recipient": "Policy ke mutabik paisa kisko milega?",
    "payout_timing": "Pension kab shuru hogi?",
    "stop_paying": "Agar aap premium dena rok dein, to aapke hisaab se kya hoga?",
    "affordability": "Kya yeh premium aapki aamdani ke hisaab se aasaan hai?",
    "retry": "Main ek baar phir aasaan shabdon mein samjhaata hoon: {explanation}. Ab aap apne shabdon mein phir se batayein.",
    "human": "Dhanyavaad. Is call ko human review ke liye bheja ja raha hai.",
    "wrap_green": "Dhanyavaad. Aapki jaankari review ke liye darj kar li gayi hai.",
}

REEXPLANATIONS = {
    "premium_amount": "record mein diya gaya annual premium",
    "premium_term": "premium har saal record mein diye gaye saalon tak dena hai",
    "policy_term": "policy record mein diye gaye poore term tak chalegi",
    "product_type": "policy record mein diya product type",
    "payout_recipient": "record mein diya payout recipient",
    "payout_timing": "record mein diya payout start samay",
    "stop_paying": "premium rokne ke baare mein record mein di gayi shart",
    "affordability": "premium aapki aamdani ke hisaab se aasaan hai ya nahi",
}

_SPOKEN_NUMBERS = {
    1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch", 6: "chhah",
    7: "saat", 8: "aath", 9: "nau", 10: "das", 11: "gyarah", 12: "barah",
    13: "terah", 14: "chaudah", 15: "pandrah", 16: "solah", 17: "satrah",
    18: "atharah", 19: "unnis", 20: "bees", 30: "tees", 40: "chaalis",
}


def format_policy_value(value: Any, *, kind: str) -> str:
    """Format a policy amount or duration as natural spoken Hinglish."""
    numeric = float(value)
    if not numeric.is_integer() and kind == "duration":
        return str(value)
    if kind == "duration":
        integer = int(numeric)
        return f"{_SPOKEN_NUMBERS.get(integer, str(integer))} saal"
    if kind != "amount":
        raise ValueError(f"unsupported policy value kind: {kind!r}")

    lakhs = numeric / 100_000
    fractional_lakh = {1.25: "sawa", 1.5: "dedh", 2.5: "dhai"}.get(lakhs)
    if fractional_lakh is not None:
        return f"{fractional_lakh} lakh"
    if numeric >= 100_000 and lakhs.is_integer():
        lakh_count = int(lakhs)
        return f"{_SPOKEN_NUMBERS.get(lakh_count, str(lakh_count))} lakh"
    if numeric.is_integer():
        integer = int(numeric)
        return _SPOKEN_NUMBERS.get(integer, str(integer))
    return str(value)


def re_explanation_for(key: str, policy: Mapping[str, Any]) -> str:
    """Build a spoken retry explanation from the supplied policy facts."""
    if key == "premium_amount":
        return f"record ke mutabik har saal ka premium {format_policy_value(policy['premium_amount'], kind='amount')} rupaye hai"
    if key == "premium_term":
        return f"premium har saal {format_policy_value(policy['premium_term_years'], kind='duration')} tak bharna hai"
    if key == "policy_term":
        return f"policy {format_policy_value(policy['policy_term_years'], kind='duration')} tak chalegi"
    if key == "payout_timing":
        return f"pension {format_policy_value(policy['payout_start_after_years'], kind='duration')} baad shuru hogi"
    if key == "stop_paying" and policy.get("stop_paying_explanation"):
        return f"record ke mutabik, {policy['stop_paying_explanation']}"
    return REEXPLANATIONS[key]


class VoiceIO(Protocol):
    async def speak(self, text: str) -> None: ...
    async def listen(self) -> str: ...
    async def stop_playback(self) -> None: ...
    async def publish(self, event: Mapping[str, Any]) -> None: ...


def normalize_tts_text(text: str) -> str:
    """Single normalization seam; transformations must be evidence-led."""
    return " ".join(text.strip().split())


def policy_question_response(text: str, policy: Mapping[str, Any]) -> str | None:
    """Return a deterministic answer for a supported policy question.

    This deliberately recognizes only question-shaped utterances. For example,
    ``pension FD`` remains an answer to the product-type check rather than
    becoming a detour. Missing policy fields produce a safe unavailable answer
    instead of an invented fact.
    """
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()

    def unavailable() -> str:
        return "Is baat ki jaankari policy record mein uplabdh nahi hai."

    if re.search(r"(?:\bfd\b|fixed deposit|फिक्स्ड डिपॉजिट|एफडी)", normalized) and (
        "kya" in normalized or "hai na" in normalized or "?" in normalized or "क्या" in normalized
    ):
        product_type = policy.get("product_type")
        if product_type is None:
            return unavailable()
        return f"Nahi, yeh fixed deposit nahi hai. Yeh {product_type} hai."

    if re.search(r"premium.*(?:kitna|कितना)|(?:kitna|कितना).*premium|कितना प्रीमियम", normalized):
        amount = policy.get("premium_amount")
        return unavailable() if amount is None else f"Record ke mutabik har saal ka premium {format_policy_value(amount, kind='amount')} rupaye hai."

    if re.search(r"premium.*(?:kitne saal|कितने साल)|(?:kitne saal|कितने साल).*premium|कितने साल.*भर", normalized):
        years = policy.get("premium_term_years")
        return unavailable() if years is None else f"Premium har saal {format_policy_value(years, kind='duration')} tak bharna hai."

    if re.search(r"policy.*(?:kitne saal|कितने साल)|(?:kitne saal|कितने साल).*policy", normalized):
        years = policy.get("policy_term_years")
        return unavailable() if years is None else f"Policy {format_policy_value(years, kind='duration')} tak chalegi."

    if re.search(r"(?:pension|payout|पेंशन|पेंशन|payout).*(?:kab|कब)|(?:kab|कब).*(?:shuru|शुरू|pension|पेंशन)", normalized):
        years = policy.get("payout_start_after_years")
        return unavailable() if years is None else f"Pension {format_policy_value(years, kind='duration')} baad shuru hogi."

    if re.search(r"(?:payout|paisa|पैसा|पेंशन).*(?:kisko|किसको|kise|किसे|kaun|कौन)|(?:kisko|किसको|kise|किसे).*(?:milega|मिलेगा)", normalized):
        recipient = policy.get("payout_recipient")
        return unavailable() if recipient is None else f"Policy record ke mutabik payout {recipient} ko milega."

    return None


@dataclass
class WorkflowResult:
    verdict: str
    hold_status: str
    checks: dict[str, CheckResult] = field(default_factory=dict)
    flags: set[Flag] = field(default_factory=set)
    consent_refused: bool = False


class CallWorkflow:
    def __init__(self, io: VoiceIO, customer: Mapping[str, Any], policy: Mapping[str, Any], *, judge: Judge | None = None, session_context: SessionContext | None = None):
        self.io, self.customer, self.policy, self.judge = io, customer, policy, judge
        self.session = session_context or SessionContext()
        self.flow = CallFlow()
        self.checks: dict[str, CheckResult] = {}
        self.flags: set[Flag] = set()
        self._last_customer_message_id: str | None = None
        self._last_answer_id: str | None = None
        self._answer_ids_by_check: dict[str, list[str]] = {}

    async def _say(self, line: str) -> None:
        logger.info("workflow_response_generated", extra={"text_length": len(line)})
        self.session.add_message("agent", line)
        await self.io.speak(normalize_tts_text(line))

    async def _answer(self) -> str:
        answer = await self.io.listen()
        logger.info("workflow_transcript_received", extra={"text_length": len(answer)})
        self._last_customer_message_id = self.session.add_message("customer", answer)
        await self.io.publish({"type": "transcript", "session_id": self.session.session_id, "role": "customer", "text": answer, "final": True, "ts_ms": 0})
        for flag, quote in detect_flags(answer):
            self.flags.add(flag)
            self.session.add_risk_flag(flag.value, quote, message_id=self._last_customer_message_id)
            await self.io.publish({"type": "flag", "session_id": self.session.session_id, "flag": flag.value, "quote": quote, "ts_ms": 0})
        return answer

    async def _answer_for_prompt(self, prompt: str, *, verification_check_id: str | None = None, attempt_number: int = 1) -> str:
        """Handle supported policy questions without changing pending state."""
        self.session.set_pending_question(prompt, verification_check_id=verification_check_id)
        while True:
            answer = await self._answer()
            response = policy_question_response(answer, self.policy)
            if response is None:
                if verification_check_id is not None:
                    self._last_answer_id = self.session.record_answer(
                        verification_check_id,
                        answer,
                        transcript_evidence=answer,
                        attempt_number=attempt_number,
                        message_id=self._last_customer_message_id,
                    )
                    self._answer_ids_by_check.setdefault(verification_check_id, []).append(self._last_answer_id)
                return answer
            await self._say(response)
            await self._say(prompt)

    async def run(self) -> WorkflowResult:
        if self.session.status == SessionStatus.CREATED:
            self.session.start()
        logger.info("workflow_stage_before", extra={"stage": self.flow.stage.value})
        self.flow.advance("connected")
        self.session.set_stage(self.flow.stage.value)
        logger.info("workflow_stage_after", extra={"stage": self.flow.stage.value})
        await self._say(PROMPTS["disclosure"])
        consent = (await self._answer_for_prompt(PROMPTS["disclosure"], verification_check_id="consent")).casefold()
        accepted = any(word in consent for word in ("haan", "ha", "yes", "sahmat", "हाँ", "जी"))
        refused = any(word in consent for word in ("nahi", "nahin", "no", "नहीं"))
        if not accepted or refused:
            self.session.consent_status = ConsentStatus.REFUSED
            if self.session.pending_question:
                self.session.resolve_pending_question(question_id=self.session.pending_question.question_id)
            self.flow.advance("consent", accepted=False)
            self.session.set_stage(self.flow.stage.value)
            await self._say(PROMPTS["human"])
            return await self._finish(consent_refused=True)
        self.flow.advance("consent", accepted=True)
        self.session.consent_status = ConsentStatus.GRANTED
        if self.session.pending_question:
            self.session.resolve_pending_question(question_id=self.session.pending_question.question_id)
        self.session.set_stage(self.flow.stage.value)
        logger.info("workflow_stage_after", extra={"stage": self.flow.stage.value})

        await self._say(PROMPTS["identity"])
        identity = await self._answer_for_prompt(PROMPTS["identity"], verification_check_id="identity")
        name = str(self.customer.get("name", ""))
        name_variants = self.customer.get("name_variants", ())
        expected_names = (name, *tuple(str(value) for value in name_variants))
        confirmed = any(candidate and candidate.casefold() in identity.casefold() for candidate in expected_names)
        self.flow.advance("identity", accepted=confirmed)
        if self._last_answer_id:
            self.session.update_answer_status(self._last_answer_id, "pass" if confirmed else "fail")
        if self.session.pending_question:
            self.session.resolve_pending_question(question_id=self.session.pending_question.question_id)
        self.session.set_stage(self.flow.stage.value)
        logger.info("workflow_identity_evaluated", extra={"accepted": confirmed})
        logger.info("workflow_stage_after", extra={"stage": self.flow.stage.value})
        if not confirmed:
            await self._say(PROMPTS["human"])
            return await self._finish(unresolved_failure=True)

        self.flow.advance("facts_confirmed")
        self.session.set_stage(self.flow.stage.value)
        logger.info("workflow_stage_after", extra={"stage": self.flow.stage.value})
        facts = PROMPTS["facts"].format(
            product_name=self.policy["product_name"],
            premium_amount=format_policy_value(self.policy["premium_amount"], kind="amount"),
            premium_term_years=format_policy_value(self.policy["premium_term_years"], kind="duration"),
            policy_term_years=format_policy_value(self.policy["policy_term_years"], kind="duration"),
        )
        await self._say(facts)
        facts_ack = (await self._answer_for_prompt(facts, verification_check_id="facts_confirmation")).casefold()
        if self.session.pending_question:
            self.session.resolve_pending_question(question_id=self.session.pending_question.question_id)
        if any(w in facts_ack for w in ("nahi", "nahin", "galat", "wrong", "नहीं")):
            await self._say(PROMPTS["human"])
            return await self._finish(unresolved_failure=True)

        for key in CHECK_ORDER:
            if self.flow.current_check != key:
                raise RuntimeError("core check order and workflow diverged")
            await self._say(PROMPTS[key])
            first = await self._answer_for_prompt(PROMPTS[key], verification_check_id=key, attempt_number=1)
            async def retry_answer() -> str:
                explanation = re_explanation_for(key, self.policy)
                retry_prompt = PROMPTS["retry"].format(explanation=explanation)
                await self._say(retry_prompt)
                return await self._answer_for_prompt(retry_prompt, verification_check_id=key, attempt_number=2)
            explanation = re_explanation_for(key, self.policy)
            result, _retry_prompt = await evaluate_with_retry(
                key, first, self.policy,
                re_explanation=explanation,
                get_retry=retry_answer,
                judge=self.judge,
            )
            logger.info("workflow_check_evaluated", extra={"check": key, "verdict": result.verdict.value, "retries": result.retries})
            self.checks[key] = result
            self.session.update_verification_result(key, result, resolved=result.verdict in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY})
            for answer_id in self._answer_ids_by_check.get(key, ()):
                self.session.update_answer_status(answer_id, result.verdict.value)
            if self.session.pending_question:
                self.session.resolve_pending_question(question_id=self.session.pending_question.question_id)
            await self.io.publish({"type": "check_update", "session_id": self.session.session_id, "key": key, "verdict": result.verdict.value, "heard": result.heard, "ts_ms": 0})
            self.flow.advance("check_passed")
            self.session.set_stage(self.flow.stage.value)
            logger.info("workflow_stage_after", extra={"stage": self.flow.stage.value, "check": key})

        verdict, hold = compute_verdict(VerdictInput(checks=self.checks, flags=self.flags))
        await self._say(PROMPTS["wrap_green"] if verdict == "green" else PROMPTS["human"])
        await self.io.publish({"type": "verdict", "session_id": self.session.session_id, "verdict": verdict, "reasons": [k for k, c in self.checks.items() if c.verdict not in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY}], "hold_status": hold})
        self.session.complete(human_handoff=verdict != "green")
        return WorkflowResult(verdict, hold, self.checks, self.flags)

    async def _finish(self, *, consent_refused: bool = False, unresolved_failure: bool = False) -> WorkflowResult:
        verdict, hold = compute_verdict(VerdictInput(checks=self.checks, flags=self.flags, consent_refused=consent_refused, unresolved_failure=unresolved_failure))
        await self.io.publish({"type": "verdict", "session_id": self.session.session_id, "verdict": verdict, "reasons": ["consent_refused"] if consent_refused else ["identity_or_fact_unresolved"], "hold_status": hold})
        self.session.complete(human_handoff=True)
        return WorkflowResult(verdict, hold, self.checks, self.flags, consent_refused)


class LiveKitGnaniConfig:
    """Gnani configuration with explicit main-thread plugin initialization."""

    _gnani_module: Any | None = None

    @classmethod
    def initialize(cls) -> Any:
        """Import Gnani once on the process main thread before job dispatch."""
        if cls._gnani_module is not None:
            return cls._gnani_module
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Gnani plugin initialization must run on the main thread")
        cls._gnani_module = importlib.import_module("livekit.plugins.gnani")
        return cls._gnani_module

    @classmethod
    def build(cls):
        """Construct providers from the module initialized before worker dispatch."""
        if cls._gnani_module is None:
            raise RuntimeError("call LiveKitGnaniConfig.initialize() before build()")
        gnani = cls._gnani_module
        return {
            "stt": gnani.STT(language="hi-IN", use_streaming=True),
            "tts": gnani.TTS(model="timbre-v2.5", language="hi-IN", synthesize_method="websocket"),
        }


class LiveKitIO:
    """Explicit adapter contract; bind concrete session playback/listen API at deployment."""
    def __init__(self, *, speak, listen, stop_playback, publish):
        self._speak, self._listen = speak, listen
        self._stop_playback, self._publish = stop_playback, publish

    async def speak(self, text: str) -> None:
        await self._speak(text)

    async def listen(self) -> str:
        return await self._listen()

    async def stop_playback(self) -> None:
        await self._stop_playback()

    async def publish(self, event: Mapping[str, Any]) -> None:
        await self._publish(dict(event))
