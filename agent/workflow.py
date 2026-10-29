"""Provider-independent per-call conversation orchestration.

Transport implementations provide async listen/speak/stop and event publishing;
the core CallFlow and evaluation functions retain authority over decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from agent.core import (
    CHECK_ORDER, MANDATORY_CHECKS, CallFlow, CheckResult, CheckVerdict, Flag,
    Judge, Stage, VerdictInput, compute_verdict, detect_flags,
    evaluate_with_retry,
)


PROMPTS = {
    "disclosure": "Namaste. Main BimaCheck AI ka automated assistant hoon. Yeh call record hogi. Kya aap baat jaari rakhne ke liye sahmat hain?",
    "identity": "Kripya apna naam batayein, taaki main policyholder ki pehchaan confirm kar sakoon.",
    "facts": "Aapki policy {product_name} hai. Premium {premium_amount} rupaye har saal, {premium_term_years} saal tak hai; policy term {policy_term_years} saal hai. Kya yeh jaankari sahi hai?",
    "premium_amount": "Aapke hisaab se har saal kitna premium dena hai?",
    "premium_term": "Aapko kitne saal tak premium bharna hai?",
    "policy_term": "Policy kitne saal chalegi?",
    "product_type": "Yeh FD hai ya kis tarah ki policy hai?",
    "payout_recipient": "Policy ke mutabik paisa kisko milega?",
    "payout_timing": "Payout ya pension kab shuru hogi?",
    "stop_paying": "Agar aap premium dena rok dein, to aapke samajh se kya hoga?",
    "affordability": "Kya yeh premium aapki aamdani ke hisaab se aasaan hai?",
    "retry": "Main ek baar phir aasaan shabdon mein samjhaata hoon: {explanation} Ab aap apne shabdon mein dobara batayein.",
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
}


class VoiceIO(Protocol):
    async def speak(self, text: str) -> None: ...
    async def listen(self) -> str: ...
    async def stop_playback(self) -> None: ...
    async def publish(self, event: Mapping[str, Any]) -> None: ...


def normalize_tts_text(text: str) -> str:
    """Single normalization seam; transformations must be evidence-led."""
    return " ".join(text.strip().split())


@dataclass
class WorkflowResult:
    verdict: str
    hold_status: str
    checks: dict[str, CheckResult] = field(default_factory=dict)
    flags: set[Flag] = field(default_factory=set)
    consent_refused: bool = False


class CallWorkflow:
    def __init__(self, io: VoiceIO, customer: Mapping[str, Any], policy: Mapping[str, Any], *, judge: Judge | None = None):
        self.io, self.customer, self.policy, self.judge = io, customer, policy, judge
        self.flow = CallFlow()
        self.checks: dict[str, CheckResult] = {}
        self.flags: set[Flag] = set()

    async def _say(self, line: str) -> None:
        await self.io.speak(normalize_tts_text(line))

    async def _answer(self) -> str:
        answer = await self.io.listen()
        await self.io.publish({"type": "transcript", "role": "customer", "text": answer, "final": True, "ts_ms": 0})
        for flag, quote in detect_flags(answer):
            self.flags.add(flag)
            await self.io.publish({"type": "flag", "flag": flag.value, "quote": quote, "ts_ms": 0})
        return answer

    async def run(self) -> WorkflowResult:
        self.flow.advance("connected")
        await self._say(PROMPTS["disclosure"])
        consent = (await self._answer()).casefold()
        accepted = any(word in consent for word in ("haan", "ha", "yes", "sahmat", "हाँ", "जी"))
        refused = any(word in consent for word in ("nahi", "nahin", "no", "नहीं"))
        if not accepted or refused:
            self.flow.advance("consent", accepted=False)
            await self._say(PROMPTS["human"])
            return await self._finish(consent_refused=True)
        self.flow.advance("consent", accepted=True)

        await self._say(PROMPTS["identity"])
        identity = await self._answer()
        name = str(self.customer.get("name", ""))
        confirmed = bool(name and name.casefold() in identity.casefold())
        self.flow.advance("identity", accepted=confirmed)
        if not confirmed:
            await self._say(PROMPTS["human"])
            return await self._finish(unresolved_failure=True)

        self.flow.advance("facts_confirmed")
        facts = PROMPTS["facts"].format(**self.policy)
        await self._say(facts)
        facts_ack = (await self._answer()).casefold()
        if any(w in facts_ack for w in ("nahi", "nahin", "galat", "wrong", "नहीं")):
            await self._say(PROMPTS["human"])
            return await self._finish(unresolved_failure=True)

        for key in CHECK_ORDER:
            if self.flow.current_check != key:
                raise RuntimeError("core check order and workflow diverged")
            await self._say(PROMPTS[key])
            first = await self._answer()
            async def retry_answer() -> str:
                await self._say(PROMPTS["retry"].format(explanation=REEXPLANATIONS[key]))
                return await self._answer()
            result, _retry_prompt = await evaluate_with_retry(
                key, first, self.policy,
                re_explanation=REEXPLANATIONS.get(key, ""),
                get_retry=retry_answer,
                judge=self.judge,
            )
            self.checks[key] = result
            await self.io.publish({"type": "check_update", "key": key, "verdict": result.verdict.value, "heard": result.heard, "ts_ms": 0})
            self.flow.advance("check_passed")

        verdict, hold = compute_verdict(VerdictInput(checks=self.checks, flags=self.flags))
        await self._say(PROMPTS["wrap_green"] if verdict == "green" else PROMPTS["human"])
        await self.io.publish({"type": "verdict", "verdict": verdict, "reasons": [k for k, c in self.checks.items() if c.verdict not in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY}], "hold_status": hold})
        return WorkflowResult(verdict, hold, self.checks, self.flags)

    async def _finish(self, *, consent_refused: bool = False, unresolved_failure: bool = False) -> WorkflowResult:
        verdict, hold = compute_verdict(VerdictInput(checks=self.checks, flags=self.flags, consent_refused=consent_refused, unresolved_failure=unresolved_failure))
        await self.io.publish({"type": "verdict", "verdict": verdict, "reasons": ["consent_refused"] if consent_refused else ["identity_or_fact_unresolved"], "hold_status": hold})
        return WorkflowResult(verdict, hold, self.checks, self.flags, consent_refused)


class LiveKitGnaniConfig:
    """Lazy constructors based on installed Gnani plugin 1.8.10 metadata/API docs."""
    @staticmethod
    def build():
        from livekit.plugins import gnani
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
