"""Authoritative Python answer verification and compliance verdicts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent.core import (
    CHECK_ORDER,
    JUDGE_CHECKS,
    MANDATORY_CHECKS,
    CheckResult,
    CheckVerdict,
    Flag,
    JudgeResult,
    evaluate_check,
    extract_number_for_check,
)
from agent.policy import Policy
from agent.session import ConsentStatus, RiskEvidence, SessionContext, SessionStatus


IDENTITY_CHECK_ID = "identity"
MANDATORY_VERIFICATION_CHECKS = (IDENTITY_CHECK_ID, *tuple(MANDATORY_CHECKS))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_confirmed_risk(flag: str | Flag, evidence: str) -> bool:
    """Confirm only critical risks backed by an approved Python pattern."""
    value = flag.value if isinstance(flag, Flag) else flag
    if value not in {Flag.GUARANTEE_CLAIM.value, Flag.SIGN_PRESSURE.value}:
        return False
    normalized = re.sub(r"\s+", " ", evidence.casefold()).strip()
    patterns = {
        Flag.GUARANTEE_CLAIM.value: ("guaranteed", "pakka return", "bank wale ne bola", "गारंटीड", "पक्का रिटर्न", "बैंक वाले ने बोला"),
        Flag.SIGN_PRESSURE.value: ("bas sign kar do", "bete ne bola", "agent ne bola kar do", "बस साइन कर दो", "बेटे ने बोला", "एजेंट ने बोला कर दो"),
    }
    return any(pattern in normalized for pattern in patterns[value])


def _quote(answer: str, token: str) -> str:
    index = answer.casefold().find(token.casefold())
    return answer[index:index + len(token)] if index >= 0 else answer


def _deterministic_judge(key: str, answer: str, policy: Policy) -> JudgeResult:
    normalized = answer.casefold()
    if key == "product_type":
        if any(term in normalized for term in ("fd", "fixed deposit", "फिक्स्ड डिपॉजिट", "एफडी")):
            return JudgeResult("incorrect", _quote(answer, "FD"), "customer described the product as a fixed deposit")
        if any(term in normalized for term in ("deferred annuity", "annuity", "pension", "पेंशन")):
            return JudgeResult("correct", _quote(answer, "pension"), "customer described a pension or annuity product")
        return JudgeResult("unclear", "", "product type was not clear")
    if key == "payout_recipient":
        recipient = str(policy.get("payout_recipient") or "")
        if recipient and recipient.casefold() in normalized:
            return JudgeResult("correct", recipient, "approved payout recipient was identified")
        if any(term in normalized for term in ("mujhe", "mere", "खुद", "मुझे", "myself")):
            return JudgeResult("incorrect", answer, "customer named a recipient different from the approved recipient")
        return JudgeResult("unclear", "", "payout recipient was not clear")
    if key == "stop_paying":
        if any(term in normalized for term in ("lapse", "benefit reduce", "benefits reduce", "kam ho", "कम हो", "लैप्स", "कम हो सकते")):
            return JudgeResult("correct", answer, "customer described an approved stopping-premium consequence")
        if any(term in normalized for term in ("kuch nahi", "कुछ नहीं", "continue", "जारी रहेगा", "same benefit")):
            return JudgeResult("incorrect", answer, "customer did not describe the approved consequence")
        return JudgeResult("unclear", "", "stopping-premium consequence was not clear")
    return JudgeResult("unclear", "", "check is not a deterministic free-text check")


@dataclass(frozen=True)
class VerdictReport:
    verdict: str
    hold_status: str
    reason_codes: tuple[str, ...]
    explanation: str
    mandatory_completed: tuple[str, ...]
    mandatory_outstanding: tuple[str, ...]
    confirmed_risks: tuple[str, ...]
    evidence_references: tuple[str, ...]
    calculated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "hold_status": self.hold_status,
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "mandatory_completed": list(self.mandatory_completed),
            "mandatory_outstanding": list(self.mandatory_outstanding),
            "confirmed_risks": list(self.confirmed_risks),
            "evidence_references": list(self.evidence_references),
            "calculated_at": self.calculated_at.isoformat(),
        }


class VerificationEngine:
    """Single Python authority for check evaluation and final verdicts."""

    def evaluate_answer(
        self,
        *,
        session: SessionContext,
        policy: Policy,
        check_id: str,
        answer_id: str | None = None,
        customer: Mapping[str, Any] | None = None,
    ) -> CheckResult:
        if check_id not in MANDATORY_VERIFICATION_CHECKS and check_id != "affordability":
            raise ValueError("unknown verification check")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError("cannot evaluate answers after handoff or termination")
        evidence = self._select_answer(session, check_id, answer_id)
        if session.pending_question is None or session.pending_question.verification_check_id != check_id:
            raise ValueError("answer is not associated with the current pending question")
        if check_id == IDENTITY_CHECK_ID:
            expected_names = (str(customer.get("name", "")), *(str(value) for value in customer.get("name_variants", ()))) if customer else ()
            matched = any(name and name.casefold() in evidence.original_answer.casefold() for name in expected_names)
            result = CheckResult(
                check_id,
                CheckVerdict.PASS if matched else CheckVerdict.FAIL,
                evidence.original_answer,
                True,
                reason="identity matched synthetic customer" if matched else "identity did not match synthetic customer",
            )
        elif check_id in JUDGE_CHECKS:
            result = evaluate_check(
                check_id,
                evidence.original_answer,
                policy,
                judge_result=_deterministic_judge(check_id, evidence.original_answer, policy),
            )
        else:
            result = evaluate_check(check_id, evidence.original_answer, policy)
        result.answer_id = evidence.answer_id
        session.update_verification_result(check_id, result, resolved=result.verdict in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY})
        session.update_answer_status(evidence.answer_id, result.verdict.value)
        return result

    @staticmethod
    def _select_answer(session: SessionContext, check_id: str, answer_id: str | None) -> Any:
        candidates = [answer for answer in session.answers if answer.verification_check_id == check_id]
        if answer_id is not None:
            candidates = [answer for answer in candidates if answer.answer_id == answer_id]
        if not candidates:
            raise ValueError("no recorded answer evidence exists for this check")
        return candidates[-1]

    def calculate_verdict(self, *, session: SessionContext) -> VerdictReport:
        checks: Mapping[str, CheckResult] = {
            key: value for key, value in session.verification_results.items()
            if isinstance(value, CheckResult)
        }
        completed = tuple(key for key in MANDATORY_VERIFICATION_CHECKS if checks.get(key, CheckResult(key, CheckVerdict.PENDING, "", True)).verdict in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY})
        outstanding = tuple(key for key in MANDATORY_VERIFICATION_CHECKS if key not in completed)
        confirmed = tuple(dict.fromkeys(risk.flag for risk in session.risk_flags if risk.confirmed))
        evidence = tuple(risk.message_id for risk in session.risk_flags if risk.confirmed and risk.message_id)
        reasons: list[str] = []

        if session.consent_status == ConsentStatus.REFUSED:
            reasons.append("consent_refused")
            return self._report("red", "hold", reasons, "Consent was refused, so the call requires human review.", completed, outstanding, confirmed, evidence)
        if Flag.GUARANTEE_CLAIM.value in confirmed:
            reasons.append("confirmed_guarantee_claim")
        if Flag.SIGN_PRESSURE.value in confirmed:
            reasons.append("confirmed_sign_pressure")
        if reasons:
            return self._report("red", "hold", reasons, "A confirmed critical compliance risk requires human review.", completed, outstanding, confirmed, evidence)

        if any(risk.flag == Flag.FD_CONFUSION.value for risk in session.risk_flags) and "product_type" in outstanding:
            reasons.append("unresolved_fd_confusion")
        if any(risk.flag == Flag.THIRD_PARTY.value for risk in session.risk_flags):
            reasons.append("third_party_review_required")
        if session.human_handoff:
            reasons.append("human_handoff")
        if session.end_call_requested:
            reasons.append("call_ended")
        if outstanding:
            reasons.extend(f"mandatory_{key}_outstanding" for key in outstanding)
        if any(checks.get(key) and checks[key].verdict == CheckVerdict.PASS_AFTER_RETRY for key in MANDATORY_VERIFICATION_CHECKS):
            reasons.append("mandatory_check_passed_after_retry")
        if reasons:
            return self._report("amber", "pending_review", reasons, "The call requires review because verification is incomplete or a non-critical risk remains unresolved.", completed, outstanding, confirmed, evidence)
        return self._report("green", "clear", ["all_mandatory_checks_passed"], "All mandatory checks passed without unresolved compliance concerns.", completed, outstanding, confirmed, evidence)

    def _report(self, verdict: str, hold_status: str, reasons: list[str], explanation: str, completed: tuple[str, ...], outstanding: tuple[str, ...], confirmed: tuple[str, ...], evidence: tuple[str, ...]) -> VerdictReport:
        return VerdictReport(verdict, hold_status, tuple(dict.fromkeys(reasons)), explanation, completed, outstanding, confirmed, evidence, _now())
