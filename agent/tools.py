"""Validated Python tools for the BimaCheck conversational agent.

Tools mutate only the explicitly supplied per-call session.  They do not
perform verification, alter verdicts, call providers, or close a transport.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from agent.conversation_agent import ToolProposal
from agent.compliance import IDENTITY_CHECK_ID, VerificationEngine, is_confirmed_risk
from agent.core import CHECK_ORDER, Flag
from agent.policy import Policy, PolicyFactResult, get_policy_fact as retrieve_policy_fact
from agent.session import SessionContext, SessionStatus


logger = logging.getLogger(__name__)
_CHECK_IDS = frozenset((IDENTITY_CHECK_ID, *CHECK_ORDER))
_RISK_FLAGS = frozenset(flag.value for flag in Flag)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    session_id: str
    success: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "session_id": self.session_id,
            "success": self.success,
            "data": dict(self.data),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _success(tool_name: str, session: SessionContext, **data: Any) -> ToolExecutionResult:
    return ToolExecutionResult(tool_name, session.session_id, True, data=dict(data))


def _failure(tool_name: str, session: SessionContext, code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(tool_name, session.session_id, False, error_code=code, error_message=message)


def _require_active(tool_name: str, session: SessionContext) -> None:
    if session.status != SessionStatus.ACTIVE:
        raise ValueError(f"session is not active: {session.status.value}")


def _optional_message(session: SessionContext, message_id: Any, *, speaker: str = "customer") -> str | None:
    if message_id is None:
        return None
    if not isinstance(message_id, str) or not session.has_message(message_id, speaker=speaker):
        raise ValueError("message_id must reference a message in this session")
    return message_id


def record_answer(
    *,
    session: SessionContext,
    check_id: str,
    answer_candidate: str,
    original_transcript: str,
    policy: Policy | None = None,
    customer: Mapping[str, Any] | None = None,
    message_id: str | None = None,
    normalized_answer: str | None = None,
    attempt_number: int = 1,
) -> ToolExecutionResult:
    """Record unverified answer evidence for the current pending question."""
    tool_name = "record_answer"
    try:
        _require_active(tool_name, session)
        if check_id not in _CHECK_IDS:
            raise ValueError("check_id is not an existing verification check")
        if check_id == IDENTITY_CHECK_ID and customer is None:
            raise ValueError("customer context is required to evaluate identity")
        if not isinstance(answer_candidate, str) or not answer_candidate.strip():
            raise ValueError("answer_candidate must be a non-empty string")
        if not isinstance(original_transcript, str) or not original_transcript.strip():
            raise ValueError("original_transcript must be a non-empty string")
        if normalized_answer is not None and not isinstance(normalized_answer, str):
            raise ValueError("normalized_answer must be a string or null")
        if not isinstance(attempt_number, int) or isinstance(attempt_number, bool) or attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        reference = _optional_message(session, message_id)
        pending = session.get_pending_question()
        if pending is None or pending.verification_check_id != check_id:
            raise ValueError("answer does not correspond to the current pending verification question")

        for evidence in session.answers:
            if (
                evidence.verification_check_id == check_id
                and evidence.attempt_number == attempt_number
                and evidence.original_answer == original_transcript
                and evidence.message_id == reference
            ):
                return _success(tool_name, session, recorded=False, duplicate=True, answer_id=evidence.answer_id, verification_status=evidence.verification_status)

        answer_id = session.record_answer(
            check_id,
            original_transcript,
            normalized_answer=normalized_answer or answer_candidate,
            transcript_evidence=original_transcript,
            attempt_number=attempt_number,
            verification_status="unverified",
            message_id=reference,
        )
        if policy is not None:
            result = VerificationEngine().evaluate_answer(
                session=session,
                policy=policy,
                check_id=check_id,
                answer_id=answer_id,
            )
            return _success(tool_name, session, recorded=True, duplicate=False, answer_id=answer_id, verification_status=result.verdict.value)
        return _success(tool_name, session, recorded=True, duplicate=False, answer_id=answer_id, verification_status="unverified")
    except Exception as exc:
        logger.warning("tool_execution_rejected", extra={"tool": tool_name, "reason": str(exc)})
        return _failure(tool_name, session, "invalid_request", str(exc))


def get_policy_fact(
    *,
    session: SessionContext,
    policy: Policy,
    fact_id: str | None = None,
    section: str | None = None,
    verification_check_id: str | None = None,
    used_in_response: bool = False,
    response_message_id: str | None = None,
) -> ToolExecutionResult:
    """Retrieve an approved policy fact without changing verification state."""
    tool_name = "get_policy_fact"
    try:
        _require_active(tool_name, session)
        if not isinstance(policy, Policy):
            raise ValueError("policy must be a validated Policy instance")
        selectors = [value for value in (fact_id, section, verification_check_id) if value is not None]
        if len(selectors) != 1:
            raise ValueError("provide exactly one fact_id, section or verification_check_id")
        if not all(isinstance(value, str) and value.strip() for value in selectors):
            raise ValueError("policy selectors must be non-empty strings")
        if not isinstance(used_in_response, bool):
            raise ValueError("used_in_response must be boolean")
        reference = _optional_message(session, response_message_id, speaker="agent") if response_message_id is not None else None
        result: PolicyFactResult = retrieve_policy_fact(
            policy,
            fact_id=fact_id,
            section=section,
            verification_check_id=verification_check_id,
        )
        if not result.exists:
            return _success(tool_name, session, fact=result.as_dict())
        if used_in_response and reference is None:
            raise ValueError("response_message_id is required when used_in_response is true")
        if used_in_response and result.fact_id is not None:
            session.disclose_policy_fact(result.fact_id, message_id=reference)
        return _success(tool_name, session, fact=result.as_dict(), disclosed=bool(used_in_response))
    except Exception as exc:
        logger.warning("tool_execution_rejected", extra={"tool": tool_name, "reason": str(exc)})
        return _failure(tool_name, session, "invalid_request", str(exc))


def flag_risk(
    *,
    session: SessionContext,
    flag: str,
    evidence: str,
    message_id: str | None = None,
    reason: str | None = None,
) -> ToolExecutionResult:
    """Record a validated risk signal with customer evidence."""
    tool_name = "flag_risk"
    try:
        _require_active(tool_name, session)
        if flag not in _RISK_FLAGS:
            raise ValueError("flag is not an approved risk category")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("evidence must be a non-empty customer statement")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        reference = _optional_message(session, message_id)
        if reference is None and not evidence.strip():
            raise ValueError("supporting customer evidence is required")
        for existing in session.risk_flags:
            if existing.flag == flag and existing.evidence == evidence and existing.message_id == reference:
                return _success(tool_name, session, recorded=False, duplicate=True, flag=flag)
        confirmed = is_confirmed_risk(flag, evidence)
        session.add_risk_flag(
            flag,
            evidence,
            message_id=reference,
            reason=reason or ("approved_python_pattern" if confirmed else "proposal_requires_compliance_review"),
            confirmed=confirmed,
        )
        return _success(tool_name, session, recorded=True, duplicate=False, flag=flag, confirmed=confirmed)
    except Exception as exc:
        logger.warning("tool_execution_rejected", extra={"tool": tool_name, "reason": str(exc)})
        return _failure(tool_name, session, "invalid_request", str(exc))


def handoff_to_human(*, session: SessionContext, reason: str, message_id: str | None = None) -> ToolExecutionResult:
    """Record a human handoff in session state without contacting a backend."""
    tool_name = "handoff_to_human"
    try:
        _require_active(tool_name, session)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("handoff reason must be a non-empty string")
        reference = _optional_message(session, message_id, speaker="customer") if message_id is not None else None
        session.request_handoff(reason.strip(), message_id=reference)
        return _success(tool_name, session, accepted=True, status=session.status.value, reason=reason.strip())
    except Exception as exc:
        logger.warning("tool_execution_rejected", extra={"tool": tool_name, "reason": str(exc)})
        return _failure(tool_name, session, "invalid_request", str(exc))


def end_call(*, session: SessionContext, reason: str, confirmation: bool, message_id: str | None = None) -> ToolExecutionResult:
    """Record a terminal end-call request without closing LiveKit transport."""
    tool_name = "end_call"
    try:
        _require_active(tool_name, session)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("end-call reason must be a non-empty string")
        if confirmation is not True:
            raise ValueError("confirmation must be true to end the call")
        reference = _optional_message(session, message_id, speaker="customer") if message_id is not None else None
        session.request_end_call(reason.strip(), message_id=reference)
        return _success(tool_name, session, accepted=True, status=session.status.value, reason=reason.strip())
    except Exception as exc:
        logger.warning("tool_execution_rejected", extra={"tool": tool_name, "reason": str(exc)})
        return _failure(tool_name, session, "invalid_request", str(exc))


ToolFunction = Callable[..., ToolExecutionResult]


class ToolExecutor:
    """Stateless allowlisted executor for Gemini tool proposals."""

    _tools: Mapping[str, ToolFunction] = {
        "record_answer": record_answer,
        "get_policy_fact": get_policy_fact,
        "flag_risk": flag_risk,
        "handoff_to_human": handoff_to_human,
        "end_call": end_call,
    }

    def execute(self, *, tool_name: str, session: SessionContext, arguments: Mapping[str, Any], policy: Policy | None = None, customer: Mapping[str, Any] | None = None) -> ToolExecutionResult:
        if not isinstance(tool_name, str) or tool_name not in self._tools:
            return _failure(tool_name if isinstance(tool_name, str) else "unknown", session, "unknown_tool", "tool is not approved")
        if not isinstance(arguments, Mapping):
            return _failure(tool_name, session, "invalid_arguments", "tool arguments must be an object")
        kwargs = dict(arguments)
        if tool_name == "get_policy_fact":
            if policy is None:
                return _failure(tool_name, session, "missing_policy", "validated policy is required")
            if "policy" in kwargs:
                return _failure(tool_name, session, "invalid_arguments", "policy is supplied by the executor")
            kwargs["policy"] = policy
        elif tool_name == "record_answer":
            if policy is None:
                return _failure(tool_name, session, "missing_policy", "validated policy is required")
            if "policy" in kwargs:
                return _failure(tool_name, session, "invalid_arguments", "policy is supplied by the executor")
            if "customer" in kwargs:
                return _failure(tool_name, session, "invalid_arguments", "customer context is supplied by the executor")
            kwargs["policy"] = policy
            if customer is not None:
                kwargs["customer"] = customer
            elif kwargs.get("check_id") == IDENTITY_CHECK_ID:
                return _failure(tool_name, session, "missing_customer", "customer context is required for identity")
        try:
            return self._tools[tool_name](session=session, **kwargs)
        except TypeError:
            logger.warning("tool_argument_validation_failed", extra={"tool": tool_name})
            return _failure(tool_name, session, "invalid_arguments", "tool arguments do not match the approved interface")
        except Exception:
            logger.exception("tool_execution_failed", extra={"tool": tool_name})
            return _failure(tool_name, session, "execution_error", "tool execution failed")

    def execute_proposal(self, *, proposal: ToolProposal, session: SessionContext, policy: Policy | None = None, customer: Mapping[str, Any] | None = None) -> ToolExecutionResult:
        if not isinstance(proposal, ToolProposal):
            return _failure("unknown", session, "invalid_proposal", "tool proposal is invalid")
        return self.execute(tool_name=proposal.name, session=session, arguments=proposal.arguments, policy=policy, customer=customer)
