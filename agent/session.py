"""Per-call session state for BimaCheck AI.

This module deliberately contains state and state transitions only.  It does
not know about LiveKit, an LLM provider, policy storage, or verdict rules.
Those concerns remain in their existing boundaries and future callers must
change this state through the methods below rather than mutating dictionaries
returned by an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    HUMAN_HANDOFF = "human_handoff"
    CLOSED = "closed"


class ConsentStatus(str, Enum):
    UNKNOWN = "unknown"
    GRANTED = "granted"
    REFUSED = "refused"


@dataclass
class PendingQuestion:
    question_id: str
    verification_check_id: str | None
    text: str
    created_at: datetime = field(default_factory=_utc_now)
    answered: bool = False
    clarification_required: bool = False


@dataclass(frozen=True)
class ConversationMessage:
    message_id: str
    speaker: str
    text: str
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AnswerEvidence:
    answer_id: str
    verification_check_id: str
    original_answer: str
    normalized_answer: str | None
    transcript_evidence: str
    timestamp: datetime
    attempt_number: int
    verification_status: str
    message_id: str | None = None


@dataclass(frozen=True)
class RiskEvidence:
    flag: str
    evidence: str
    timestamp: datetime
    message_id: str | None = None
    reason: str | None = None


@dataclass
class SessionContext:
    """Authoritative, isolated state for one customer call."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=_utc_now)
    status: SessionStatus = SessionStatus.CREATED
    current_stage: str = "connect"
    completed: bool = False
    human_handoff: bool = False
    consent_status: ConsentStatus = ConsentStatus.UNKNOWN
    pending_question: PendingQuestion | None = None
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    answers: list[AnswerEvidence] = field(default_factory=list)
    verification_results: dict[str, Any] = field(default_factory=dict)
    completed_checks: set[str] = field(default_factory=set)
    pending_checks: set[str] = field(default_factory=set)
    retry_counts: dict[str, int] = field(default_factory=dict)
    unresolved_checks: set[str] = field(default_factory=set)
    risk_flags: list[RiskEvidence] = field(default_factory=list)
    disclosed_policy_facts: set[str] = field(default_factory=set)
    _answer_index: dict[str, AnswerEvidence] = field(default_factory=dict, repr=False)

    def start(self) -> None:
        if self.status != SessionStatus.CREATED:
            raise ValueError(f"cannot start session in {self.status.value} state")
        self.status = SessionStatus.ACTIVE

    def set_stage(self, stage: str) -> None:
        if not stage:
            raise ValueError("stage must not be empty")
        if self.status in {SessionStatus.COMPLETED, SessionStatus.CLOSED}:
            raise ValueError("cannot change stage after session completion")
        self.current_stage = stage

    def complete(self, *, human_handoff: bool = False) -> None:
        if self.status == SessionStatus.CLOSED:
            raise ValueError("cannot complete a closed session")
        if self.status == SessionStatus.COMPLETED:
            if self.human_handoff != human_handoff:
                raise ValueError("completed session handoff status cannot change")
            return
        self.completed = True
        self.human_handoff = human_handoff
        self.status = SessionStatus.HUMAN_HANDOFF if human_handoff else SessionStatus.COMPLETED
        self.pending_question = None

    def close(self) -> None:
        if self.status == SessionStatus.CLOSED:
            return
        self.completed = True
        self.status = SessionStatus.CLOSED
        self.pending_question = None

    def set_pending_question(self, text: str, *, verification_check_id: str | None = None, question_id: str | None = None) -> PendingQuestion:
        if not text.strip():
            raise ValueError("pending question text must not be empty")
        if self.pending_question is not None and not self.pending_question.answered:
            if verification_check_id != self.pending_question.verification_check_id:
                raise ValueError("cannot overwrite an unanswered pending question")
            return self.pending_question
        question = PendingQuestion(
            question_id=question_id or str(uuid4()),
            verification_check_id=verification_check_id,
            text=text,
        )
        self.pending_question = question
        if verification_check_id is not None:
            self.pending_checks.add(verification_check_id)
        return question

    def get_pending_question(self) -> PendingQuestion | None:
        return self.pending_question

    def preserve_pending_question(self) -> PendingQuestion | None:
        """Return the pending question without changing it during a side turn."""
        return self.pending_question

    def recent_messages(self, limit: int = 12) -> tuple[ConversationMessage, ...]:
        if limit < 1:
            raise ValueError("history limit must be positive")
        return tuple(self.conversation_history[-limit:])

    def request_clarification(self, *, question_id: str) -> None:
        pending = self._require_pending(question_id)
        if pending.answered:
            raise ValueError("cannot request clarification for an answered question")
        pending.clarification_required = True

    def resolve_pending_question(self, *, question_id: str, answered: bool = True, clarification_required: bool = False) -> None:
        pending = self._require_pending(question_id)
        pending.answered = answered
        pending.clarification_required = clarification_required
        if answered and not clarification_required:
            if pending.verification_check_id is not None:
                self.pending_checks.discard(pending.verification_check_id)
            self.pending_question = None

    def clear_pending_question(self, *, question_id: str) -> None:
        pending = self._require_pending(question_id)
        if not pending.answered or pending.clarification_required:
            raise ValueError("pending question is not genuinely resolved")
        self.pending_question = None

    def _require_pending(self, question_id: str) -> PendingQuestion:
        if self.pending_question is None or self.pending_question.question_id != question_id:
            raise ValueError("pending question does not match the supplied question ID")
        return self.pending_question

    def add_message(self, speaker: str, text: str, *, metadata: Mapping[str, Any] | None = None) -> str:
        if speaker not in {"customer", "agent"}:
            raise ValueError("speaker must be 'customer' or 'agent'")
        if not isinstance(text, str):
            raise TypeError("message text must be a string")
        message_id = str(uuid4())
        self.conversation_history.append(
            ConversationMessage(message_id, speaker, text, _utc_now(), dict(metadata or {}))
        )
        return message_id

    def record_answer(
        self,
        verification_check_id: str,
        original_answer: str,
        *,
        normalized_answer: str | None = None,
        transcript_evidence: str | None = None,
        attempt_number: int = 1,
        verification_status: str = "unverified",
        message_id: str | None = None,
    ) -> str:
        if not verification_check_id:
            raise ValueError("verification_check_id must not be empty")
        if attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        answer = AnswerEvidence(
            answer_id=str(uuid4()),
            verification_check_id=verification_check_id,
            original_answer=original_answer,
            normalized_answer=normalized_answer,
            transcript_evidence=original_answer if transcript_evidence is None else transcript_evidence,
            timestamp=_utc_now(),
            attempt_number=attempt_number,
            verification_status=verification_status,
            message_id=message_id,
        )
        self.answers.append(answer)
        self._answer_index[answer.answer_id] = answer
        self.retry_counts[verification_check_id] = max(
            self.retry_counts.get(verification_check_id, 0), attempt_number - 1
        )
        return answer.answer_id

    def update_answer_status(self, answer_id: str, status: str) -> None:
        try:
            self._answer_index[answer_id].verification_status = status
        except KeyError as exc:
            raise ValueError("unknown answer ID") from exc

    def update_verification_result(self, check_id: str, result: Any, *, resolved: bool) -> None:
        self.verification_results[check_id] = result
        if resolved:
            self.completed_checks.add(check_id)
            self.unresolved_checks.discard(check_id)
        else:
            self.unresolved_checks.add(check_id)
            self.completed_checks.discard(check_id)

    def add_risk_flag(self, flag: str, evidence: str, *, message_id: str | None = None, reason: str | None = None) -> None:
        if not flag:
            raise ValueError("flag must not be empty")
        self.risk_flags.append(RiskEvidence(flag, evidence, _utc_now(), message_id, reason))

    def disclose_policy_fact(self, fact_id: str) -> None:
        if not fact_id:
            raise ValueError("fact_id must not be empty")
        self.disclosed_policy_facts.add(fact_id)
