from __future__ import annotations

import pytest

from agent.session import SessionContext, SessionStatus


def test_sessions_have_isolated_ids_and_state(customer):
    first, second = SessionContext(), SessionContext()
    first.start()
    first.add_message("customer", customer["name"])
    assert first.session_id != second.session_id
    assert first.conversation_history
    assert second.conversation_history == []
    assert second.answers == []


def test_history_is_bounded_by_recent_messages(session):
    ids = [session.add_message("customer", str(index)) for index in range(20)]
    assert [message.message_id for message in session.recent_messages(12)] == ids[-12:]
    with pytest.raises(ValueError):
        session.recent_messages(0)


def test_pending_question_cannot_be_replaced_by_another_check(pending_session):
    session, question = pending_session
    assert session.preserve_pending_question() is question
    with pytest.raises(ValueError):
        session.set_pending_question("Policy term?", verification_check_id="policy_term")


def test_pending_question_resolves_and_clears(pending_session):
    session, question = pending_session
    session.resolve_pending_question(question_id=question.question_id)
    assert session.pending_question is None
    assert "premium_amount" not in session.pending_checks


def test_answer_and_risk_evidence_preserve_message_references(session):
    customer_message = session.add_message("customer", "डेढ़ लाख")
    answer_id = session.record_answer(
        "premium_amount", "डेढ़ लाख", attempt_number=1, message_id=customer_message
    )
    session.add_risk_flag("FD_CONFUSION", "fixed deposit", message_id=customer_message)
    assert session.answers[0].answer_id == answer_id
    assert session.answers[0].message_id == customer_message
    assert session.risk_flags[0].message_id == customer_message
    assert session.has_message(customer_message, speaker="customer")


def test_handoff_and_end_call_are_terminal(session):
    session.request_handoff("customer requested help")
    assert session.status == SessionStatus.HUMAN_HANDOFF
    with pytest.raises(ValueError):
        session.request_end_call("too late")

    other = SessionContext()
    other.start()
    other.request_end_call("customer requested end")
    assert other.status == SessionStatus.COMPLETED
    assert other.end_call_requested is True


def test_final_verdict_storage_is_idempotent(session):
    first = session.store_final_verdict({"verdict": "amber"})
    second = session.store_final_verdict({"verdict": "green"})
    assert first == second == {"verdict": "amber"}
