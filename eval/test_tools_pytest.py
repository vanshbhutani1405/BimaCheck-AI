from __future__ import annotations

from agent.conversation_agent import ToolProposal
from agent.core import Flag
from agent.tools import ToolExecutor


def test_record_answer_requires_pending_question_and_python_evaluation(bound_policy, customer, session):
    executor = ToolExecutor()
    message_id = session.add_message("customer", "डेढ़ लाख")
    result = executor.execute(
        tool_name="record_answer",
        session=session,
        policy=bound_policy,
        customer=customer,
        arguments={
            "check_id": "premium_amount",
            "answer_candidate": "150000",
            "original_transcript": "डेढ़ लाख",
            "message_id": message_id,
        },
    )
    assert result.success is False
    session.set_pending_question("Premium?", verification_check_id="premium_amount")
    result = executor.execute(
        tool_name="record_answer",
        session=session,
        policy=bound_policy,
        customer=customer,
        arguments={
            "check_id": "premium_amount",
            "answer_candidate": "150000",
            "original_transcript": "डेढ़ लाख",
            "message_id": message_id,
        },
    )
    assert result.success is True
    assert result.data["verification_status"] == "pass"
    assert session.answers[0].verification_status == "pass"


def test_tools_reject_unknown_invalid_and_post_terminal_requests(session, bound_policy):
    executor = ToolExecutor()
    unknown = executor.execute(tool_name="not_approved", session=session, arguments={})
    assert unknown.success is False and unknown.error_code == "unknown_tool"
    session.request_handoff("review required")
    rejected = executor.execute(
        tool_name="flag_risk", session=session,
        arguments={"flag": Flag.GUARANTEE_CLAIM.value, "evidence": "guaranteed"},
        policy=bound_policy,
    )
    assert rejected.success is False


def test_get_fact_requires_one_selector_and_tracks_agent_disclosure(session, policy):
    executor = ToolExecutor()
    no_selector = executor.execute(tool_name="get_policy_fact", session=session, policy=policy, arguments={})
    assert no_selector.success is False
    agent_message = session.add_message("agent", "Premium dedh lakh hai")
    result = executor.execute(
        tool_name="get_policy_fact", session=session, policy=policy,
        arguments={"fact_id": "premium.amount", "used_in_response": True, "response_message_id": agent_message},
    )
    assert result.success is True
    assert "premium.amount" in session.disclosed_policy_facts


def test_risk_evidence_is_confirmed_only_by_python_pattern(session):
    executor = ToolExecutor()
    message_id = session.add_message("customer", "Agent said guaranteed return")
    result = executor.execute(
        tool_name="flag_risk", session=session,
        arguments={"flag": Flag.GUARANTEE_CLAIM.value, "evidence": "guaranteed return", "message_id": message_id},
    )
    assert result.success and result.data["confirmed"] is True
    assert session.risk_flags[0].message_id == message_id


def test_handoff_and_end_call_require_active_session_and_confirmation(session):
    executor = ToolExecutor()
    handoff = executor.execute(tool_name="handoff_to_human", session=session, arguments={"reason": "help"})
    assert handoff.success and session.human_handoff
    other = type(session)(); other.start()
    end = executor.execute(tool_name="end_call", session=other, arguments={"reason": "done", "confirmation": False})
    assert end.success is False


def test_execute_proposal_uses_allowlist(bound_policy, session):
    result = ToolExecutor().execute_proposal(
        proposal=ToolProposal("unknown", {}), session=session, policy=bound_policy
    )
    assert result.success is False and result.error_code == "unknown_tool"
