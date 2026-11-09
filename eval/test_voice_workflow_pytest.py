from __future__ import annotations

import asyncio
import json

import pytest

from agent.conversation_agent import ConversationAction, ConversationIntent, ConversationResponse, ToolProposal
from agent.livekit_worker import LiveKitAgentSessionPort, run_conversational_session
from agent.voice_workflow import VOICE_PROMPTS, ConversationalCallWorkflow


def response(intent=ConversationIntent.ANSWER, candidate=None, action=ConversationAction.PROPOSE_RECORD_ANSWER, tool_proposal=None, spoken="जी, धन्यवाद।", handoff=False, end_call=False):
    return ConversationResponse(
        spoken_response=spoken,
        intent=intent,
        conversation_action=action,
        tool_proposal=tool_proposal,
        requires_clarification=intent != ConversationIntent.ANSWER,
        clarification_reason=None,
        answer_candidate=candidate,
        policy_fact_ids=(),
        handoff_requested=handoff,
        end_call_requested=end_call,
    )


class FakeIO:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.spoken = []
        self.events = []

    async def listen(self):
        return next(self.replies)

    async def speak(self, text):
        self.spoken.append(text)

    async def publish(self, event):
        self.events.append(dict(event))


class FakeConversationAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def respond(self, **kwargs):
        pending = kwargs["session"].pending_question
        self.calls.append({**kwargs, "pending_check": pending.verification_check_id if pending else None})
        return next(self.responses)


def enter_comprehension(workflow):
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)
    workflow.flow.advance("identity", accepted=True)
    workflow.flow.advance("facts_confirmed")


@pytest.mark.asyncio
async def test_one_check_records_evidence_and_advances_python_workflow(bound_policy, customer, session):
    io = FakeIO(["डेढ़ लाख"])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, FakeConversationAgent([response(candidate="डेढ़ लाख")]))
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert session.verification_results["premium_amount"].verdict.value == "pass"
    assert session.answers[0].original_answer == "डेढ़ लाख"
    assert session.pending_question is None
    assert workflow.flow.current_check == "premium_term"
    assert io.events[0]["final"] is True


@pytest.mark.asyncio
async def test_unrelated_gemini_tool_proposal_cannot_redirect_current_check(bound_policy, customer, session):
    io = FakeIO(["डेढ़ लाख"])
    agent = FakeConversationAgent([
        response(
            candidate="डेढ़ लाख",
            tool_proposal=ToolProposal("get_policy_fact", {"verification_check_id": "policy_term"}),
        )
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)

    await workflow._process_check("premium_amount")

    assert session.verification_results["premium_amount"].verdict.value == "pass"
    assert "policy_term" not in session.verification_results
    assert workflow.flow.current_check == "premium_term"


@pytest.mark.asyncio
async def test_consent_turn_cannot_record_identity_answer(bound_policy, customer, session):
    io = FakeIO(["जी हाँ।"])
    agent = FakeConversationAgent([
        response(
            candidate="जी हाँ।",
            tool_proposal=ToolProposal("record_answer", {"check_id": "identity", "answer_candidate": "जी हाँ।"}),
        )
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    transcript, _, execution = await workflow._turn(pending_check=None)
    assert transcript == "जी हाँ।"
    assert execution is None
    assert session.answers == []


@pytest.mark.asyncio
async def test_malformed_identity_proposal_is_replaced_by_python_arguments(bound_policy, customer, session):
    io = FakeIO(["वंश"])
    agent = FakeConversationAgent([
        response(
            candidate="वंश",
            tool_proposal=ToolProposal("record_answer", {"check_id": "identity", "answer": "वंश"}),
        )
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)
    await workflow._process_check("identity")
    assert len(session.answers) == 1
    assert session.answers[0].original_answer == "वंश"
    assert session.verification_results["identity"].verdict.value == "pass"


@pytest.mark.parametrize("name_transcript", ["वंश", "वंशपूटानी"])
@pytest.mark.asyncio
async def test_identity_name_is_evaluated_even_when_gemini_classifies_it_as_confusion(
    bound_policy, customer, session, name_transcript
):
    io = FakeIO([name_transcript, "हाँ"])
    agent = FakeConversationAgent([
        response(
            ConversationIntent.CONFUSION,
            candidate=None,
            action=ConversationAction.REQUEST_CLARIFICATION,
            tool_proposal=ToolProposal("get_policy_fact", {"verification_check_id": "policy_term"}),
            spoken="अब आपकी policy का term क्या है?",
        ),
        response(candidate=None, action=ConversationAction.RESPOND),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)

    await workflow._process_check("identity", advance=False)

    assert session.verification_results["identity"].verdict.value == "pass"
    assert session.answers[0].original_answer == name_transcript
    assert session.identity_verified is False
    assert await workflow._confirm_identity() is True
    assert session.identity_verified is False
    assert io.spoken[-1] == VOICE_PROMPTS["identity_confirm"].format(name="वंश")
    assert VOICE_PROMPTS["policy_term"] not in io.spoken


@pytest.mark.asyncio
async def test_fresh_identity_turn_hides_expected_name_from_gemini_and_speech(bound_policy, customer, session):
    session.set_pending_question(VOICE_PROMPTS["identity"], verification_check_id="identity")
    io = FakeIO(["जी"])
    agent = FakeConversationAgent([
        response(ConversationIntent.CONFUSION, action=ConversationAction.REQUEST_CLARIFICATION, spoken="धन्यवाद वंश जी।")
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)

    await workflow._turn(pending_check="identity")

    assert agent.calls[0]["customer_context"] == {}
    assert all("वंश" not in text and "Vansh" not in text for text in io.spoken)
    assert VOICE_PROMPTS["identity"] == "सबसे पहले, क्या मैं आपका नाम जान सकता हूँ?"


@pytest.mark.asyncio
async def test_identity_confirmation_advances_to_policy_facts_and_first_check(bound_policy, customer, session):
    replies = [
        "हाँ", "वंश", "हाँ", "हाँ", "डेढ़ लाख", "पाँच साल", "बीस साल",
        "pension", "सुनीता को मिलेगा", "पाँच साल बाद", "policy lapse ho sakti hai",
    ]
    responses = [response() for _ in replies]
    responses[1] = response(ConversationIntent.UNKNOWN, action=ConversationAction.REQUEST_CLARIFICATION)
    io = FakeIO(replies)
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, FakeConversationAgent(responses))

    await workflow.run()

    assert session.identity_verified is True
    assert session.policy_explanation_completed is True
    assert "premium_amount" in session.verification_results
    assert any("policy Suraksha Pension Plus" in spoken for spoken in io.spoken)
    assert "आपका annual premium कितना है?" in io.spoken
    mandatory_sequence = [
        "premium_amount", "premium_term", "policy_term", "product_type",
        "payout_recipient", "payout_timing", "stop_paying",
    ]
    spoken_checks = [
        call["pending_check"] for call in workflow.agent.calls
        if call["pending_check"] in mandatory_sequence
    ]
    assert spoken_checks == mandatory_sequence


@pytest.mark.parametrize("recipient_answer", ["सुनीता", "सुनीता को मिलेगा", "Sunita", "Sunita को मिलेगा"])
@pytest.mark.asyncio
async def test_payout_recipient_in_supported_script_passes_without_retry(
    bound_policy, customer, session, recipient_answer
):
    io = FakeIO([recipient_answer])
    agent = FakeConversationAgent([response(candidate=recipient_answer)])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    for _ in range(4):
        workflow.flow.advance("check_passed")

    await workflow._process_check("payout_recipient")

    assert session.verification_results["payout_recipient"].verdict.value == "pass"
    assert len(session.answers) == 1
    assert workflow.flow.current_check == "payout_timing"
    assert not any("थोड़ा आसान" in spoken for spoken in io.spoken)


@pytest.mark.asyncio
async def test_unmatched_identity_does_not_verify_customer(bound_policy, customer, session):
    io = FakeIO(["किसी और का नाम", "दूसरा नाम"])
    agent = FakeConversationAgent([
        response(spoken="धन्यवाद वंश जी।"), response(spoken="वंश जी, कृपया बताइए।")
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)

    await workflow._process_check("identity", advance=False)

    assert session.identity_verified is False
    assert session.verification_results["identity"].verdict.value == "fail"
    assert all("वंश" not in text and "Vansh" not in text for text in io.spoken)


@pytest.mark.asyncio
async def test_identity_retry_uses_identity_explanation_and_accepts_permitted_variant(bound_policy, customer, session):
    io = FakeIO(["Ravi", "वंश"])
    agent = FakeConversationAgent([
        response(candidate="Ravi"),
        response(candidate="वंश"),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)
    await workflow._process_check("identity")
    assert session.verification_results["identity"].verdict.value == "pass_after_retry"
    assert session.retry_counts["identity"] == 1
    assert any("policyholder" in spoken for spoken in io.spoken)


@pytest.mark.asyncio
async def test_identity_confirmation_is_required_before_flow_advances(bound_policy, customer, session):
    io = FakeIO(["वंश", "हाँ"])
    agent = FakeConversationAgent([response(candidate="वंश"), response()])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)
    await workflow._process_check("identity", advance=False)
    assert workflow.flow.stage.value == "identity"
    assert await workflow._confirm_identity() is True
    workflow.flow.advance("identity", accepted=True)
    assert workflow.flow.stage.value == "fact_confirmation"
    assert any("वंश" in spoken for spoken in io.spoken)


@pytest.mark.asyncio
async def test_clarification_does_not_record_answer_or_consume_retry(bound_policy, customer, session):
    io = FakeIO(["Premium kitna hai?", "डेढ़ लाख"])
    agent = FakeConversationAgent([response(ConversationIntent.CUSTOMER_QUESTION), response(candidate="डेढ़ लाख")])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert len(session.answers) == 1
    assert session.retry_counts.get("premium_amount", 0) == 0
    assert session.answers[0].attempt_number == 1
    assert session.verification_results["premium_amount"].verdict.value == "pass"


@pytest.mark.asyncio
async def test_clarification_turn_preserves_pending_question(bound_policy, customer, session):
    io = FakeIO(["समझ नहीं आया", "डेढ़ लाख"])
    agent = FakeConversationAgent([
        response(ConversationIntent.CLARIFICATION),
        response(candidate="डेढ़ लाख"),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert agent.calls[0]["pending_check"] == "premium_amount"
    assert session.answers[0].attempt_number == 1
    assert session.retry_counts["premium_amount"] == 0


@pytest.mark.asyncio
async def test_unrelated_turn_does_not_consume_retry(bound_policy, customer, session):
    io = FakeIO(["मौसम अच्छा है", "डेढ़ लाख"])
    agent = FakeConversationAgent([
        response(ConversationIntent.UNRELATED),
        response(candidate="डेढ़ लाख"),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert session.answers[0].attempt_number == 1
    assert session.retry_counts.get("premium_amount", 0) == 0


@pytest.mark.asyncio
async def test_repeated_policy_questions_do_not_exhaust_retry(bound_policy, customer, session):
    io = FakeIO(["Premium kitna hai?", "Premium kitna hai?", "डेढ़ लाख"])
    agent = FakeConversationAgent([
        response(ConversationIntent.CUSTOMER_QUESTION),
        response(ConversationIntent.CUSTOMER_QUESTION),
        response(candidate="डेढ़ लाख"),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert len(session.answers) == 1
    assert session.answers[0].attempt_number == 1
    assert session.retry_counts.get("premium_amount", 0) == 0


@pytest.mark.asyncio
async def test_genuine_incorrect_answer_consumes_one_retry_before_correction(bound_policy, customer, session):
    io = FakeIO(["1,00,000", "1,50,000 है"])
    agent = FakeConversationAgent([
        response(candidate="1,00,000"),
        response(
            candidate="1,50,000 है",
            action=ConversationAction.PROPOSE_HANDOFF,
            tool_proposal=ToolProposal("handoff_to_human", {"reason": "Gemini misclassified retry"}),
            handoff=True,
        ),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert [answer.attempt_number for answer in session.answers] == [1, 2]
    assert session.retry_counts["premium_amount"] == 1
    assert session.verification_results["premium_amount"].verdict.value == "pass_after_retry"
    assert workflow.flow.current_check == "premium_term"
    assert session.human_handoff is False
    assert session.end_call_requested is False


@pytest.mark.parametrize("wrong_answer", ["दस साल", "1,00,000"])
@pytest.mark.asyncio
async def test_failed_premium_amount_retries_once_and_does_not_advance(
    bound_policy, customer, session, wrong_answer
):
    workflow = ConversationalCallWorkflow(
        FakeIO([wrong_answer, wrong_answer]),
        customer,
        bound_policy,
        session,
        FakeConversationAgent([
            response(candidate=wrong_answer, spoken="अब policy term कितने साल चलेगी?"),
            response(candidate=wrong_answer, spoken="अब policy term कितने साल चलेगी?"),
        ]),
    )
    enter_comprehension(workflow)

    await workflow._process_check("premium_amount")

    assert [answer.attempt_number for answer in session.answers] == [1, 2]
    assert all(answer.verification_check_id == "premium_amount" for answer in session.answers)
    assert workflow.flow.current_check == "premium_amount"
    assert "premium_amount" not in session.completed_checks
    assert all("policy term" not in text for text in workflow.io.spoken)
    assert session.human_handoff is True
    assert session.end_call_requested is False


@pytest.mark.asyncio
async def test_policy_term_cannot_run_before_current_premium_check(bound_policy, customer, session):
    workflow = ConversationalCallWorkflow(
        FakeIO([]), customer, bound_policy, session, FakeConversationAgent([])
    )
    enter_comprehension(workflow)

    with pytest.raises(ValueError, match="not the current mandatory check"):
        await workflow._process_check("policy_term")


class FakeAgentSession:
    def __init__(self, **kwargs):
        self.handlers = {}
        self.kwargs = kwargs

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler
        return register

    async def start(self, **kwargs):
        self.started = kwargs

    def shutdown(self, **kwargs):
        self.shutdown_args = kwargs

    async def interrupt(self, **kwargs):
        self.interrupt_args = kwargs


class FakeLiveKitAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRoom:
    class Participant:
        async def publish_data(self, payload, **kwargs):
            self.last = (payload, kwargs)

    def __init__(self):
        self.local_participant = self.Participant()


@pytest.mark.asyncio
async def test_final_transcripts_are_deduplicated_while_pending():
    port = LiveKitAgentSessionPort(FakeRoom(), session_factory=FakeAgentSession, agent_factory=FakeLiveKitAgent)
    await port.start(stt=object(), tts=object())
    event = type("Event", (), {"is_final": True, "transcript": "हाँ"})()
    port._session.handlers["user_input_transcribed"](event)
    port._session.handlers["user_input_transcribed"](event)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert await port.listen() == "हाँ"
    assert port._transcripts.empty()
    await port.close()


class CleanupSession:
    async def start(self, **kwargs):
        self.started = kwargs

    async def listen(self):
        raise RuntimeError("not reached")

    async def speak(self, text):
        pass

    async def stop_playback(self):
        self.stopped = True

    async def publish(self, event):
        pass

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_worker_exception_stores_terminal_verdict_and_cleans_up(monkeypatch, bound_policy, customer, session):
    def fail_agent(*args, **kwargs):
        raise AssertionError("provider setup failed")

    monkeypatch.setattr("agent.livekit_worker.LiveKitGnaniConfig.build", lambda: {"stt": object(), "tts": object()})
    monkeypatch.setattr("agent.livekit_worker.GeminiConversationAgent.from_environment", fail_agent)
    transport = CleanupSession()
    with pytest.raises(AssertionError):
        await run_conversational_session(transport, customer, bound_policy, session_context=session)
    assert session.final_verdict is not None
    assert session.final_verdict["verdict"] == "amber"
    assert getattr(transport, "closed", False) is True
