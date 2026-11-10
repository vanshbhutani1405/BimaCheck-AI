from __future__ import annotations

import asyncio
import json
import time

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
        self.latency_marks = {}
        self._turn_sequence = 0
        self.last_transcript_timing = None

    async def listen(self):
        text = next(self.replies)
        self._turn_sequence += 1
        self.last_transcript_timing = (self._turn_sequence, time.perf_counter())
        return text

    async def speak(self, text):
        self.spoken.append(text)

    async def publish(self, event):
        self.events.append(dict(event))

    def update_latency(self, **marks):
        self.latency_marks.update(marks)


class WaitingFakeIO(FakeIO):
    async def listen(self):
        try:
            text = next(self.replies)
        except StopIteration:
            await asyncio.Event().wait()
        self._turn_sequence += 1
        self.last_transcript_timing = (self._turn_sequence, time.perf_counter())
        return text


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
async def test_one_check_records_evidence_and_advances_python_workflow(bound_policy, customer, session, caplog):
    caplog.set_level("INFO", logger="agent.voice_workflow")
    io = FakeIO(["डेढ़ लाख"])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, FakeConversationAgent([response(candidate="डेढ़ लाख")]))
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert session.verification_results["premium_amount"].verdict.value == "pass"
    assert session.answers[0].original_answer == "डेढ़ लाख"
    assert session.pending_question is None
    assert workflow.flow.current_check == "premium_term"
    assert io.events[0]["final"] is True
    assert io.latency_marks["gemini_started_at"] <= io.latency_marks["gemini_completed_at"]
    assert io.latency_marks["python_started_at"] <= io.latency_marks["python_completed_at"]
    assert io.latency_marks["evaluation_started_at"] <= io.latency_marks["evaluation_completed_at"]
    latency_logs = "\n".join(record.getMessage() for record in caplog.records if "LATENCY turn=" in record.getMessage())
    assert "GEMINI_REQUEST_START" in latency_logs
    assert "GEMINI_RESPONSE_RECEIVED" in latency_logs
    assert "PYTHON_WORK_END" in latency_logs
    assert "डेढ़ लाख" not in latency_logs


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
    io = FakeIO([name_transcript])
    agent = FakeConversationAgent([
        response(
            ConversationIntent.CONFUSION,
            candidate=None,
            action=ConversationAction.REQUEST_CLARIFICATION,
            tool_proposal=ToolProposal("get_policy_fact", {"verification_check_id": "policy_term"}),
            spoken="अब आपकी policy का term क्या है?",
        ),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    workflow.flow.advance("connected")
    workflow.flow.advance("consent", accepted=True)

    await workflow._process_check("identity", advance=False)

    assert session.verification_results["identity"].verdict.value == "pass"
    assert session.answers[0].original_answer == name_transcript
    assert session.identity_verified is False
    assert not any("क्या आपका नाम वंश है" in text for text in io.spoken)


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
async def test_identity_match_advances_without_confirmation_and_grouped_checks_complete(bound_policy, customer, session):
    replies = [
        "हाँ", "वंश", "हाँ", "डेढ़ लाख और पाँच साल", "बीस साल और pension",
        "सुनीता और पाँच साल बाद", "policy lapse ho sakti hai",
    ]
    responses = [response() for _ in replies]
    responses[1] = response(ConversationIntent.UNKNOWN, action=ConversationAction.REQUEST_CLARIFICATION)
    io = FakeIO(replies)
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, FakeConversationAgent(responses))

    report = await workflow.run()

    assert session.identity_verified is True
    assert session.policy_explanation_completed is True
    assert "premium_amount" in session.verification_results
    assert any("policy Suraksha Pension Plus" in spoken for spoken in io.spoken)
    assert VOICE_PROMPTS["group_premium"] in io.spoken
    mandatory_sequence = [
        "premium_amount", "premium_term", "policy_term", "product_type",
        "payout_recipient", "payout_timing", "stop_paying",
    ]
    spoken_checks = [
        call["pending_check"] for call in workflow.agent.calls
        if call["pending_check"] in {"premium_amount", "policy_term", "payout_recipient", "stop_paying"}
    ]
    assert spoken_checks == ["premium_amount", "policy_term", "payout_recipient", "stop_paying"]
    assert report["verdict"] == "green"
    assert session.workflow_complete is True
    assert session.status.value == "completed"
    assert all(session.retry_counts.get(check_id, 0) == 0 for check_id in mandatory_sequence)
    assert [line for line in io.spoken if line in {
        VOICE_PROMPTS["group_premium"], VOICE_PROMPTS["group_term"],
        VOICE_PROMPTS["group_payout"], VOICE_PROMPTS["stop_paying"],
    }] == [VOICE_PROMPTS["group_premium"], VOICE_PROMPTS["group_term"],
          VOICE_PROMPTS["group_payout"], VOICE_PROMPTS["stop_paying"]]
    assert [answer.verification_check_id for answer in session.answers if answer.verification_check_id in mandatory_sequence] == mandatory_sequence
    answers = {answer.verification_check_id: answer.original_answer for answer in session.answers}
    assert answers["premium_amount"] == answers["premium_term"] == replies[3]
    assert answers["policy_term"] == answers["product_type"] == replies[4]
    assert answers["payout_recipient"] == answers["payout_timing"] == replies[5]
    assert not any("क्या आपका नाम वंश है" in line for line in io.spoken)


@pytest.mark.parametrize(
    "check_ids,prompt_key,first_fragment,second_fragment,expected",
    [
        (("premium_amount", "premium_term"), "group_premium", "मेरा annual premium 1,50,000 है", "5 साल तक भरना होगा", ("pass", "pass")),
        (("policy_term", "product_type"), "group_term", "मेरी policy", "20 साल तक चलेगी, यह pension product है", ("pass", "pass")),
        (("payout_recipient", "payout_timing"), "group_payout", "Sunita को payout मिलेगा", "5 साल बाद pension शुरू होगी", ("pass", "pass")),
    ],
)
@pytest.mark.asyncio
async def test_grouped_answer_fragments_are_combined_before_evaluation(
    bound_policy, customer, session, check_ids, prompt_key, first_fragment, second_fragment, expected
):
    io = FakeIO([first_fragment, second_fragment])
    agent = FakeConversationAgent([response(candidate=f"{first_fragment} {second_fragment}")])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    mandatory_order = (
        "premium_amount", "premium_term", "policy_term", "product_type",
        "payout_recipient", "payout_timing", "stop_paying",
    )
    for _ in range(mandatory_order.index(check_ids[0])):
        workflow.flow.advance("check_passed")

    await workflow._process_group(check_ids, prompt_key)

    assert [session.verification_results[key].verdict.value for key in check_ids] == list(expected)
    assert [answer.original_answer for answer in session.answers] == [f"{first_fragment} {second_fragment}"] * 2
    assert workflow.flow.current_check == {
        "premium_amount": "policy_term",
        "policy_term": "payout_recipient",
        "payout_recipient": "stop_paying",
    }[check_ids[0]]
    assert len(agent.calls) == 1
    assert len([line for line in io.spoken if line == VOICE_PROMPTS[prompt_key]]) == 1
    assert io.latency_marks["turn_id"] == 1


@pytest.mark.asyncio
async def test_split_wrong_premium_and_correct_payment_term_are_evaluated_independently(
    bound_policy, customer, session
):
    first, second = "मेरा annual premium 1,00,000 है", "5 साल तक भरना होगा"
    io = FakeIO([first, second])
    agent = FakeConversationAgent([response(candidate=f"{first} {second}")])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)

    await workflow._process_group(("premium_amount", "premium_term"), "group_premium")

    assert workflow.session.verification_results["premium_amount"].verdict.value == "fail"
    assert workflow.session.verification_results["premium_term"].verdict.value == "pass"
    assert [answer.verification_check_id for answer in session.answers] == ["premium_amount", "premium_term"]
    assert session.retry_counts.get("premium_amount", 0) == session.retry_counts.get("premium_term", 0) == 0
    assert workflow.flow.current_check == "policy_term"


@pytest.mark.asyncio
async def test_separate_new_answer_is_buffered_instead_of_merged(bound_policy, customer, session):
    first, separate = "मेरा annual premium 1,50,000 है", "मेरी policy 20 साल तक चलेगी"
    io = FakeIO([first, separate])
    agent = FakeConversationAgent([response(candidate=first)])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)

    await workflow._process_group(("premium_amount", "premium_term"), "group_premium")

    assert [answer.original_answer for answer in session.answers] == [first, first]
    assert workflow._buffered_customer_input is not None
    assert workflow._buffered_customer_input[0] == separate


@pytest.mark.asyncio
async def test_split_stop_paying_answer_is_evaluated_once_without_reasking(
    bound_policy, customer, session
):
    first, second = "अगर आप premium देना बंद कर देते हैं,", "इसके बारे में मुझे नहीं पता"
    io = FakeIO([first, second])
    agent = FakeConversationAgent([response(candidate=f"{first} {second}")])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    for _ in range(6):
        workflow.flow.advance("check_passed")

    await workflow._process_check("stop_paying")

    assert len([answer for answer in session.answers if answer.verification_check_id == "stop_paying"]) == 1
    assert session.answers[-1].original_answer == f"{first} {second}"
    assert session.verification_results["stop_paying"].verdict.value == "unclear"
    assert len(agent.calls) == 1
    assert io.spoken.count(VOICE_PROMPTS["stop_paying"]) == 1
    assert workflow.flow.current_check == "affordability"


@pytest.mark.asyncio
async def test_genuinely_unclear_stop_paying_answer_is_one_discrepancy_without_retry(
    bound_policy, customer, session, monkeypatch
):
    monkeypatch.setattr("agent.voice_workflow.ANSWER_FRAGMENT_TIMEOUT_SECONDS", 0.01)
    io = WaitingFakeIO(["मुझे नहीं पता"])
    agent = FakeConversationAgent([response(candidate="मुझे नहीं पता")])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    for _ in range(6):
        workflow.flow.advance("check_passed")

    await workflow._process_check("stop_paying")

    assert session.verification_results["stop_paying"].verdict.value == "unclear"
    assert len(session.answers) == 1
    assert session.retry_counts.get("stop_paying", 0) == 0
    assert io.spoken.count(VOICE_PROMPTS["stop_paying"]) == 1
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_split_answer_followed_by_explicit_end_call_is_not_merged(
    bound_policy, customer, session
):
    io = FakeIO(["annual premium 1,50,000", "कॉल बंद कर दीजिए"])
    agent = FakeConversationAgent([
        response(candidate="कॉल बंद कर दीजिए", action=ConversationAction.PROPOSE_END_CALL,
                 tool_proposal=ToolProposal("end_call", {"reason": "customer request", "confirmation": True}),
                 end_call=True)
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)

    await workflow._process_group(("premium_amount", "premium_term"), "group_premium")

    assert session.end_call_requested is True
    assert session.answers == []
    assert len(agent.calls) == 1


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
async def test_current_answer_wins_over_gemini_handoff_and_moves_on(bound_policy, customer, session):
    io = FakeIO(["1,50,000 है"])
    agent = FakeConversationAgent([
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
    assert [answer.attempt_number for answer in session.answers] == [1]
    assert session.retry_counts["premium_amount"] == 0
    assert session.verification_results["premium_amount"].verdict.value == "pass"
    assert workflow.flow.current_check == "premium_term"
    assert session.human_handoff is False
    assert session.end_call_requested is False


@pytest.mark.asyncio
async def test_discrepancy_wins_over_gemini_handoff_and_moves_on(bound_policy, customer, session):
    io = FakeIO(["1,00,000"])
    agent = FakeConversationAgent([
        response(
            candidate="1,00,000",
            action=ConversationAction.PROPOSE_HANDOFF,
            tool_proposal=ToolProposal("handoff_to_human", {"reason": "Gemini proposed review early"}),
            handoff=True,
        )
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)

    await workflow._process_check("premium_amount")

    assert session.verification_results["premium_amount"].verdict.value == "fail"
    assert workflow.flow.current_check == "premium_term"
    assert session.human_handoff is False
    assert session.status.value == "active"


@pytest.mark.parametrize("wrong_answer", ["दस साल", "1,00,000"])
@pytest.mark.asyncio
async def test_failed_premium_amount_is_recorded_and_advances_without_retry(
    bound_policy, customer, session, wrong_answer
):
    workflow = ConversationalCallWorkflow(
        FakeIO([wrong_answer]),
        customer,
        bound_policy,
        session,
        FakeConversationAgent([response(candidate=wrong_answer, spoken="अब policy term कितने साल चलेगी?")]),
    )
    enter_comprehension(workflow)

    await workflow._process_check("premium_amount")

    assert [answer.attempt_number for answer in session.answers] == [1]
    assert all(answer.verification_check_id == "premium_amount" for answer in session.answers)
    assert workflow.flow.current_check == "premium_term"
    assert "premium_amount" in session.completed_checks
    assert all("policy term" not in text for text in workflow.io.spoken)
    assert session.human_handoff is False
    assert session.end_call_requested is False


@pytest.mark.parametrize(
    "premium_answer,payout_timing_answer,expected_discrepancies",
    [
        ("1,00,000", "5 साल बाद", 1),
        ("1,00,000", "10 साल बाद", 2),
    ],
)
@pytest.mark.asyncio
async def test_one_pass_finishes_seven_checks_then_returns_amber_for_discrepancies(
    bound_policy, customer, session, premium_answer, payout_timing_answer, expected_discrepancies
):
    replies = [
        "हाँ", "वंश", "हाँ", f"{premium_answer} और 5 साल", "20 साल और pension",
        f"सुनीता और {payout_timing_answer}", "policy may lapse",
    ]
    workflow = ConversationalCallWorkflow(
        FakeIO(replies), customer, bound_policy, session,
        FakeConversationAgent([response() for _ in replies]),
    )

    report = await workflow.run()

    mandatory = [
        "premium_amount", "premium_term", "policy_term", "product_type",
        "payout_recipient", "payout_timing", "stop_paying",
    ]
    assert report["verdict"] == "amber"
    assert workflow.session.workflow_complete is True
    assert workflow.session.completed_checks.issuperset(mandatory)
    assert len([answer for answer in workflow.session.answers if answer.verification_check_id in mandatory]) == 7
    assert sum(answer.attempt_number > 1 for answer in workflow.session.answers) == 0
    assert len([line for line in workflow.io.spoken if line in {
        VOICE_PROMPTS["group_premium"], VOICE_PROMPTS["group_term"],
        VOICE_PROMPTS["group_payout"], VOICE_PROMPTS["stop_paying"],
    }]) == 4
    assert [answer.verification_check_id for answer in workflow.session.answers if answer.verification_check_id in mandatory] == mandatory
    assert len([key for key in mandatory if workflow.session.verification_results[key].verdict.value in {"fail", "unclear", "refused"}]) == expected_discrepancies
    assert workflow.session.status.value == "human_handoff"
    assert "human review" in workflow.io.spoken[-1]


@pytest.mark.asyncio
async def test_incorrect_payout_timing_advances_to_stop_paying(bound_policy, customer, session):
    workflow = ConversationalCallWorkflow(
        FakeIO(["10 साल बाद"]), customer, bound_policy, session,
        FakeConversationAgent([response(candidate="10 साल बाद")]),
    )
    enter_comprehension(workflow)
    for _ in range(5):
        workflow.flow.advance("check_passed")

    await workflow._process_check("payout_timing")

    assert workflow.session.verification_results["payout_timing"].verdict.value == "fail"
    assert workflow.flow.current_check == "stop_paying"
    assert len(workflow.session.answers) == 1
    assert workflow.session.retry_counts["payout_timing"] == 0


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
