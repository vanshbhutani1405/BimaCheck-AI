from __future__ import annotations

import asyncio
import json

import pytest

from agent.conversation_agent import ConversationAction, ConversationIntent, ConversationResponse
from agent.livekit_worker import LiveKitAgentSessionPort, run_conversational_session
from agent.voice_workflow import ConversationalCallWorkflow


def response(intent=ConversationIntent.ANSWER, candidate=None, action=ConversationAction.PROPOSE_RECORD_ANSWER):
    return ConversationResponse(
        spoken_response="जी, धन्यवाद।",
        intent=intent,
        conversation_action=action,
        tool_proposal=None,
        requires_clarification=intent != ConversationIntent.ANSWER,
        clarification_reason=None,
        answer_candidate=candidate,
        policy_fact_ids=(),
        handoff_requested=False,
        end_call_requested=False,
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
    assert io.events[0]["final"] is True


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
    io = FakeIO(["दो लाख", "डेढ़ लाख"])
    agent = FakeConversationAgent([
        response(candidate="दो लाख"),
        response(candidate="डेढ़ लाख"),
    ])
    workflow = ConversationalCallWorkflow(io, customer, bound_policy, session, agent)
    enter_comprehension(workflow)
    await workflow._process_check("premium_amount")
    assert [answer.attempt_number for answer in session.answers] == [1, 2]
    assert session.retry_counts["premium_amount"] == 1
    assert session.verification_results["premium_amount"].verdict.value == "pass_after_retry"


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
