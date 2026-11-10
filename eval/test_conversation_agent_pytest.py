from __future__ import annotations

import json
import sys
import types

import pytest

from agent.conversation_agent import (
    ConversationAction,
    ConversationIntent,
    GeminiConversationAgent,
    GeminiSettings,
    RESPONSE_FIELDS,
)


def response_payload(*, action=None, **overrides):
    payload = {
        "spoken_response": "जी, कृपया अपना जवाब बताइए।",
        "intent": "answer",
        "conversation_action": "propose_record_answer",
        "tool_proposal": None,
        "requires_clarification": False,
        "clarification_reason": None,
        "answer_candidate": "डेढ़ लाख",
        "policy_fact_ids": [],
        "handoff_requested": False,
        "end_call_requested": False,
    }
    if action is not None:
        overrides["conversation_action"] = action
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_structured_response_is_strictly_parsed():
    raw = response_payload()
    assert set(json.loads(raw)) == RESPONSE_FIELDS
    response = GeminiConversationAgent.parse_response(raw)
    assert response.intent == ConversationIntent.ANSWER
    assert response.conversation_action == ConversationAction.PROPOSE_RECORD_ANSWER


def test_missing_optional_response_fields_get_safe_defaults():
    payload = {"spoken_response": "जी।", "intent": "answer"}
    response = GeminiConversationAgent.parse_response(json.dumps(payload, ensure_ascii=False))
    assert response.conversation_action == ConversationAction.RESPOND
    assert response.tool_proposal is None
    assert response.answer_candidate is None
    assert response.policy_fact_ids == ()
    assert response.handoff_requested is False
    assert response.end_call_requested is False


def test_unexpected_response_fields_are_ignored():
    response = GeminiConversationAgent.parse_response(
        response_payload(unexpected_model_note="do not use this")
    )
    assert response.intent == ConversationIntent.ANSWER
    assert response.tool_proposal is None


@pytest.mark.parametrize("intent", [value.value for value in ConversationIntent])
def test_every_supported_intent_parses(intent):
    assert GeminiConversationAgent.parse_response(response_payload(intent=intent)).intent.value == intent


@pytest.mark.parametrize("action", [value.value for value in ConversationAction])
def test_every_supported_action_parses(action):
    assert GeminiConversationAgent.parse_response(response_payload(action=action)).conversation_action.value == action


def test_natural_enum_aliases_are_explicitly_normalized():
    response = GeminiConversationAgent.parse_response(
        response_payload(intent="affirmation", conversation_action="continue_verification")
    )
    assert response.intent == ConversationIntent.ANSWER
    assert response.conversation_action == ConversationAction.RESPOND


def test_affirmation_alias_does_not_create_a_tool_or_verification_result():
    response = GeminiConversationAgent.parse_response(
        response_payload(intent="affirmation", conversation_action="continue_verification", answer_candidate=None)
    )
    assert response.tool_proposal is None
    assert response.answer_candidate is None


def test_json_code_fence_is_normalized_without_relaxing_schema():
    response = GeminiConversationAgent.parse_response("```json\n" + response_payload() + "\n```")
    assert response.intent == ConversationIntent.ANSWER


@pytest.mark.parametrize("raw", ["not json", "{}", response_payload(intent="not-valid")])
def test_malformed_response_is_rejected(raw):
    with pytest.raises(ValueError):
        GeminiConversationAgent.parse_response(raw)


def test_missing_semantic_field_is_rejected():
    payload = json.loads(response_payload())
    del payload["intent"]
    with pytest.raises(ValueError, match="missing required fields"):
        GeminiConversationAgent.parse_response(json.dumps(payload, ensure_ascii=False))


def test_response_rejects_unapproved_tool():
    with pytest.raises(ValueError):
        GeminiConversationAgent.parse_response(response_payload(tool_proposal={"name": "set_verdict", "arguments": {}}))


def test_unknown_enum_values_are_rejected_without_fabricating_an_action():
    with pytest.raises(ValueError, match="unsupported intent"):
        GeminiConversationAgent.parse_response(response_payload(intent="made_up_intent"))
    with pytest.raises(ValueError, match="unsupported conversation_action"):
        GeminiConversationAgent.parse_response(response_payload(action="made_up_action"))


def test_prompt_contains_pending_question_and_only_minimal_customer_context(bound_policy, customer, session):
    session.set_pending_question("Premium?", verification_check_id="premium_amount")
    agent = GeminiConversationAgent(object(), GeminiSettings(max_history_messages=2))
    prompt = agent._build_turn_prompt(session, "कितना?", bound_policy, {**customer, "declared_income": 600000}, ())
    assert "Premium?" in prompt
    assert "declared_income" not in prompt
    assert "premium.amount" in prompt


@pytest.mark.asyncio
async def test_provider_exception_returns_safe_fallback(bound_policy, session):
    class BrokenAgent(GeminiConversationAgent):
        async def _request(self, prompt):
            raise RuntimeError("provider unavailable")

    response = await BrokenAgent(object(), GeminiSettings(timeout_seconds=0.1)).respond(
        session=session, customer_message="hello", policy=bound_policy
    )
    assert response.intent == ConversationIntent.UNKNOWN
    assert response.requires_clarification is True
    assert "समझ" in response.spoken_response


@pytest.mark.asyncio
async def test_malformed_response_returns_safe_fallback(bound_policy, session):
    class MalformedAgent(GeminiConversationAgent):
        async def _request(self, prompt):
            return "not json"

    response = await MalformedAgent(object(), GeminiSettings()).respond(
        session=session, customer_message="hello", policy=bound_policy
    )

    assert response.intent == ConversationIntent.UNKNOWN
    assert response.requires_clarification is True
    assert response.tool_proposal is None
    assert response.handoff_requested is False
    assert response.end_call_requested is False


@pytest.mark.asyncio
async def test_livekit_stream_chunks_are_aggregated_in_order(monkeypatch, bound_policy, session):
    class FakeChatContext:
        def add_message(self, **kwargs):
            pass

    class FakeConnectOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            async def chunks():
                for part in (response_payload()[:80], response_payload()[80:]):
                    yield types.SimpleNamespace(delta=types.SimpleNamespace(content=part))
            return chunks()

        async def aclose(self):
            self.closed = True

    class FakeLLM:
        def __init__(self):
            self.stream = FakeStream()

        def chat(self, **kwargs):
            return self.stream

    fake_agents = types.ModuleType("livekit.agents")
    fake_agents.APIConnectOptions = FakeConnectOptions
    fake_agents.ChatContext = FakeChatContext
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    agent = GeminiConversationAgent(FakeLLM(), GeminiSettings(timeout_seconds=1))
    raw = await agent._request("{}")
    assert json.loads(raw)["intent"] == "answer"
