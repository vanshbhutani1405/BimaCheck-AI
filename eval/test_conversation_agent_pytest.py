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
)


def response_payload(**overrides):
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
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_structured_response_is_strictly_parsed():
    response = GeminiConversationAgent.parse_response(response_payload())
    assert response.intent == ConversationIntent.ANSWER
    assert response.conversation_action == ConversationAction.PROPOSE_RECORD_ANSWER


def test_json_code_fence_is_normalized_without_relaxing_schema():
    response = GeminiConversationAgent.parse_response("```json\n" + response_payload() + "\n```")
    assert response.intent == ConversationIntent.ANSWER


@pytest.mark.parametrize("raw", ["not json", "{}", response_payload(intent="not-valid")])
def test_malformed_response_is_rejected(raw):
    with pytest.raises(ValueError):
        GeminiConversationAgent.parse_response(raw)


def test_response_rejects_unapproved_tool():
    with pytest.raises(ValueError):
        GeminiConversationAgent.parse_response(response_payload(tool_proposal={"name": "set_verdict", "arguments": {}}))


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
