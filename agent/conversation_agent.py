"""Single Gemini conversational agent boundary for BimaCheck AI.

The agent interprets a turn and proposes actions. It does not execute tools,
record verification evidence, change session state, or calculate a verdict.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from agent.conversation_prompt import RESPONSE_JSON_SCHEMA, SYSTEM_PROMPT
from agent.policy import Policy
from agent.session import SessionContext


logger = logging.getLogger(__name__)
MAX_DIAGNOSTIC_SAMPLE = 240


def _diagnostic_sample(value: Any) -> str:
    """Return a bounded, control-character-safe sample without secrets."""
    text = value if isinstance(value, str) else repr(value)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"(?i)(authorization|api[_-]?key|secret|token)\s*[:=]\s*[^,}\s]+", r"\1=<redacted>", text)
    return text[:MAX_DIAGNOSTIC_SAMPLE]


RESPONSE_FIELDS = frozenset(RESPONSE_JSON_SCHEMA["required"])


class ConversationIntent(str, Enum):
    ANSWER = "answer"
    CUSTOMER_QUESTION = "customer_question"
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    CONFUSION = "confusion"
    REFUSAL = "refusal"
    UNRELATED = "unrelated"
    INTERRUPTION = "interruption"
    INCOMPLETE = "incomplete"
    SILENCE = "silence"
    UNKNOWN = "unknown"


class ConversationAction(str, Enum):
    RESPOND = "respond"
    ANSWER_POLICY_THEN_RESUME = "answer_policy_then_resume"
    REQUEST_CLARIFICATION = "request_clarification"
    PROPOSE_RECORD_ANSWER = "propose_record_answer"
    PROPOSE_GET_POLICY_FACT = "propose_get_policy_fact"
    PROPOSE_FLAG_RISK = "propose_flag_risk"
    PROPOSE_HANDOFF = "propose_handoff"
    PROPOSE_END_CALL = "propose_end_call"
    NONE = "none"


ALLOWED_TOOL_NAMES = frozenset({
    "record_answer",
    "get_policy_fact",
    "flag_risk",
    "handoff_to_human",
    "end_call",
})


@dataclass(frozen=True)
class ToolProposal:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ConversationResponse:
    spoken_response: str
    intent: ConversationIntent
    conversation_action: ConversationAction
    tool_proposal: ToolProposal | None
    requires_clarification: bool
    clarification_reason: str | None
    answer_candidate: str | None
    policy_fact_ids: tuple[str, ...]
    handoff_requested: bool
    end_call_requested: bool


@dataclass(frozen=True)
class GeminiSettings:
    model: str = "google/gemini-3.5-flash-lite"
    timeout_seconds: float = 8.0
    max_history_messages: int = 12
    max_output_tokens: int = 300

    @classmethod
    def from_environment(cls) -> "GeminiSettings":
        missing = [
            name for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise ValueError(f"LiveKit credentials are required: {', '.join(missing)}")
        model = os.getenv("GEMINI_MODEL", "google/gemini-3.5-flash-lite").strip()
        if not model:
            raise ValueError("GEMINI_MODEL must not be empty")
        return cls(
            model=model,
            timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "8.0")),
            max_history_messages=int(os.getenv("GEMINI_MAX_HISTORY_MESSAGES", "12")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "300")),
        )


class GeminiConversationAgent:
    """LiveKit Inference adapter with strict response validation and fallback."""

    def __init__(self, llm: Any, settings: GeminiSettings):
        self.llm = llm
        self.settings = settings

    @classmethod
    def from_environment(cls, settings: GeminiSettings | None = None) -> "GeminiConversationAgent":
        chosen = settings or GeminiSettings.from_environment()
        try:
            from livekit.agents import inference
        except ImportError as exc:
            raise RuntimeError("livekit-agents with inference support is required") from exc
        return cls(
            inference.LLM(
                model=chosen.model,
                extra_kwargs={
                    "reasoning_effort": "low",
                    "max_completion_tokens": chosen.max_output_tokens,
                },
            ),
            chosen,
        )

    async def respond(
        self,
        *,
        session: SessionContext,
        customer_message: str,
        policy: Policy,
        customer_context: Mapping[str, Any] | None = None,
        relevant_fact_ids: Sequence[str] = (),
    ) -> ConversationResponse:
        if not isinstance(customer_message, str) or not customer_message.strip():
            return self._fallback("empty customer message")
        prompt = self._build_turn_prompt(session, customer_message, policy, customer_context, relevant_fact_ids)
        started = time.perf_counter()
        logger.info("gemini_conversation_request_started", extra={"model": self.settings.model, "message_length": len(customer_message)})
        try:
            raw = await asyncio.wait_for(self._request(prompt), timeout=self.settings.timeout_seconds)
            logger.info(
                "gemini_conversation_request_completed",
                extra={"model": self.settings.model, "response_length": len(raw), "duration_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
            return self.parse_response(raw)
        except asyncio.TimeoutError:
            logger.warning("gemini_conversation_timeout", extra={"model": self.settings.model, "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
            return self._fallback("provider timeout")
        except Exception as exc:
            logger.exception(
                "gemini_conversation_failed",
                extra={"model": self.settings.model, "error_type": type(exc).__name__, "duration_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
            return self._fallback("provider failure")

    async def _request(self, prompt: str) -> str:
        """Run one standalone LiveKit Inference chat request and collect text."""
        from livekit.agents import APIConnectOptions, ChatContext

        started = time.perf_counter()
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="system", content=SYSTEM_PROMPT)
        chat_ctx.add_message(role="user", content=prompt)
        stream = self.llm.chat(
            chat_ctx=chat_ctx,
            conn_options=APIConnectOptions(
                max_retry=0,
                timeout=self.settings.timeout_seconds,
                liveness_timeout=max(1.0, self.settings.timeout_seconds),
            ),
        )
        chunks: list[str] = []
        chunk_count = 0
        content_types: set[str] = set()
        try:
            async for chunk in stream:
                chunk_count += 1
                content = getattr(getattr(chunk, "delta", None), "content", None)
                content_types.add(type(content).__name__)
                if isinstance(content, str) and content:
                    chunks.append(content)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        raw = "".join(chunks)
        logger.info(
            "gemini_inference_response_extracted",
            extra={
                "stream_type": type(stream).__name__,
                "chunk_count": chunk_count,
                "content_types": tuple(sorted(content_types)),
                "text_length": len(raw),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        if not raw.strip():
            logger.warning(
                "gemini_response_extraction_failed",
                extra={"reason": "empty_text_after_stream", "chunk_count": chunk_count},
            )
            raise ValueError("LiveKit Inference returned no text")
        return raw

    def _build_turn_prompt(
        self,
        session: SessionContext,
        customer_message: str,
        policy: Policy,
        customer_context: Mapping[str, Any] | None,
        relevant_fact_ids: Sequence[str],
    ) -> str:
        pending = session.get_pending_question()
        fact_ids = tuple(relevant_fact_ids)
        if not fact_ids and pending and pending.verification_check_id:
            mapped_fact = policy.get_fact(verification_check_id=pending.verification_check_id)
            if mapped_fact.exists:
                fact_ids = (mapped_fact.fact_id or "",)
        approved_facts = []
        for fact_id in fact_ids:
            if not fact_id:
                continue
            fact = policy.get_fact(fact_id=fact_id)
            if fact.exists:
                approved_facts.append({
                    "fact_id": fact.fact_id,
                    "available": fact.available,
                    "value": fact.value if fact.available else None,
                    "explanation": fact.explanation if fact.available else None,
                    "limitation": fact.limitation,
                    "policy_version": fact.policy_version,
                })
        history = [
            {"speaker": message.speaker, "text": message.text, "timestamp": message.timestamp.isoformat()}
            for message in session.recent_messages(self.settings.max_history_messages)
        ]
        pending_payload = None if pending is None else {
            "question_id": pending.question_id,
            "verification_check_id": pending.verification_check_id,
            "text": pending.text,
            "answered": pending.answered,
            "clarification_required": pending.clarification_required,
        }
        return json.dumps({
            "customer_message": customer_message,
            "customer_context": {
                key: customer_context[key]
                for key in ("name", "name_variants")
                if customer_context is not None and key in customer_context
            },
            "pending_question": pending_payload,
            "conversation_history": history,
            "approved_policy_facts": approved_facts,
        }, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def parse_response(cls, raw: Any) -> ConversationResponse:
        if not isinstance(raw, str) or not raw.strip():
            logger.warning("gemini_response_parsing_failed", extra={"reason": "empty_or_non_string"})
            raise ValueError("Gemini returned an empty response")
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or not lines[-1].strip().startswith("```"):
                logger.warning("gemini_response_parsing_failed", extra={"reason": "unterminated_code_fence", "sample": _diagnostic_sample(text)})
                raise ValueError("Gemini response had an unterminated JSON code fence")
            text = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "gemini_response_parsing_failed",
                extra={"reason": "invalid_json", "sample": _diagnostic_sample(text)},
            )
            raise ValueError("Gemini response was not valid JSON") from exc
        if not isinstance(payload, dict):
            logger.warning("gemini_response_validation_failed", extra={"reason": "response_not_object"})
            raise ValueError("Gemini response must be an object")
        actual_fields = set(payload)
        if actual_fields != RESPONSE_FIELDS:
            logger.warning(
                "gemini_response_schema_validation_failed",
                extra={
                    "missing_fields": tuple(sorted(RESPONSE_FIELDS - actual_fields)),
                    "unexpected_fields": tuple(sorted(actual_fields - RESPONSE_FIELDS)),
                    "sample": _diagnostic_sample(text),
                },
            )
            raise ValueError("Gemini response fields do not match the response contract")
        spoken = payload["spoken_response"]
        if not isinstance(spoken, str) or not spoken.strip() or len(spoken) > 1000:
            raise ValueError("spoken_response must be a non-empty string under 1000 characters")
        try:
            intent = ConversationIntent(payload["intent"])
            action = ConversationAction(payload["conversation_action"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Gemini response contains an unsupported intent or action") from exc
        if not isinstance(payload["requires_clarification"], bool) or not isinstance(payload["handoff_requested"], bool) or not isinstance(payload["end_call_requested"], bool):
            raise ValueError("Gemini response boolean fields are invalid")
        if payload["clarification_reason"] is not None and not isinstance(payload["clarification_reason"], str):
            raise ValueError("clarification_reason must be a string or null")
        if payload["answer_candidate"] is not None and not isinstance(payload["answer_candidate"], str):
            raise ValueError("answer_candidate must be a string or null")
        fact_ids = payload["policy_fact_ids"]
        if not isinstance(fact_ids, list) or not all(isinstance(value, str) for value in fact_ids):
            raise ValueError("policy_fact_ids must be a list of strings")
        proposal_payload = payload["tool_proposal"]
        proposal = None
        if proposal_payload is not None:
            if not isinstance(proposal_payload, dict) or set(proposal_payload) != {"name", "arguments"}:
                raise ValueError("tool_proposal has an invalid shape")
            name = proposal_payload["name"]
            arguments = proposal_payload["arguments"]
            if name not in ALLOWED_TOOL_NAMES or not isinstance(arguments, dict):
                raise ValueError("tool_proposal contains an unsupported tool or arguments")
            proposal = ToolProposal(name, dict(arguments))
        return ConversationResponse(
            spoken_response=spoken.strip(),
            intent=intent,
            conversation_action=action,
            tool_proposal=proposal,
            requires_clarification=payload["requires_clarification"],
            clarification_reason=payload["clarification_reason"],
            answer_candidate=payload["answer_candidate"],
            policy_fact_ids=tuple(fact_ids),
            handoff_requested=payload["handoff_requested"],
            end_call_requested=payload["end_call_requested"],
        )

    @staticmethod
    def _fallback(reason: str) -> ConversationResponse:
        logger.info("gemini_safe_fallback", extra={"reason": reason})
        return ConversationResponse(
            spoken_response="मुझे आपकी बात पूरी तरह समझ नहीं आई। कृपया अपना जवाब एक बार फिर बताइए।",
            intent=ConversationIntent.UNKNOWN,
            conversation_action=ConversationAction.REQUEST_CLARIFICATION,
            tool_proposal=None,
            requires_clarification=True,
            clarification_reason="The customer message could not be safely interpreted.",
            answer_candidate=None,
            policy_fact_ids=(),
            handoff_requested=False,
            end_call_requested=False,
        )
