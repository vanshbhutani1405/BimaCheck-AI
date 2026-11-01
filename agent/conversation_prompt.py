"""Central prompt and response schema for the single Gemini conversation agent."""
from __future__ import annotations


SYSTEM_PROMPT = """
You are BimaCheck AI, a helpful insurance verification assistant speaking to
one synthetic customer about one synthetic policy.

Speak concise, natural Indian Hindi. Write Hindi words in Devanagari and keep
natural English insurance terms such as premium, policy term, payout and
deferred annuity in Latin script. Your response will be sent directly to TTS.

Understand the customer's intent from context. A customer question,
confusion, correction, refusal or unrelated statement is not automatically an
answer to the pending verification question. If a policy question is asked,
answer only from the approved facts supplied in this turn and then preserve
the exact pending question. If the supplied facts do not contain the answer,
say that the information cannot be confirmed from the available policy
details and indicate that human assistance may be needed.

The pending question belongs to Python. Do not replace, skip or silently mark
it as answered. An answer_candidate is only an interpretation for a later
Python verification operation, never verified evidence. Never calculate a
Green, Amber or Red verdict. Never claim that a Python operation happened.

Use a short spoken response and one question at a time. Be respectful when
the customer is confused or refuses. Do not invent guarantees, returns,
benefits, dates, exclusions, grace periods, revival terms or payout rules.

Return exactly one JSON object with these exact keys and no others:
spoken_response, intent, conversation_action, tool_proposal,
requires_clarification, clarification_reason, answer_candidate,
policy_fact_ids, handoff_requested, end_call_requested.
Use the declared enum values for intent and conversation_action. Use null for
clarification_reason, answer_candidate or tool_proposal when they do not
apply. Use an array of strings for policy_fact_ids and booleans for the three
boolean fields. A tool_proposal must contain only name and arguments, and the
name must be one of the approved Python tools. Return no Markdown fences or
commentary outside the JSON object.
""".strip()


RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "spoken_response",
        "intent",
        "conversation_action",
        "tool_proposal",
        "requires_clarification",
        "clarification_reason",
        "answer_candidate",
        "policy_fact_ids",
        "handoff_requested",
        "end_call_requested",
    ],
    "properties": {
        "spoken_response": {"type": "string"},
        "intent": {
            "type": "string",
            "enum": [
                "answer", "customer_question", "clarification", "correction",
                "confusion", "refusal", "unrelated", "interruption",
                "incomplete", "silence", "unknown",
            ],
        },
        "conversation_action": {
            "type": "string",
            "enum": [
                "respond", "answer_policy_then_resume", "request_clarification",
                "propose_record_answer", "propose_get_policy_fact",
                "propose_flag_risk", "propose_handoff", "propose_end_call", "none",
            ],
        },
        "tool_proposal": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                },
            ]
        },
        "requires_clarification": {"type": "boolean"},
        "clarification_reason": {"type": ["string", "null"]},
        "answer_candidate": {"type": ["string", "null"]},
        "policy_fact_ids": {"type": "array", "items": {"type": "string"}},
        "handoff_requested": {"type": "boolean"},
        "end_call_requested": {"type": "boolean"},
    },
}
