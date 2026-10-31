"""Real-time conversational orchestration above the LiveKit transport seam."""
from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.compliance import VerificationEngine
from agent.conversation_agent import ConversationIntent, ConversationResponse, GeminiConversationAgent, ToolProposal
from agent.core import CHECK_ORDER, CallFlow, CheckVerdict, detect_flags
from agent.policy import Policy
from agent.session import ConsentStatus, SessionContext, SessionStatus
from agent.tools import ToolExecutionResult, ToolExecutor
from agent.workflow import PROMPTS, VoiceIO, format_policy_value, normalize_tts_text, re_explanation_for


logger = logging.getLogger(__name__)


class ConversationalCallWorkflow:
    """One customer call driven by Gemini interpretation and Python controls."""

    def __init__(self, io: VoiceIO, customer: Mapping[str, Any], policy: Policy, session: SessionContext, agent: GeminiConversationAgent, *, executor: ToolExecutor | None = None):
        self.io = io
        self.customer = customer
        self.policy = policy
        self.session = session
        self.agent = agent
        self.executor = executor or ToolExecutor()
        self.verifier = VerificationEngine()
        self.flow = CallFlow()
        self._finalized = False

    async def _speak(self, text: str) -> None:
        self.session.add_message("agent", text)
        await self.io.speak(normalize_tts_text(text))

    async def _listen(self) -> tuple[str, str]:
        text = await self.io.listen()
        message_id = self.session.add_message("customer", text)
        await self.io.publish({
            "type": "transcript",
            "session_id": self.session.session_id,
            "role": "customer",
            "text": text,
            "final": True,
            "ts_ms": 0,
        })
        for flag, quote in detect_flags(text):
            self.executor.execute(
                tool_name="flag_risk",
                session=self.session,
                arguments={"flag": flag.value, "evidence": quote, "message_id": message_id},
                policy=self.policy,
                customer=self.customer,
            )
        return text, message_id

    async def _turn(self, *, pending_check: str | None, attempt: int = 1) -> tuple[str, ConversationResponse, ToolExecutionResult | None]:
        transcript, message_id = await self._listen()
        relevant = ()
        if pending_check:
            fact = self.policy.get_fact(verification_check_id=pending_check)
            relevant = (fact.fact_id,) if fact.exists and fact.fact_id else ()
        response = await self.agent.respond(
            session=self.session,
            customer_message=transcript,
            policy=self.policy,
            customer_context=self.customer,
            relevant_fact_ids=relevant,
        )
        proposal = response.tool_proposal
        if proposal is None and pending_check and response.intent in {ConversationIntent.ANSWER, ConversationIntent.CORRECTION, ConversationIntent.REFUSAL}:
            proposal = ToolProposal(
                "record_answer",
                {
                    "check_id": pending_check,
                    "answer_candidate": response.answer_candidate or transcript,
                    "original_transcript": transcript,
                    "message_id": message_id,
                    "attempt_number": attempt,
                },
            )
        elif proposal is None and response.handoff_requested:
            proposal = ToolProposal("handoff_to_human", {"reason": response.clarification_reason or "human assistance requested", "message_id": message_id})
        elif proposal is None and response.end_call_requested:
            proposal = ToolProposal("end_call", {"reason": response.clarification_reason or "customer requested the call to end", "confirmation": True, "message_id": message_id})

        execution = None
        if proposal is not None:
            arguments = dict(proposal.arguments)
            if proposal.name == "record_answer" and pending_check:
                arguments.update({"check_id": pending_check, "original_transcript": transcript, "message_id": message_id, "attempt_number": attempt})
            elif proposal.name == "flag_risk":
                arguments.update({"evidence": transcript, "message_id": message_id})
            elif proposal.name == "handoff_to_human":
                arguments.setdefault("message_id", message_id)
            elif proposal.name == "end_call":
                arguments.setdefault("message_id", message_id)
            execution = self.executor.execute(
                tool_name=proposal.name,
                session=self.session,
                arguments=arguments,
                policy=self.policy,
                customer=self.customer,
            )
        await self._speak(response.spoken_response)
        for fact_id in response.policy_fact_ids:
            fact = self.policy.get_fact(fact_id=fact_id)
            if fact.exists:
                self.session.disclose_policy_fact(fact_id, message_id=self.session.conversation_history[-1].message_id)
        return transcript, response, execution

    async def _process_check(self, check_id: str) -> None:
        prompt = PROMPTS[check_id]
        question = self.session.set_pending_question(prompt, verification_check_id=check_id)
        await self._speak(prompt)
        attempt = 1
        while attempt <= 2:
            transcript, response, execution = await self._turn(pending_check=check_id, attempt=attempt)
            if execution is not None and execution.tool_name == "record_answer" and execution.success:
                status = execution.data.get("verification_status")
                result = self.session.verification_results.get(check_id)
                if status == CheckVerdict.PASS.value:
                    if attempt == 2 and result is not None:
                        result.verdict = CheckVerdict.PASS_AFTER_RETRY
                        result.retries = 1
                        self.session.update_answer_status(execution.data["answer_id"], result.verdict.value)
                        self.session.update_verification_result(check_id, result, resolved=True)
                    self.session.resolve_pending_question(question_id=question.question_id)
                    if check_id == "identity":
                        self.flow.advance("identity", accepted=True)
                    else:
                        self.flow.advance("check_passed")
                    self.session.set_stage(self.flow.stage.value)
                    return
                if status == CheckVerdict.REFUSED.value:
                    self.executor.execute(
                        tool_name="handoff_to_human",
                        session=self.session,
                        arguments={"reason": "customer refused a mandatory verification answer", "message_id": self.session.conversation_history[-2].message_id},
                        policy=self.policy,
                        customer=self.customer,
                    )
                    return
                if attempt == 1:
                    await self._speak(PROMPTS["retry"].format(explanation=re_explanation_for(check_id, self.policy)))
                    attempt = 2
                    continue
                self.session.resolve_pending_question(question_id=question.question_id)
                if check_id == "identity":
                    self.executor.execute(
                        tool_name="handoff_to_human",
                        session=self.session,
                        arguments={"reason": "identity verification failed"},
                        policy=self.policy,
                        customer=self.customer,
                    )
                    return
                self.flow.advance("check_passed")
                self.session.set_stage(self.flow.stage.value)
                return

            if self.session.status != SessionStatus.ACTIVE:
                return
            if response.intent == ConversationIntent.REFUSAL:
                self.executor.execute(
                    tool_name="handoff_to_human",
                    session=self.session,
                    arguments={"reason": "customer refused verification", "message_id": self.session.conversation_history[-2].message_id},
                    policy=self.policy,
                    customer=self.customer,
                )
                return
            # Customer questions, clarification requests, interruptions and
            # unrelated turns are conversational side turns, not verification
            # attempts. Keep the same Python-owned attempt number and pending
            # question until a record_answer operation succeeds.
            await self._speak(question.text)

    async def _finalize(self) -> dict[str, Any]:
        if self.session.final_verdict is not None:
            return dict(self.session.final_verdict)
        report = self.verifier.calculate_verdict(session=self.session)
        stored = self.session.store_final_verdict(report.as_dict())
        if self.session.status == SessionStatus.ACTIVE:
            self.session.complete(human_handoff=report.verdict != "green")
        await self.io.publish({"type": "verdict", "session_id": self.session.session_id, **stored})
        self._finalized = True
        return stored

    async def run(self) -> dict[str, Any]:
        self.flow.advance("connected")
        self.session.set_stage(self.flow.stage.value)

        consent_question = self.session.set_pending_question(PROMPTS["disclosure"], verification_check_id="consent")
        await self._speak(PROMPTS["disclosure"])
        consent, response, _ = await self._turn(pending_check=None)
        if self.session.status != SessionStatus.ACTIVE:
            return await self._finalize()
        normalized_consent = consent.casefold()
        accepted = any(word in normalized_consent for word in ("haan", "ha", "yes", "sahmat", "हाँ", "जी"))
        refused = any(word in normalized_consent for word in ("nahi", "nahin", "no", "नहीं"))
        if refused or not accepted:
            self.session.consent_status = ConsentStatus.REFUSED
            self.session.resolve_pending_question(question_id=consent_question.question_id)
            self.executor.execute(
                tool_name="handoff_to_human",
                session=self.session,
                arguments={"reason": "customer did not consent to continue", "message_id": self.session.conversation_history[-2].message_id},
                policy=self.policy,
                customer=self.customer,
            )
            await self._speak(PROMPTS["human"])
            return await self._finalize()
        self.session.consent_status = ConsentStatus.GRANTED
        self.session.resolve_pending_question(question_id=consent_question.question_id)
        self.flow.advance("consent", accepted=True)
        self.session.set_stage(self.flow.stage.value)

        await self._process_check("identity")
        if self.session.status != SessionStatus.ACTIVE:
            return await self._finalize()
        identity = self.session.verification_results.get("identity")
        if identity is None or identity.verdict != CheckVerdict.PASS:
            return await self._finalize()

        facts = PROMPTS["facts"].format(
            product_name=self.policy["product_name"],
            premium_amount=format_policy_value(self.policy["premium_amount"], kind="amount"),
            premium_term_years=format_policy_value(self.policy["premium_term_years"], kind="duration"),
            policy_term_years=format_policy_value(self.policy["policy_term_years"], kind="duration"),
        )
        facts_question = self.session.set_pending_question(facts, verification_check_id="facts_confirmation")
        await self._speak(facts)
        facts_answer, facts_response, _ = await self._turn(pending_check=None)
        if self.session.status != SessionStatus.ACTIVE:
            return await self._finalize()
        self.session.resolve_pending_question(question_id=facts_question.question_id)
        if any(word in facts_answer.casefold() for word in ("nahi", "nahin", "galat", "wrong", "नहीं")):
            self.executor.execute(
                tool_name="handoff_to_human",
                session=self.session,
                arguments={"reason": "customer did not confirm the policy facts"},
                policy=self.policy,
                customer=self.customer,
            )
            await self._speak(PROMPTS["human"])
            return await self._finalize()
        self.flow.advance("facts_confirmed")
        self.session.set_stage(self.flow.stage.value)

        for check_id in CHECK_ORDER:
            await self._process_check(check_id)
            if self.session.status != SessionStatus.ACTIVE:
                break
        report = await self._finalize()
        await self._speak(PROMPTS["wrap_green"] if report["verdict"] == "green" else PROMPTS["human"])
        return report
