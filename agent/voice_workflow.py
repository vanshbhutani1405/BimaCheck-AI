"""Real-time conversational orchestration above the LiveKit transport seam."""
from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from agent.compliance import MANDATORY_VERIFICATION_CHECKS, VerificationEngine
from agent.conversation_agent import ConversationIntent, ConversationResponse, GeminiConversationAgent, ToolProposal
from agent.core import CHECK_ORDER, CallFlow, CheckVerdict, detect_flags
from agent.policy import Policy
from agent.session import ConsentStatus, SessionContext, SessionStatus
from agent.tools import ToolExecutionResult, ToolExecutor
from agent.workflow import VoiceIO, format_policy_value, normalize_tts_text, re_explanation_for


logger = logging.getLogger(__name__)

VOICE_PROMPTS = {
    "disclosure": "नमस्ते! मैं BimaCheck से बात कर रहा हूँ। आपकी insurance policy की कुछ जानकारी verify करनी है। क्या हम शुरू कर सकते हैं?",
    "identity": "सबसे पहले, क्या मैं आपका नाम जान सकता हूँ?",
    "identity_confirm": "धन्यवाद। क्या आपका नाम {name} है?",
    "facts": "आपकी policy {product_name} एक {product_type} pension product है, fixed deposit नहीं। Annual premium {premium_amount} रुपये है और यह premium हर साल {premium_term_years} साल तक भरना है। आपकी policy {policy_term_years} साल तक चलेगी। Policy record के मुताबिक payout {payout_recipient} को मिलेगा और pension {payout_timing} शुरू होगी। {stop_paying_explanation}। क्या यह जानकारी सही है?",
    "premium_amount": "आपका annual premium कितना है?",
    "premium_term": "आपको हर साल premium कितने साल तक भरना होगा?",
    "policy_term": "आपकी policy कितने साल तक चलेगी?",
    "product_type": "यह product fixed deposit है या pension product?",
    "payout_recipient": "Policy record के मुताबिक payout किसे मिलेगा?",
    "payout_timing": "Pension कब शुरू होगी?",
    "stop_paying": "अगर आप premium देना बंद कर देते हैं, तो क्या हो सकता है?",
    "affordability": "क्या यह premium आपकी आमदनी के हिसाब से manageable है?",
    "retry": "मैं इसे थोड़ा आसान तरीके से समझाता हूँ: {explanation}। अब आप अपने शब्दों में फिर से बताइए।",
    "discrepancy": "Record के मुताबिक {record_value}। आपने {heard} बताया है, इसलिए इसे review के लिए note कर रहा हूँ। चलिए बाकी जानकारी भी verify करते हैं।",
    "unclear": "मैं आपका जवाब साफ़ नहीं समझ पाया। इसे review के लिए note कर रहा हूँ। चलिए बाकी जानकारी भी verify करते हैं।",
    "human": "धन्यवाद। इस call को human review के लिए भेजा जा रहा है।",
    "wrap_green": "धन्यवाद। आपकी जानकारी review के लिए दर्ज कर ली गई है।",
}


def _record_value_for(check_id: str, policy: Mapping[str, Any]) -> str:
    if check_id == "premium_amount":
        return f"annual premium {format_policy_value(policy['premium_amount'], kind='amount')} रुपये है"
    if check_id == "premium_term":
        return f"premium हर साल {format_policy_value(policy['premium_term_years'], kind='duration')} तक भरना है"
    if check_id == "policy_term":
        return f"policy {format_policy_value(policy['policy_term_years'], kind='duration')} तक चलेगी"
    if check_id == "product_type":
        return f"यह {policy['product_type']} है"
    if check_id == "payout_recipient":
        return f"payout {policy['payout_recipient']} को मिलेगा"
    if check_id == "payout_timing":
        return f"pension {format_policy_value(policy['payout_start_after_years'], kind='duration')} बाद शुरू होगी"
    return str(policy["stop_paying_explanation"])


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
        self._turn_sequence = 0

    def _update_latency(self, **marks: Any) -> None:
        update = getattr(self.io, "update_latency", None)
        if callable(update):
            update(**marks)

    @staticmethod
    def _customer_requested_end(transcript: str) -> bool:
        """Require explicit customer language before honoring end_call."""
        normalized = transcript.casefold()
        return any(
            phrase in normalized
            for phrase in (
                "end the call", "stop the call", "call khatam", "call band",
                "baat khatam", "baat band", "baat nahi karni", "कॉल खत्म",
                "कॉल बंद", "बात खत्म", "बात बंद", "बात नहीं करनी",
            )
        )

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
        self._turn_sequence += 1
        timing = getattr(self.io, "last_transcript_timing", None)
        if isinstance(timing, tuple) and len(timing) == 2:
            turn_id, transcript_final_at = timing
        else:
            turn_id, transcript_final_at = self._turn_sequence, time.perf_counter()
        self._update_latency(turn_id=turn_id, transcript_final_at=transcript_final_at)
        relevant = ()
        if pending_check:
            fact = self.policy.get_fact(verification_check_id=pending_check)
            relevant = (fact.fact_id,) if fact.exists and fact.fact_id else ()
        gemini_started_at = time.perf_counter()
        self._update_latency(gemini_started_at=gemini_started_at)
        logger.info(
            "LATENCY turn=%s GEMINI_REQUEST_START STT_TO_GEMINI_MS=%.1f",
            turn_id,
            (gemini_started_at - transcript_final_at) * 1000,
        )
        response = await self.agent.respond(
            session=self.session,
            customer_message=transcript,
            policy=self.policy,
            # Keep the expected identity out of Gemini's input until Python
            # has completed the explicit confirmation step.
            customer_context=self.customer if self.session.identity_verified else {},
            relevant_fact_ids=relevant,
        )
        gemini_completed_at = time.perf_counter()
        python_started_at = gemini_completed_at
        self._update_latency(gemini_completed_at=gemini_completed_at, python_started_at=python_started_at)
        logger.info(
            "LATENCY turn=%s GEMINI_RESPONSE_RECEIVED GEMINI_MS=%.1f",
            turn_id,
            (gemini_completed_at - gemini_started_at) * 1000,
        )
        logger.info("LATENCY turn=%s PYTHON_WORK_START", turn_id)
        proposal = response.tool_proposal
        if proposal is not None and proposal.name == "record_answer" and pending_check is None:
            logger.warning(
                "record_answer_proposal_without_pending_check",
                extra={"session_id": self.session.session_id},
            )
            proposal = None
        if proposal is not None and proposal.name == "end_call" and not self._customer_requested_end(transcript):
            logger.warning(
                "end_call_proposal_without_explicit_customer_request",
                extra={"session_id": self.session.session_id},
            )
            proposal = None
        in_mandatory_check = pending_check is not None and pending_check == self.flow.current_check
        current_mandatory_answer = (
            in_mandatory_check
            and response.intent not in {ConversationIntent.CUSTOMER_QUESTION, ConversationIntent.CLARIFICATION, ConversationIntent.UNRELATED}
        )
        explicit_end_call = (
            (response.end_call_requested or (proposal is not None and proposal.name == "end_call"))
            and self._customer_requested_end(transcript)
        )
        if in_mandatory_check and not explicit_end_call:
            if proposal is not None and proposal.name in {"handoff_to_human", "end_call"}:
                proposal = None
            if not current_mandatory_answer and proposal is not None and proposal.name == "record_answer":
                proposal = None
        preserve_control_action = explicit_end_call or (
            pending_check != "identity"
            and not in_mandatory_check
            and (
                response.handoff_requested
                or (proposal is not None and proposal.name in {"handoff_to_human", "end_call"})
            )
        )
        python_evaluates_turn = (
            pending_check == "identity"
            or current_mandatory_answer
        )
        if python_evaluates_turn and not preserve_control_action:
            # The pending Python check owns evaluation. Gemini's intent cannot
            # redirect an identity turn, and an unrelated tool proposal cannot
            # redirect an answer to the current mandatory check.
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
        elif proposal is None and pending_check and response.intent in {ConversationIntent.ANSWER, ConversationIntent.CORRECTION, ConversationIntent.REFUSAL}:
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
        elif proposal is None and response.handoff_requested and not in_mandatory_check:
            proposal = ToolProposal("handoff_to_human", {"reason": response.clarification_reason or "human assistance requested", "message_id": message_id})
        elif proposal is None and response.end_call_requested and self._customer_requested_end(transcript):
            proposal = ToolProposal("end_call", {"reason": response.clarification_reason or "customer requested the call to end", "confirmation": True, "message_id": message_id})

        execution = None
        evaluation_started_at = None
        evaluation_completed_at = None
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
            if proposal.name == "record_answer":
                evaluation_started_at = time.perf_counter()
            execution = self.executor.execute(
                tool_name=proposal.name,
                session=self.session,
                arguments=arguments,
                policy=self.policy,
                customer=self.customer,
            )
            if evaluation_started_at is not None:
                evaluation_completed_at = time.perf_counter()
            if (
                proposal.name == "record_answer"
                and not execution.success
                and pending_check
                and response.intent in {ConversationIntent.ANSWER, ConversationIntent.CORRECTION, ConversationIntent.REFUSAL}
            ):
                logger.warning(
                    "invalid_record_answer_proposal_replaced_with_python_canonical_arguments",
                    extra={"session_id": self.session.session_id, "check_id": pending_check, "error_code": execution.error_code},
                )
                execution = self.executor.execute(
                    tool_name="record_answer",
                    session=self.session,
                    arguments={
                        "check_id": pending_check,
                        "answer_candidate": response.answer_candidate or transcript,
                        "original_transcript": transcript,
                        "message_id": message_id,
                        "attempt_number": attempt,
                    },
                    policy=self.policy,
                    customer=self.customer,
                )
                if evaluation_started_at is not None:
                    evaluation_completed_at = time.perf_counter()
        spoken_response = response.spoken_response
        identity_result = self.session.verification_results.get("identity")
        identity_matched = identity_result is not None and identity_result.verdict in {
            CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY,
        }
        if not identity_matched:
            expected_names = (
                str(self.customer.get("name", "")),
                *(str(value) for value in self.customer.get("name_variants", ())),
            )
            spoken_folded = spoken_response.casefold()
            if any(name and name.casefold() in spoken_folded for name in expected_names):
                spoken_response = "जी, धन्यवाद।"
        if (
            pending_check is not None
            and pending_check == self.flow.current_check
            and execution is not None
            and execution.tool_name == "record_answer"
        ):
            # Python will decide whether to retry or ask the next check. Do not
            # let Gemini's answer-turn wording announce a different question.
            spoken_response = "जी, धन्यवाद।"
        pending_question = self.session.get_pending_question()
        python_transition_turn = pending_question is not None and pending_question.verification_check_id in {
            "consent", "identity", "identity_confirmation", "facts_confirmation",
        }
        python_completed_at = time.perf_counter()
        self._update_latency(
            python_completed_at=python_completed_at,
            evaluation_started_at=evaluation_started_at,
            evaluation_completed_at=evaluation_completed_at,
        )
        logger.info(
            "LATENCY turn=%s PYTHON_WORK_END PYTHON_MS=%.1f EVAL_MS=%s",
            turn_id,
            (python_completed_at - python_started_at) * 1000,
            "unavailable" if evaluation_started_at is None or evaluation_completed_at is None
            else f"{(evaluation_completed_at - evaluation_started_at) * 1000:.1f}",
        )
        if not python_transition_turn:
            await self._speak(spoken_response)
        for fact_id in response.policy_fact_ids:
            fact = self.policy.get_fact(fact_id=fact_id)
            if fact.exists:
                self.session.disclose_policy_fact(fact_id, message_id=self.session.conversation_history[-1].message_id)
        return transcript, response, execution

    async def _process_check(self, check_id: str, *, advance: bool = True) -> None:
        if check_id != "identity" and self.flow.current_check != check_id:
            raise ValueError(f"{check_id!r} is not the current mandatory check")
        prompt = VOICE_PROMPTS[check_id]
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
                    if check_id == "identity" and advance:
                        self.flow.advance("identity", accepted=True)
                    elif check_id != "identity":
                        self.flow.advance("check_passed")
                    self.session.set_stage(self.flow.stage.value)
                    return
                if check_id != "identity":
                    self.session.resolve_pending_question(question_id=question.question_id)
                    if result is not None:
                        self.session.update_verification_result(check_id, result, resolved=True)
                    if status == CheckVerdict.FAIL.value and result is not None:
                        await self._speak(VOICE_PROMPTS["discrepancy"].format(
                            record_value=_record_value_for(check_id, self.policy), heard=transcript
                        ))
                    else:
                        await self._speak(VOICE_PROMPTS["unclear"])
                    self.flow.advance("check_passed")
                    self.session.set_stage(self.flow.stage.value)
                    return
                if attempt == 1:
                    await self._speak(VOICE_PROMPTS["retry"].format(explanation=re_explanation_for(check_id, self.policy)))
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
                self.executor.execute(
                    tool_name="handoff_to_human",
                    session=self.session,
                    arguments={"reason": f"mandatory verification failed after retry: {check_id}"},
                    policy=self.policy,
                    customer=self.customer,
                )
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
        lifecycle_complete = (
            self.session.identity_verified
            and self.session.policy_explanation_completed
            and all(check_id in self.session.completed_checks for check_id in MANDATORY_VERIFICATION_CHECKS)
        )
        if lifecycle_complete:
            self.session.workflow_complete = True
        if lifecycle_complete and self.session.status == SessionStatus.ACTIVE:
            self.session.complete(human_handoff=report.verdict != "green")
        await self.io.publish({"type": "verdict", "session_id": self.session.session_id, **stored})
        self._finalized = True
        return stored

    async def _confirm_identity(self) -> bool:
        """Require an explicit customer confirmation after name matching."""
        variants = self.customer.get("name_variants", ())
        display_name = next((str(value) for value in variants if str(value).strip()), str(self.customer.get("name", "")))
        question = self.session.set_pending_question(
            VOICE_PROMPTS["identity_confirm"].format(name=display_name),
            verification_check_id="identity_confirmation",
        )
        await self._speak(question.text)
        confirmation, _, _ = await self._turn(pending_check=None)
        if self.session.status != SessionStatus.ACTIVE:
            return False
        self.session.resolve_pending_question(question_id=question.question_id)
        normalized = confirmation.casefold()
        refused = any(word in normalized for word in ("nahi", "nahin", "no", "नहीं"))
        accepted = any(word in normalized for word in ("haan", "ha", "yes", "sahmat", "हाँ", "जी"))
        if refused or not accepted:
            return False
        return True

    async def run(self) -> dict[str, Any]:
        self.flow.advance("connected")
        self.session.set_stage(self.flow.stage.value)

        consent_question = self.session.set_pending_question(VOICE_PROMPTS["disclosure"], verification_check_id="consent")
        await self._speak(VOICE_PROMPTS["disclosure"])
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
            await self._speak(VOICE_PROMPTS["human"])
            return await self._finalize()
        self.session.consent_status = ConsentStatus.GRANTED
        self.session.resolve_pending_question(question_id=consent_question.question_id)
        self.flow.advance("consent", accepted=True)
        self.session.set_stage(self.flow.stage.value)

        await self._process_check("identity", advance=False)
        if self.session.status != SessionStatus.ACTIVE:
            return await self._finalize()
        identity = self.session.verification_results.get("identity")
        if identity is None or identity.verdict != CheckVerdict.PASS:
            return await self._finalize()

        if not await self._confirm_identity():
            self.executor.execute(
                tool_name="handoff_to_human",
                session=self.session,
                arguments={"reason": "customer did not confirm the configured identity"},
                policy=self.policy,
                customer=self.customer,
            )
            await self._speak(VOICE_PROMPTS["human"])
            return await self._finalize()
        self.flow.advance("identity", accepted=True)
        self.session.identity_verified = True
        self.session.set_stage(self.flow.stage.value)

        facts = VOICE_PROMPTS["facts"].format(
            product_name=self.policy["product_name"],
            premium_amount=format_policy_value(self.policy["premium_amount"], kind="amount"),
            premium_term_years=format_policy_value(self.policy["premium_term_years"], kind="duration"),
            policy_term_years=format_policy_value(self.policy["policy_term_years"], kind="duration"),
            product_type=self.policy["product_type"],
            payout_recipient=self.policy["payout_recipient"],
            payout_timing=format_policy_value(self.policy["payout_start_after_years"], kind="duration"),
            stop_paying_explanation=self.policy["stop_paying_explanation"],
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
            await self._speak(VOICE_PROMPTS["human"])
            return await self._finalize()
        self.flow.advance("facts_confirmed")
        self.session.policy_explanation_completed = True
        self.session.set_stage(self.flow.stage.value)

        for check_id in CHECK_ORDER:
            if check_id not in MANDATORY_VERIFICATION_CHECKS:
                continue
            if self.flow.current_check != check_id:
                raise RuntimeError("workflow check order diverged from Python call state")
            await self._process_check(check_id)
            if self.session.status != SessionStatus.ACTIVE:
                break
            if self.flow.current_check == check_id:
                # A check that did not pass after its single retry is terminal;
                # never enter a later mandatory check.
                break
        report = await self._finalize()
        if report["verdict"] == "green":
            await self._speak("आपका verification पूरा हो गया है। आपकी दी गई सभी जानकारी policy record से match हुई है। Verification सफल रहा।")
        elif self.session.workflow_complete:
            labels = {
                "premium_amount": "annual premium", "premium_term": "premium भरने की अवधि",
                "policy_term": "policy की अवधि", "product_type": "product type",
                "payout_recipient": "payout पाने वाला", "payout_timing": "pension शुरू होने का समय",
                "stop_paying": "premium रोकने के परिणाम",
            }
            discrepancies = [labels[key] for key in report.get("mandatory_completed", ())
                             if key in labels and key in self.session.verification_results
                             and self.session.verification_results[key].verdict not in {CheckVerdict.PASS, CheckVerdict.PASS_AFTER_RETRY}]
            if discrepancies:
                await self._speak(f"आपका verification पूरा हो गया है। ये जानकारियाँ policy record से match नहीं हुईं: {', '.join(discrepancies)}। इसलिए इस case को human review के लिए भेजा जा रहा है।")
            else:
                await self._speak(VOICE_PROMPTS["human"])
        else:
            await self._speak(VOICE_PROMPTS["human"])
        return report
