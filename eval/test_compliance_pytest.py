from __future__ import annotations

from agent.compliance import MANDATORY_VERIFICATION_CHECKS, VerificationEngine
from agent.core import CheckResult, CheckVerdict, Flag


def _complete_check(session, policy, check_id, answer, customer=None, attempt=1):
    session.set_pending_question(check_id, verification_check_id=check_id)
    message_id = session.add_message("customer", answer)
    answer_id = session.record_answer(check_id, answer, attempt_number=attempt, message_id=message_id)
    result = VerificationEngine().evaluate_answer(
        session=session, policy=policy, check_id=check_id, answer_id=answer_id, customer=customer
    )
    session.resolve_pending_question(question_id=session.pending_question.question_id)
    return result


def test_identity_and_numeric_answers_are_authoritatively_evaluated(bound_policy, customer, session):
    assert _complete_check(session, bound_policy, "identity", "वंश", customer).verdict == CheckVerdict.PASS
    assert _complete_check(session, bound_policy, "premium_amount", "डेढ़ लाख").verdict == CheckVerdict.PASS
    assert _complete_check(session, bound_policy, "premium_term", "पाँच साल").verdict == CheckVerdict.PASS
    assert _complete_check(session, bound_policy, "policy_term", "बीस साल").verdict == CheckVerdict.PASS


def test_unclear_and_refused_answers_do_not_complete_check(bound_policy, session):
    unclear = _complete_check(session, bound_policy, "premium_amount", "लगभग उतना ही")
    assert unclear.verdict == CheckVerdict.UNCLEAR
    assert "premium_amount" in session.unresolved_checks
    refused_session = type(session)()
    refused_session.start()
    refused_session.set_pending_question("Again?", verification_check_id="premium_term")
    refused_session.add_message("customer", "मैं नहीं बताना चाहता")
    answer_id = refused_session.record_answer("premium_term", "मैं नहीं बताना चाहता")
    result = VerificationEngine().evaluate_answer(session=refused_session, policy=bound_policy, check_id="premium_term", answer_id=answer_id)
    assert result.verdict != CheckVerdict.PASS


def test_pass_after_retry_is_preserved_but_verdict_is_amber(bound_policy, session):
    for check_id in MANDATORY_VERIFICATION_CHECKS:
        answer = "वंश" if check_id == "identity" else {
            "premium_amount": "डेढ़ लाख", "premium_term": "पाँच साल", "policy_term": "बीस साल",
            "product_type": "pension", "payout_recipient": "Sunita", "payout_timing": "पाँच साल",
            "stop_paying": "policy may lapse", "affordability": "yes",
        }.get(check_id, "yes")
        result = _complete_check(session, bound_policy, check_id, answer, {"name": "Vansh", "name_variants": ("वंश",)} if check_id == "identity" else None)
        assert result.verdict in {CheckVerdict.PASS, CheckVerdict.UNCLEAR}
    result = session.verification_results["premium_amount"]
    result.verdict = CheckVerdict.PASS_AFTER_RETRY
    report = VerificationEngine().calculate_verdict(session=session)
    assert report.verdict == "amber"
    assert "mandatory_check_passed_after_retry" in report.reason_codes


def test_confirmed_critical_risks_are_red_and_unconfirmed_risks_are_not(session):
    session.add_risk_flag(Flag.GUARANTEE_CLAIM.value, "guaranteed return", confirmed=True)
    report = VerificationEngine().calculate_verdict(session=session)
    assert report.verdict == "red"
    assert Flag.GUARANTEE_CLAIM.value in report.confirmed_risks

    other = type(session)()
    other.start()
    other.add_risk_flag(Flag.GUARANTEE_CLAIM.value, "maybe a guarantee", confirmed=False)
    assert VerificationEngine().calculate_verdict(session=other).verdict == "amber"


def test_incomplete_and_advisory_affordability_never_become_green(bound_policy, session):
    session.set_pending_question("Premium?", verification_check_id="premium_amount")
    result = VerificationEngine().calculate_verdict(session=session)
    assert result.verdict == "amber"
    assert "premium_amount" in result.mandatory_outstanding
