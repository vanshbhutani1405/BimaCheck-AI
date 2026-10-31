from __future__ import annotations

import json

import pytest

from agent.policy import get_policy_fact, load_policy


def test_suraksha_policy_contains_approved_values(policy):
    assert policy["product_name"] == "Suraksha Pension Plus"
    assert policy["premium_amount"] == 150000
    assert policy["premium_term_years"] == 5
    assert policy["policy_term_years"] == 20
    assert policy["product_type"] == "deferred annuity"
    assert policy["payout_recipient"] == "Sunita"
    assert policy["payout_start_after_years"] == 5


def test_policy_fact_retrieval_by_id_section_and_check(policy):
    by_id = get_policy_fact(policy, fact_id="premium.amount")
    by_section = get_policy_fact(policy, section="premium")
    by_check = get_policy_fact(policy, verification_check_id="premium_amount")
    assert by_id.exists and by_id.available and by_id.value == 150000
    assert by_section.exists and by_section.section == "premium"
    assert by_check.fact_id == "premium.amount"


def test_unavailable_fact_is_explicit(policy):
    fact = policy.get_fact(fact_id="exclusions_and_limitations.known_limitations")
    assert fact.exists is True
    assert fact.available is False
    assert fact.value is None
    assert fact.limitation


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps({"policy_id": "only-id"})])
def test_malformed_policy_files_fail_closed(tmp_path, content):
    path = tmp_path / "policy.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(path)


def test_missing_policy_file_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_policy(tmp_path / "missing.json")


def test_policy_disclosure_tracking_is_per_policy_instance(policy):
    policy.disclosed_fact_ids()
    assert policy.get_fact(fact_id="premium.amount").fact_id == "premium.amount"
