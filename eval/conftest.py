from __future__ import annotations

import pytest

from agent.policy import load_policy
from agent.session import SessionContext


@pytest.fixture
def policy():
    return load_policy()


@pytest.fixture
def customer():
    return {"name": "Vansh", "name_variants": ("वंश",), "declared_income": 600000}


@pytest.fixture
def bound_policy(policy, customer):
    return policy.with_customer_context(customer)


@pytest.fixture
def session():
    value = SessionContext()
    value.start()
    return value


@pytest.fixture
def pending_session(session):
    question = session.set_pending_question("Aapka premium kitna hai?", verification_check_id="premium_amount")
    return session, question
