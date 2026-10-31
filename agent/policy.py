"""Structured synthetic policy loading and approved fact retrieval."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "suraksha_pension_plus.json"


@dataclass(frozen=True)
class PolicyFactResult:
    exists: bool
    available: bool
    fact_id: str | None
    value: Any = None
    explanation: str | None = None
    policy_version: str | None = None
    limitation: str | None = None
    section: str | None = None
    verification_check_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "available": self.available,
            "fact_id": self.fact_id,
            "value": self.value,
            "explanation": self.explanation,
            "policy_version": self.policy_version,
            "limitation": self.limitation,
            "section": self.section,
            "verification_check_id": self.verification_check_id,
        }


class Policy(Mapping[str, Any]):
    """Validated policy with a compatibility mapping for the current engine."""

    def __init__(self, data: Mapping[str, Any], *, source: Path, customer_context: Mapping[str, Any] | None = None):
        self._data = dict(data)
        self.source = source
        self.policy_id = self._data["policy_id"]
        self.policy_version = self._data["policy_version"]
        self._legacy = self._build_legacy_mapping()
        if customer_context is not None and "declared_income" in customer_context:
            self._legacy["declared_income"] = customer_context["declared_income"]

    def __getitem__(self, key: str) -> Any:
        return self._legacy[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._legacy)

    def __len__(self) -> int:
        return len(self._legacy)

    def get(self, key: str, default: Any = None) -> Any:
        return self._legacy.get(key, default)

    def _build_legacy_mapping(self) -> dict[str, Any]:
        return {
            "product_name": self._value("product_identity.product_name"),
            "premium_amount": self._value("premium.amount"),
            "premium_frequency": self._value("premium.frequency"),
            "premium_term_years": self._value("premium.payment_term_years"),
            "policy_term_years": self._value("policy_duration.total_term_years"),
            "payout_start_after_years": self._value("benefits_and_payouts.payout_timing", "years_after_start"),
            "product_type": self._value("product_identity.product_type"),
            "payout_recipient": self._value("benefits_and_payouts.payout_recipient"),
            "stop_paying_explanation": self._value("discontinuance.consequences"),
        }

    def _fact_node(self, fact_id: str) -> tuple[dict[str, Any] | None, str | None]:
        parts = fact_id.split(".")
        if len(parts) < 2:
            return None, None
        node: Any = self._data.get(parts[0])
        for part in parts[1:]:
            if not isinstance(node, Mapping):
                return None, parts[0]
            node = node.get(part)
        return node if isinstance(node, dict) else None, parts[0]

    def _value(self, fact_id: str, nested_key: str = "value") -> Any:
        node, _ = self._fact_node(fact_id)
        if not node or not node.get("available", False):
            return None
        return node.get(nested_key)

    def get_fact(
        self,
        *,
        fact_id: str | None = None,
        section: str | None = None,
        verification_check_id: str | None = None,
    ) -> PolicyFactResult:
        if fact_id is None and verification_check_id is not None:
            fact_id = self._data.get("verification_mapping", {}).get(verification_check_id)
        if fact_id is None and section is not None:
            section_node = self._data.get(section)
            if isinstance(section_node, Mapping):
                return PolicyFactResult(
                    True,
                    True,
                    section,
                    dict(section_node),
                    policy_version=self.policy_version,
                    section=section,
                )
            fact_id = section
        if not fact_id:
            return PolicyFactResult(False, False, None, policy_version=self.policy_version)

        node, node_section = self._fact_node(fact_id)
        mapped_check = next(
            (key for key, mapped in self._data.get("verification_mapping", {}).items() if mapped == fact_id),
            verification_check_id,
        )
        if node is None:
            return PolicyFactResult(False, False, fact_id, policy_version=self.policy_version, verification_check_id=mapped_check)
        return PolicyFactResult(
            True,
            bool(node.get("available", False)),
            fact_id,
            node.get("value"),
            node.get("explanation"),
            self.policy_version,
            node.get("limitation"),
            node_section,
            mapped_check,
        )

    def disclosed_fact_ids(self) -> tuple[str, ...]:
        return tuple(self._data.get("verification_mapping", {}).values())

    def with_customer_context(self, customer: Mapping[str, Any]) -> "Policy":
        """Bind customer-only affordability data without adding it to policy JSON."""
        return Policy(self._data, source=self.source, customer_context=customer)

    def approved_explanation(self, explanation_id: str) -> str | None:
        value = self._data.get("approved_explanations", {}).get(explanation_id)
        return value if isinstance(value, str) else None


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"policy field {path!r} must be an object")
    return value


def _validate(data: Mapping[str, Any], source: Path) -> None:
    if not isinstance(data.get("policy_id"), str) or not data["policy_id"]:
        raise ValueError(f"policy {source} requires a non-empty string policy_id")
    if not isinstance(data.get("policy_version"), str) or not data["policy_version"]:
        raise ValueError(f"policy {source} requires a non-empty string policy_version")
    sections = ("product_identity", "premium", "policy_duration", "benefits_and_payouts", "discontinuance", "exclusions_and_limitations", "approved_explanations", "verification_mapping")
    for section in sections:
        _require_mapping(data.get(section), section)
    for field in ("product_name", "product_category", "product_type", "currency"):
        node = _require_mapping(data["product_identity"].get(field), f"product_identity.{field}")
        if not isinstance(node.get("available"), bool):
            raise ValueError(f"policy field product_identity.{field}.available must be boolean")
        if node["available"] and not isinstance(node.get("value"), str):
            raise ValueError(f"available policy field product_identity.{field}.value must be a string")
    for field in ("amount", "frequency", "payment_term_years", "payment_details"):
        node = _require_mapping(data["premium"].get(field), f"premium.{field}")
        if not isinstance(node.get("available"), bool):
            raise ValueError(f"policy field premium.{field}.available must be boolean")
        if node["available"] and node.get("value") is None:
            raise ValueError(f"available policy field premium.{field}.value is required")
    for fact_id in ("policy_duration.total_term_years", "benefits_and_payouts.payout_recipient", "benefits_and_payouts.payout_timing", "discontinuance.consequences"):
        parts = fact_id.split(".")
        node = _require_mapping(data[parts[0]].get(parts[1]), fact_id)
        if not isinstance(node.get("available"), bool):
            raise ValueError(f"policy field {fact_id}.available must be boolean")
        if node["available"] and node.get("value") is None:
            raise ValueError(f"available policy field {fact_id}.value is required")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in data["verification_mapping"].items()):
        raise ValueError("verification_mapping must contain string check IDs and fact IDs")


def load_policy(path: str | Path | None = None) -> Policy:
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        with policy_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"synthetic policy file not found: {policy_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"synthetic policy file is not valid JSON: {policy_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"synthetic policy root must be an object: {policy_path}")
    _validate(data, policy_path)
    return Policy(data, source=policy_path)


def get_policy_fact(
    policy: Policy,
    *,
    fact_id: str | None = None,
    section: str | None = None,
    verification_check_id: str | None = None,
) -> PolicyFactResult:
    """Stable retrieval seam for the future ``get_policy_fact`` tool."""
    return policy.get_fact(
        fact_id=fact_id,
        section=section,
        verification_check_id=verification_check_id,
    )
