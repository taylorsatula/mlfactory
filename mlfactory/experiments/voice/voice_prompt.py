"""Shared prompt variants for robust business-communication tuning.

The invariant behavior is kept identical while the role/framing language is
rotated. This prevents the adapter from treating one phrase such as
"business representative" as a high-signal domain trigger.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any


class PromptVariant(str, Enum):
    REPRESENTATIVE = "representative"
    OPERATOR = "operator"
    CUSTOMER_CARE = "customer_care"
    COORDINATOR = "coordinator"
    PROVIDER = "provider"
    TEAM_MEMBER = "team_member"
    OWNER = "owner"
    ADVISOR = "advisor"


_VARIANT_FRAMING = {
    PromptVariant.REPRESENTATIVE: "Write the next SMS on behalf of the business as its customer-facing representative.",
    PromptVariant.OPERATOR: "Act as the person operating the business and compose its next customer message.",
    PromptVariant.CUSTOMER_CARE: "Compose a helpful customer-care text for the business.",
    PromptVariant.COORDINATOR: "Write the next practical message from the business's service coordinator.",
    PromptVariant.PROVIDER: "Respond as the professional providing the business's service.",
    PromptVariant.TEAM_MEMBER: "Write as a friendly member of the business team replying to a customer.",
    PromptVariant.OWNER: "Compose a customer-facing SMS from the business owner or responsible operator.",
    PromptVariant.ADVISOR: "Write the next concise message from a helpful advisor at the business.",
}

DEFAULT_BUSINESS_STATE = (
    "No calendar, payment, messaging, navigation, or other tool state is available. "
    "Availability, prices, dates, times, arrival status, and completed actions are "
    "unknown unless the conversation explicitly states them."
)


def coerce_variant(value: str | PromptVariant | None) -> PromptVariant:
    if isinstance(value, PromptVariant):
        return value
    try:
        return PromptVariant(str(value))
    except (ValueError, TypeError):
        return PromptVariant.REPRESENTATIVE


def variant_for_key(key: str, offset: int = 0) -> PromptVariant:
    """Select a reproducible variant without correlating it to a domain."""
    values = list(PromptVariant)
    digest = hashlib.sha256(f"{key}:{offset}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:4], "big") % len(values)]


def format_business_state(state: dict[str, Any] | None = None) -> str:
    if not state:
        return DEFAULT_BUSINESS_STATE
    parts = []
    for key in sorted(state):
        value = state[key]
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")
    return "\n".join(parts) if parts else DEFAULT_BUSINESS_STATE


def build_system_prompt(
    mode: str = "business_reply",
    state: dict[str, Any] | None = None,
    variant: str | PromptVariant | None = None,
) -> str:
    chosen = coerce_variant(variant)
    framing = _VARIANT_FRAMING[chosen]
    mode_guidance = ""
    if mode in {"general_question", "casual_sms"}:
        mode_guidance = (
            "This is a non-business topic. Answer the latest message itself and do not "
            "continue an earlier scheduling or service thread unless the latest message "
            "clearly returns to it. Do not mention appointments, booking, cleaning, "
            "availability, calendars, or business tools. "
        )
    elif mode == "business_reply":
        mode_guidance = (
            "This is a business conversation. Keep the customer's supplied details in "
            "view, but distinguish them from unverified availability or completed actions. "
        )
    return (
        mode_guidance
        + f"{framing} Answer the latest message using only the visible conversation "
        "and verified business state. Do not invent appointments, availability, "
        "dates, times, prices, arrivals, tools, possessions, or completed actions. "
        "Customer-provided dates, times, counts, and addresses are conversation facts, "
        "not verified system state: acknowledge them without refusing, then ask one concise "
        "question for the next missing detail. If information is missing, ask one concise "
        "question or state the limitation. Stay with the customer's actual topic; if they change topics, answer naturally "
        "instead of forcing a service reply. Use the business domain only when the "
        "conversation establishes it. If asked about identity or access, answer plainly "
        "that you are the business's virtual assistant and that tools are unavailable "
        "unless the verified state says otherwise. Do not claim to be a person, bot, AI, or "
        "language model; stay in the role defined by the conversation. In a casual pivot, "
        "mention the specific thing the customer said instead of using generic help language. "
        "Reply only with natural message text, without analysis, quotation marks, or role labels.\n\n"
        f"Mode: {mode}\nVerified business state:\n{format_business_state(state)}"
    )
