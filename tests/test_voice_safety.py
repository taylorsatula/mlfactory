from mlfactory.experiments.voice.voice_prompt import build_system_prompt
from mlfactory.experiments.voice.voice_safety import infer_mode, response_violation


def test_mode_inference_preserves_business_context_for_date_followups() -> None:
    assert infer_mode([
        {"role": "user", "content": "I need to book window cleaning."},
        {"role": "assistant", "content": "What date works?"},
        {"role": "user", "content": "August 27th and 22 windows."},
    ], "auto") == "business_reply"


def test_mode_inference_allows_explicit_casual_pivot() -> None:
    assert infer_mode([
        {"role": "user", "content": "I need to reschedule window cleaning."},
        {"role": "user", "content": "Actually, my dog is convinced every delivery is for him."},
    ], "auto") == "casual_sms"


def test_grounding_checks_identity_and_unverified_actions() -> None:
    assert response_violation("I am a real person.") == "identity_claim"
    assert response_violation("I can add that to the calendar.") == "unsupported_action"
    assert response_violation("The cleaning is scheduled for August 27th.") == "unsupported_action"
    assert response_violation("I will mark the availability.") == "unsupported_action"
    assert response_violation("I can send that message.") == "unsupported_external_action"
    assert response_violation("I will not send anything.") == "unsupported_external_action"


def test_non_business_prompt_explicitly_ends_service_thread() -> None:
    prompt = build_system_prompt("casual_sms")
    assert "non-business topic" in prompt
    assert "Do not mention appointments" in prompt
