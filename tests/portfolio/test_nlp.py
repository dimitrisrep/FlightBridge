import pytest

from flightbridge.service import FlightBridge


@pytest.mark.parametrize(
    "message,intent",
    [
        ("I do not want a refund, I want rebooking", "REBOOKING"),
        ("Δεν θέλω επιστροφή χρημάτων, θέλω άλλη πτήση", "REBOOKING"),
        ("My Aegean flight was cancelled. I want a refund. No alternative offered.", "REFUND"),
        ("Η πτήση μου ακυρώθηκε και θέλω επιστροφή χρημάτων", "REFUND"),
    ],
)
def test_intent_with_negation_and_bilingual_messages(message, intent):
    assert FlightBridge().analyze(message)["facts"]["intent"] == intent


def test_unrelated_duration_is_not_a_delay():
    assert "delay_minutes" not in FlightBridge().analyze("I contacted support 2 days ago")["facts"]


def test_compound_duration_and_alias():
    result = FlightBridge().analyze("My A3 flight moved by 5 hours and 20 minutes. I want a refund.")
    assert result["facts"]["delay_minutes"] == 320
    assert result["facts"]["airline"] == "AEGEAN"


def test_multiple_requests_require_clarification():
    result = FlightBridge().analyze("I want a refund and I want rebooking")
    assert result["status"] == "clarification_required"
    assert "intent" in result["conflicts"]


def test_baggage_is_routed_away_from_entitlement_model():
    assert FlightBridge().analyze("My suitcase is lost")["route"] == "specialist_support"


def test_no_schedule_change_has_first_line_route():
    assert FlightBridge().analyze("I have no schedule change")["route"] == "first_line"


def test_no_message_is_automatically_confirmed():
    assert FlightBridge().analyze("Please refund my ticket")["confirmed"] is False


def test_model_does_not_overrule_negated_intent():
    class ConfidentModel:
        classes_ = ["REFUND", "REBOOKING"]

        def predict_proba(self, messages):
            return [[0.99, 0.01]]

    result = FlightBridge(intent_model=ConfidentModel()).analyze("I do not want a refund")
    assert result["facts"].get("intent") is None
