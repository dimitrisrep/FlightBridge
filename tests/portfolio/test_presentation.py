from flightbridge.presentation import case_rows, definition_list, handoff_text, signal_card


def test_handoff_preserves_independent_signals_and_missing_cancellation_duration():
    handoff = {
        "intent": "REFUND",
        "confirmed_facts": {
            "airline": "AEGEAN",
            "airline_type": "LEGACY",
            "cancelled": True,
            "alternative_provided": "NO",
            "delay_minutes": None,
        },
        "prediction": {
            "sc_type": "CR SC",
            "predictions": {
                "full_refund_allowed": {"label": "YES", "model_score": 0.98},
                "rebooking_allowed": {"label": "NO", "model_score": 0.75},
            },
        },
        "next_step": "Verify the current airline policy and availability.",
    }
    text = handoff_text(handoff)
    assert "Refund: YES" in text and "Rebooking: NO" in text
    assert "Schedule difference: Not applicable" in text
    assert "Cancellation" in text and "not probabilities of entitlement" in text
    assert "No booking or refund has been executed" in text
    assert "NULL" not in text
    assert dict(case_rows(handoff["confirmed_facts"]))["Alternative flight"] == "Not offered"


def test_html_views_escape_passenger_values_and_do_not_invent_unavailable_scores():
    html = definition_list([("Airline", '<img src=x onerror="alert(1)">')])
    assert "<img" not in html and "&lt;img" in html
    assert "No model signal" in signal_card("Refund", None)
    assert "0.000" not in signal_card("Refund", None)
    assert "98%" not in signal_card("Refund", {"label": "YES", "model_score": 0.98})
