"""Readable service responses; formatting only, with no prediction logic."""

from html import escape

INTENTS = {
    "REFUND": "Refund",
    "REBOOKING": "Rebooking",
    "OPTIONS": "Available options",
    "COMPENSATION": "Compensation",
    "BAGGAGE": "Baggage",
    "OTHER": "Other request",
}
AIRLINE_TYPES = {"LEGACY": "Full-service carrier", "LCC": "Low-cost carrier"}
EVENTS = {"CHANGED": "Schedule changed", "CANCELLED": "Flight cancelled"}
CATEGORIES = {
    "TR SC": "Small schedule change",
    "MI SC": "Minor schedule change",
    "MA SC": "Major schedule change",
    "CR SC": "Cancellation",
}
FIELDS = {
    "intent": "Passenger request",
    "airline": "Airline",
    "airline_type": "Airline type",
    "disruption": "Disruption",
    "delay_minutes": "Schedule difference",
    "alternative_provided": "Alternative flight",
    "journey": "Journey",
    "flight_number": "Flight number",
}
REASONS = {
    "airline_not_seen_in_training": "This airline was not represented in the training cases.",
    "airline_type_conflict": "The airline type differs from the training records.",
    "duration_outside_training_range": "The schedule difference exceeds the model's training range.",
}


def label(mapping, value, fallback="Not provided"):
    return mapping.get(value, str(value) if value is not None else fallback)


def duration(value):
    if value is None:
        return "Not provided"
    minutes = float(value)
    if minutes == int(minutes):
        hours, remainder = divmod(int(minutes), 60)
        return (
            f"{hours} h {remainder} min"
            if hours and remainder
            else f"{hours} h"
            if hours
            else f"{remainder} min"
        )
    return f"{minutes:g} min"


def case_rows(facts):
    no_difference = facts.get("cancelled") and facts.get("alternative_provided") == "NO"
    return [
        ("Airline", facts.get("airline") or "Not provided"),
        ("Airline type", label(AIRLINE_TYPES, facts.get("airline_type"))),
        ("Disruption", "Flight cancelled" if facts.get("cancelled") else "Schedule changed"),
        (
            "Alternative flight",
            {"YES": "Offered", "NO": "Not offered"}.get(facts.get("alternative_provided"), "Not provided"),
        ),
        ("Schedule difference", "Not applicable" if no_difference else duration(facts.get("delay_minutes"))),
    ]


def definition_list(rows):
    return (
        '<dl class="fact-list">'
        + "".join(
            f"<div><dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd></div>" for key, value in rows
        )
        + "</dl>"
    )


def signal_card(title, prediction):
    prediction = prediction or {}
    value = prediction.get("label")
    tone = "positive" if value == "YES" else "negative" if value == "NO" else "unavailable"
    display = f"Model signal: {value}" if value in {"YES", "NO"} else "No model signal"
    score = prediction.get("model_score")
    caption = (
        f"Uncalibrated model score · {score:.3f}"
        if isinstance(score, (int, float))
        else "Manual review needed"
    )
    return (
        f'<section class="signal-card {tone}"><div class="signal-label">{escape(title)}</div>'
        f'<div class="signal-value">{escape(display)}</div><div class="signal-caption">{escape(caption)}</div></section>'
    )


def handoff_text(handoff):
    prediction = handoff.get("prediction", {})
    lines = [
        "FLIGHTBRIDGE | AGENT HANDOFF",
        "",
        "Status: agent review required",
        "Request: " + label(INTENTS, handoff.get("intent")),
        "",
    ]
    lines += [f"{key}: {value}" for key, value in case_rows(handoff.get("confirmed_facts") or {})]
    if category := prediction.get("sc_type"):
        lines.append("Schedule category: " + label(CATEGORIES, category))
    lines += ["", "RECORDED-CASE MODEL SIGNALS"]
    for target, title in [("full_refund_allowed", "Refund"), ("rebooking_allowed", "Rebooking")]:
        item = prediction.get("predictions", {}).get(target)
        lines.append(
            f"{title}: {item['label']} (uncalibrated score {item['model_score']:.3f})"
            if item
            else f"{title}: no model signal"
        )
    lines += [REASONS.get(reason, reason) for reason in prediction.get("reasons", [])]
    lines += [
        "",
        "NEXT STEP",
        handoff.get("next_step", "Verify airline policy and availability."),
        "Model scores are not probabilities of entitlement. No booking or refund has been executed.",
    ]
    return "\n".join(lines) + "\n"
