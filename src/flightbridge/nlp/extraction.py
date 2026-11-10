"""Local keyword and phrase extraction for passenger requests."""

import re
from dataclasses import dataclass, field

from .data import AIRLINE_ALIASES, DURATION_PART, canonical_airline, normalize, parse_duration
from .keywords import analyze_message, polarity

FIELDS = [
    "intent",
    "airline",
    "airline_type",
    "disruption",
    "delay_minutes",
    "alternative_provided",
    "journey",
    "flight_number",
]
ENUMS = {
    "intent": {"REFUND", "REBOOKING", "OPTIONS", "COMPENSATION", "BAGGAGE", "OTHER"},
    "airline_type": {"LCC", "LEGACY"},
    "disruption": {"CANCELLED", "CHANGED", "OTHER", "NONE"},
    "alternative_provided": {"YES", "NO"},
}


@dataclass
class Extraction:
    facts: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    notices: list = field(default_factory=list)
    mode: str = "offline"
    analysis: dict = field(default_factory=dict)

    def put(self, key, value, quote):
        self.facts[key] = value
        self.evidence[key] = quote[:500]


def is_negated(text, start):
    before = text[max(0, start - 55) : start]
    return bool(
        re.search(
            r"(?:\bnot|\bno|\bdon't want|\bdo not want|\bδεν θελω|\bδεν|\bοχι|\bhasn't|\bwasn't|\bisn't)\s+(?:(?:been|a|an|any|the|μια|την|να)\s+)?$",
            before,
        )
    )


def positive_matches(pattern, text):
    return [m for m in re.finditer(pattern, text) if not is_negated(text, m.start())]


def extract_offline(message, awaiting=None):
    """Limited, deterministic demo parser. Unrecognized text remains unknown."""
    text = normalize(message)
    result = Extraction(analysis=analyze_message(message))
    uncertain = bool(
        re.search(r"don't know|do not know|not sure|δεν ξερω|δεν γνωριζω|δεν ειμαι σιγουρ", text)
    )
    if uncertain and awaiting in FIELDS:
        result.put(awaiting, None, message)
        result.notices.append("unknown_answer")
        return result

    airlines = {}
    for name, aliases in AIRLINE_ALIASES.items():
        for alias in sorted(set(aliases + [name]), key=len, reverse=True):
            match = re.search(r"(?<!\w)" + re.escape(normalize(alias)) + r"(?!\w)", text)
            if match and not is_negated(text, match.start()):
                airlines[name] = match.group()
                break
    if len(airlines) == 1:
        name, quote = next(iter(airlines.items()))
        result.put("airline", name, quote)
    elif len(airlines) > 1:
        result.conflicts.append("airline")
    elif awaiting == "airline" and not uncertain and len(text) < 65 and re.fullmatch(r"[\w\s\-\'\.]+", text):
        if not re.search(r"\b(yes|no|ναι|οχι|hello|hi|refund|rebook)\b", text):
            result.put("airline", canonical_airline(message), message)

    for pattern, value in [
        (r"\blcc\b|low.cost|χαμηλου κοστους", "LCC"),
        (r"\blegacy\b|full.service|παραδοσιακη", "LEGACY"),
    ]:
        found = positive_matches(pattern, text)
        if found:
            result.put("airline_type", value, found[-1].group())

    analysis = result.analysis
    if analysis["primary_intent"]:
        intent = analysis["primary_intent"]
        quotes = [
            m["phrase"] for m in analysis["matches"] if m["code"] == intent and m["status"] == "affirmed"
        ]
        result.put("intent", intent, quotes[-1] if quotes else message)
    elif analysis["needs_intent_clarification"]:
        result.conflicts.append("intent")
    elif awaiting == "intent" and re.fullmatch(r"other|κατι αλλο", text):
        result.put("intent", "OTHER", message)

    events = analysis["event_types"]
    cancellations = "CANCELLED" in events
    changes = bool(set(events) & {"DELAYED", "SCHEDULE_CHANGED"})
    voluntary = "VOLUNTARY_CANCELLATION" in events
    event_values = (
        ({"CANCELLED"} if cancellations else set())
        | ({"CHANGED"} if changes else set())
        | ({"OTHER"} if voluntary else set())
    )
    if "NO_SCHEDULE_CHANGE" in events:
        event_values.add("NONE")
    if len(event_values) > 1:
        result.conflicts.append("disruption")
    elif event_values:
        quote = next(
            m["phrase"] for m in analysis["matches"] if m["category"] == "event" and m["status"] == "affirmed"
        )
        result.put("disruption", next(iter(event_values)), quote)
    elif awaiting == "disruption" and re.fullmatch(r"other|κατι αλλο", text):
        result.put("disruption", "OTHER", message)

    negative_alt = re.search(
        r"no (?:alternative|replacement|other flight)|without (?:an? )?(?:alternative|replacement)|"
        r"(?:not|never|haven\x27t|didn\x27t|did not|have not) (?:been )?(?:received|offered|get|receive|offer)(?: me)?(?: an?| any)? (?:alternative|replacement|another flight)|"
        r"δεν (?:μου )?(?:εδωσαν|προτειναν|προταθηκε|εχω λαβει)(?: καποια)? (?:αλλη|εναλλακτικη)|χωρις εναλλακτικη",
        text,
    )
    positive_alt = re.search(
        r"(?:offered|received|proposed)(?: me)?(?: an?| another| a new)? (?:alternative|replacement|flight)|"
        r"(?:μου )?(?:προτειναν|εδωσαν) (?:μια )?(?:αλλη|εναλλακτικη) πτηση|εχω εναλλακτικη",
        text,
    )
    positive_alt_valid = (
        positive_alt
        and not uncertain
        and polarity(text, positive_alt.start(), positive_alt.end(), "entity") == "affirmed"
    )
    if negative_alt and positive_alt_valid:
        result.conflicts.append("alternative_provided")
    elif negative_alt and not uncertain:
        result.put("alternative_provided", "NO", negative_alt.group())
    elif positive_alt_valid:
        result.put("alternative_provided", "YES", positive_alt.group())
    elif awaiting == "alternative_provided":
        if re.fullmatch(r"yes[.! ]*|ναι[.! ]*", text):
            result.put("alternative_provided", "YES", message)
        elif re.fullmatch(r"no[.! ]*|οχι[.! ]*", text):
            result.put("alternative_provided", "NO", message)

    parts = [
        m
        for m in DURATION_PART.finditer(text)
        if not is_negated(text, m.start()) and not re.search(r"-\s*$", text[: m.start()])
    ]
    # Duration extraction is restricted to a schedule context to avoid treating
    # 'I contacted support 2 days ago' as a flight delay.
    short_correction = awaiting == "confirmation" and bool(
        re.search(r"\bactually\b|\bcorrection\b|λαθος|εννοουσα|τελικα", text)
    )
    schedule_context = awaiting == "delay_minutes" or bool(cancellations or changes) or short_correction
    if parts and schedule_context:
        groups = []
        for part in parts:
            if groups and re.fullmatch(r"\s*(?:and|και)?\s*", text[groups[-1][-1].end() : part.start()]):
                groups[-1].append(part)
            else:
                groups.append([part])
        # If another duration refers to an unrelated event, ask rather than guess.
        if len(groups) == 1 and not re.search(r"\bago\b|\bin \d+|σε \d+ .*μερε|πριν απο", text):
            quote = text[groups[0][0].start() : groups[0][-1].end()]
            value = parse_duration(quote)
            if value is not None:
                result.put("delay_minutes", value, quote)
        else:
            result.conflicts.append("delay_minutes")
    elif awaiting == "delay_minutes" and re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        result.put("delay_minutes", parse_duration(text), message)

    journey = re.search(r"\b([A-Z]{3})\s*(?:-|→|to)\s*([A-Z]{3})\b", message)
    if journey:
        result.put("journey", f"{journey[1]}–{journey[2]}", journey.group())
    flight = re.search(r"\b(?:flight|πτ[ηή]ση)\s+([A-Z0-9]{2}\s?\d{2,4})\b", message, re.I)
    if flight:
        result.put("flight_number", flight[1].replace(" ", "").upper(), flight.group())
    return result
