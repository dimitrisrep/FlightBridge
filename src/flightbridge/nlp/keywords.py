"""Explainable bilingual keyword/phrase analysis. No API or NLP downloads needed.

Rules classify what the passenger says, never refund eligibility. The original
phrase and its offsets are kept so the UI can explain each classification.
"""

import re
import unicodedata

LABELS = {
    "REFUND": ("Refund", "Επιστροφή χρημάτων"),
    "REBOOKING": ("Rebooking", "Αλλαγή / νέα κράτηση"),
    "OPTIONS": ("Available options", "Διαθέσιμες επιλογές"),
    "COMPENSATION": ("Compensation", "Αποζημίωση"),
    "BAGGAGE": ("Baggage", "Αποσκευές"),
    "OTHER": ("Other request", "Άλλο αίτημα"),
    "CANCELLED": ("Flight cancellation", "Ακύρωση πτήσης"),
    "DELAYED": ("Flight delay", "Καθυστέρηση πτήσης"),
    "SCHEDULE_CHANGED": ("Schedule change", "Αλλαγή προγράμματος"),
    "VOLUNTARY_CANCELLATION": ("Passenger wants to cancel", "Ακύρωση με επιθυμία επιβάτη"),
    "NO_SCHEDULE_CHANGE": ("No schedule change", "Χωρίς schedule change"),
}

# Multiword phrases are kept together: "money back" carries one meaning.
RULES = [
    (
        "event",
        "NO_SCHEDULE_CHANGE",
        r"\bno schedule changes?\b|\bwithout (?:a |any )?schedule changes?\b|"
        r"\bi (?:don't|do not) have (?:a |any )?schedule changes?\b|"
        r"\b(?:my|the|our) flight (?:has not|hasn't) (?:been )?changed\b|"
        r"δεν (?:εχω|υπαρχει|εγινε) (?:καποια )?schedule change|"
        r"δεν (?:εχω|υπαρχει|εγινε) (?:καποια )?αλλαγη (?:στην πτηση|ωρας|προγραμματος)|"
        r"χωρις (?:schedule change|αλλαγη (?:στην πτηση|ωρας|προγραμματος))|"
        r"(?:η )?πτηση (?:μου |μας )?δεν (?:εχει αλλαξει|αλλαξε)",
    ),
    (
        "intent",
        "REFUND",
        r"\brefund(?:s)?\b|\breimburse(?:ment|ments| me)?\b|\bmoney back\b|"
        r"επιστροφ\w* χρηματων|(?:τα )?(?:χρηματα|λεφτα) (?:μου )?πισω|"
        r"επιστραφ\w* (?:τα |μου )*(?:χρηματα|λεφτα)|"
        r"^\s*(?:θελω )?επιστροφη(?: παρακαλω)?[.! ]*$",
    ),
    (
        "intent",
        "REBOOKING",
        r"\bre[ -]?book(?:ing)?\b|\breschedul(?:e|ing)\b|"
        r"αλλαγη (?:της )?(?:πτησης|εισιτηριου|κρατησης)|επανεκδοση|"
        r"(?:θελω|θελα|ηθελα|χρειαζομαι) (?:μια )?(?:αλλη|νεα|διαφορετικη) πτηση|"
        r"(?:θελω|μπορω|μπορειτε) να αλλαξ\w* (?:την |το |μου )*(?:πτηση|εισιτηριο)|"
        r"(?:want|need|like) (?:to (?:take|book) )?(?:a |an )?(?:new|another|different) flight|"
        r"(?:change|switch) (?:my|the|our) (?:flight|ticket|booking)",
    ),
    ("intent", "COMPENSATION", r"\bcompensation\b|\bcompensate\b|αποζημιω\w*"),
    ("intent", "BAGGAGE", r"\bbaggage\b|\bluggage\b|\bsuitcases?\b|αποσκευ\w*|βαλιτσ\w*"),
    (
        "intent",
        "OPTIONS",
        r"\boptions\b|what can i do|what am i entitled to|"
        r"επιλογες|τι (?:μπορω να κανω|δικαιουμαι)",
    ),
    (
        "event",
        "VOLUNTARY_CANCELLATION",
        r"\bi (?:want to |need to |would like to )?cancell?(?:ed)?\b|"
        r"\bcan (?:i|you) cancel (?:my|the|our) flight|"
        r"(?:θελω|ηθελα|μπορω|μπορειτε) να ακυρωσ\w*|(?:θελω|ηθελα) ακυρωση|εγω ακυρωσα",
    ),
    ("event", "CANCELLED", r"\bcancell?(?:ed|ation|ations)\b|ακυρωθηκε|ακυρωσαν|ακυρωμεν\w*|ακυρωση"),
    ("event", "DELAYED", r"\bdelay(?:ed|s)?\b|καθυστερ\w*"),
    (
        "event",
        "SCHEDULE_CHANGED",
        r"\brescheduled\b|\bschedule change\b|\bchanged\b|\bmoved\b|"
        r"\bearlier\b|\blater\b|αλλαξε|αλλαξαν|μετακινηθηκε|νωριτερα|αργοτερα|"
        r"αλλαγη (?:της |στην |του |στο )?(?:ωρας|ωρα|προγραμματος|προγραμμα)",
    ),
]
COMPILED_RULES = [(group, code, re.compile(pattern)) for group, code, pattern in RULES]
STOP_WORDS = set(
    """i me my we our the a an is was were are have has had been be to of for
    with and or but that this it in on at by do does did want would like please can you
    η ο το τα τη την τον της του τις τους μου μας σας με σε για απο και ή αλλα ενα μια
    ειναι ηταν εχω εχει εχουν ειχε να θα δεν θελω θα ηθελα παρακαλω μπορω μου""".split()
)
CLAUSE_BREAK = re.compile(
    r"[.!?;,\n]|\b(?:but|however|so|αλλα|ομως|οποτε|αρα)\b|"
    r"\b(?:and|και)\s+(?=(?:i|we|they|θελω|ηθελα|χρειαζομαι)\b)"
)
NEGATION = re.compile(
    r"\b(?:not|no|never|don't|do not|didn't|did not|haven't|hasn't|"
    r"wasn't|isn't|cannot|can't|without|δεν|οχι|χωρις|μην)\b"
)
UNCERTAINTY = re.compile(
    r"\b(?:if|unless|whether|maybe|might|not sure|don't know|do not know|"
    r"αν|ισως|μηπως|δεν ξερω|δεν γνωριζω|δεν ειμαι σιγουρ\w*)\b"
)


def normalized_with_offsets(message):
    """Normalize accents/case while retaining offsets into the original string."""
    chars, offsets = [], []
    for index, char in enumerate(message):
        for normalized in unicodedata.normalize("NFD", char.lower().replace("’", "'")):
            if not unicodedata.combining(normalized):
                chars.append(normalized)
                offsets.append(index)
    return "".join(chars), offsets


def label(code, language="en"):
    return LABELS.get(code, (code, code))[1 if language == "el" else 0]


def clause_before(text, start):
    breaks = list(CLAUSE_BREAK.finditer(text[:start]))
    return text[breaks[-1].end() if breaks else 0 : start]


def polarity(text, start, end, group):
    before = clause_before(text, start)
    if UNCERTAINTY.search(before):
        return "uncertain"
    # "not only a refund but also ..." is not rejection of a refund.
    if NEGATION.search(re.sub(r"\bnot only\b|\bοχι μονο\b", "", before)):
        return "negated"
    if group == "event":
        # "Was my flight cancelled?" asks a question; it is not a confirmed event.
        tail = re.split(r"[.!?;\n]", text[end:], maxsplit=1)[0]
        terminator = text[end + len(tail) : end + len(tail) + 1]
        if terminator in {"?", ";"} and re.match(
            r"\s*(?:was|is|has|did|will|μηπως|ακυρωθηκε|καθυστερησε)\b", before + text[start:end]
        ):
            return "uncertain"
    return "affirmed"


def _event_is_about_other_topic(text, start):
    before = clause_before(text, start)
    topics = list(
        re.finditer(
            r"\b(?:refund|baggage|luggage|flight|plane)\b|"
            r"επιστροφ\w* χρηματων|αποσκευ\w*|βαλιτσ\w*|πτηση|αεροπλανο",
            before,
        )
    )
    return bool(topics and not re.fullmatch(r"flight|plane|πτηση|αεροπλανο", topics[-1].group()))


def analyze_message(message):
    text, offsets = normalized_with_offsets(message)
    tokens = re.findall(r"[^\W_]+(?:'[^\W_]+)?", text)
    content_keywords = list(dict.fromkeys(t for t in tokens if t not in STOP_WORDS))
    matches = []
    for group, code, pattern in COMPILED_RULES:
        for match in pattern.finditer(text):
            # A rescheduling request and a reported reschedule have different roles.
            status = polarity(text, match.start(), match.end(), group)
            if group == "event" and _event_is_about_other_topic(text, match.start()):
                status = "other_context"
            if code == "CANCELLED" and (
                re.match(r"\s+(?:fee|fees|policy|policies)\b", text[match.end() :])
                or re.search(r"(?:κοστος|τελος|πολιτικη)\s*$", text[: match.start()])
            ):
                status = "other_context"
            if code == "REBOOKING" and re.search(
                r"\b(?:offered|offers|offering|suggested)\b|προτειν\w*|προτειναν",
                clause_before(text, match.start()),
            ):
                status = "other_context"
            begin, end = offsets[match.start()], offsets[match.end() - 1] + 1
            matches.append(
                {
                    "category": group,
                    "code": code,
                    "phrase": message[begin:end],
                    "normalized": match.group(),
                    "start": begin,
                    "end": end,
                    "status": status,
                }
            )

    # "I cancelled" must never become "the airline cancelled" via an overlapping hit.
    voluntary = [m for m in matches if m["code"] == "VOLUNTARY_CANCELLATION"]
    for hit in matches:
        if hit["code"] == "CANCELLED" and any(
            v["start"] <= hit["start"] and hit["end"] <= v["end"] for v in voluntary
        ):
            hit["status"] = "other_context"
    matches.sort(key=lambda m: (m["start"], -(m["end"] - m["start"])))
    intents = list(
        dict.fromkeys(m["code"] for m in matches if m["category"] == "intent" and m["status"] == "affirmed")
    )
    events = list(
        dict.fromkeys(m["code"] for m in matches if m["category"] == "event" and m["status"] == "affirmed")
    )
    # An OR between the two solution phrases explicitly asks to compare options.
    solution_hits = [m for m in matches if m["code"] in {"REFUND", "REBOOKING"} and m["status"] == "affirmed"]
    alternative_options = False
    for first, second in zip(solution_hits, solution_hits[1:]):
        if first["code"] != second["code"] and re.search(
            r"\b(?:or|η)\b", normalized_with_offsets(message[first["end"] : second["start"]])[0]
        ):
            alternative_options = True
    primary = intents[0] if len(intents) == 1 else None
    if alternative_options and set(intents) <= {"REFUND", "REBOOKING", "OPTIONS"}:
        primary = "OPTIONS"
    elif "OPTIONS" in intents and len(set(intents) - {"OPTIONS"}) == 1:
        primary = next(code for code in intents if code != "OPTIONS")
    return {
        "method": "keyword_rules_v2",
        "text": message,
        "tokens": tokens,
        "content_keywords": content_keywords,
        "keywords": list(dict.fromkeys(m["phrase"] for m in matches if m["status"] == "affirmed")),
        "matches": matches,
        "intents": intents,
        "event_types": events,
        "primary_intent": primary,
        "needs_intent_clarification": bool(intents and primary is None),
    }
