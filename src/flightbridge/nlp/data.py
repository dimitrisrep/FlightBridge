"""Shared cleaning rules: training and chat must use the same units and names."""

import math
import re
import unicodedata


def normalize(value):
    text = unicodedata.normalize("NFD", str(value).lower().replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


# Alias normalization only; airline type is read from the supplied dataset.
AIRLINE_ALIASES = {
    "ITA AIRWAYS": ["ita", "ita airways", "az"],
    "QATAR AIRWAYS": ["qatar", "qatar airways", "qr"],
    "AIR FRANCE": ["air france", "af"],
    "TURKISH AIRLINES": ["turkish", "turkish airlines", "tk"],
    "BRITISH AIRWAYS": ["british airways", "ba"],
    "AIR CHINA": ["air china", "ca"],
    "AIR NEW ZEALAND": ["air new zealand", "nz"],
    "CHINA AIRLINES": ["china airlines", "ci"],
    "EVA AIR": ["eva air", "eva"],
    "AZERBAIJAN AIRLINES": ["azerbaijan airlines", "j2"],
    "SMARTWINGS": ["smartwings", "qs"],
    "VOLOTEA": ["volotea", "v7"],
    "TWAY AIR": ["t'way air", "tway air", "t way air"],
    "WIZZAIR": ["wizzair", "wizz air"],
    "AEGEAN": ["aegean", "aegean airlines", "αιγαιου", "a3"],
    "AIRASIA": ["airasia", "air asia"],
    "RYANAIR": ["ryanair", "ραιανερ"],
    "KLM": ["klm"],
    "LATAM": ["latam"],
    "ETIHAD": ["etihad"],
    "AUSTRIAN": ["austrian", "austrian airlines"],
    "SAS": ["sas", "scandinavian airlines"],
    "ANIMAWINGS": ["animawings"],
    "MYANMAR AIRWAYS": ["myanmar airways"],
    "IBERIA": ["iberia"],
    "FLYBONDI": ["flybondi"],
}
ALIAS_LOOKUP = {normalize(a): name for name, aliases in AIRLINE_ALIASES.items() for a in aliases + [name]}


def canonical_airline(value):
    if value is None or str(value).strip().lower() in {"", "nan", "n/a"}:
        return None
    return ALIAS_LOOKUP.get(normalize(value), str(value).strip().upper())


DURATION_PART = re.compile(
    r"(?P<number>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>days?|d\b|ημερες|ημερα|μερες|μερα|hours?|hrs?|h(?![a-z])|ωρες|ωρα|"
    r"minutes?|mins?|min|m\b|λεπτα|λεπτο)",
    re.I,
)


def parse_duration(value):
    """Return non-negative minutes, or None for absent/unparseable data.

    Numeric CSV cells already mean minutes. Chat only calls this on an explicit
    duration or on a direct answer to the delay question (also in minutes).
    """
    if value is None:
        return None
    text = normalize(value)
    if text in {"", "nan", "n/a", "none", "null"}:
        return None
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        number = float(text.replace(",", "."))
        return number if math.isfinite(number) else None
    # Signed values must not be silently turned into positive ones.
    if re.search(r"-\s*\d", text):
        return None
    parts = list(DURATION_PART.finditer(text))
    if not parts:
        return None
    total = 0.0
    for part in parts:
        number = float(part["number"].replace(",", "."))
        unit = part["unit"]
        multiplier = (
            1440 if unit.startswith(("d", "ημερ", "μερ")) else 60 if unit.startswith(("h", "ωρ")) else 1
        )
        total += number * multiplier
    return total if math.isfinite(total) else None
