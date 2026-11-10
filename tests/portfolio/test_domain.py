import math

import pytest
from pydantic import ValidationError

from flightbridge.domain import CaseFacts, schedule_category


@pytest.mark.parametrize(
    "minutes,expected",
    [(0, "TR SC"), (30, "TR SC"), (30.5, "MI SC"), (31, "MI SC"), (300, "MI SC"), (300.1, "MA SC")],
)
def test_category_boundaries(minutes, expected):
    assert schedule_category(minutes) == expected


@pytest.mark.parametrize("minutes", [-1, None, math.inf, math.nan, 600000])
def test_invalid_duration(minutes):
    with pytest.raises(ValueError):
        schedule_category(minutes)


def test_cancellation_without_replacement_does_not_require_duration():
    facts = CaseFacts(airline="A3", airline_type="LEGACY", alternative_provided="NO", cancelled=True)
    assert facts.sc_type == "CR SC"


def test_cannot_supply_contradictory_category():
    with pytest.raises(ValidationError):
        CaseFacts(
            airline="A3", airline_type="LEGACY", alternative_provided="YES", delay_minutes=30, sc_type="MA SC"
        )


def test_changed_flight_needs_duration():
    with pytest.raises(ValidationError):
        CaseFacts(airline="A3", airline_type="LEGACY", alternative_provided="YES")


def test_no_replacement_rejects_a_fabricated_time_difference():
    with pytest.raises(ValidationError):
        CaseFacts(
            airline="A3", airline_type="LEGACY", alternative_provided="NO", cancelled=True, delay_minutes=100
        )
