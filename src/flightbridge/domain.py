"""Serving contracts and schedule categories shared by all entry points."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    airline: str = Field(min_length=1, max_length=80)
    airline_type: Literal["LCC", "LEGACY"]
    alternative_provided: Literal["YES", "NO"]
    cancelled: bool = False
    delay_minutes: float | None = Field(default=None, ge=0, le=525600)

    @model_validator(mode="after")
    def require_duration(self):
        if self.cancelled and self.alternative_provided == "NO" and self.delay_minutes not in (None, 0):
            raise ValueError("A cancellation without a replacement has no schedule difference.")
        if self.delay_minutes is None and not (self.cancelled and self.alternative_provided == "NO"):
            raise ValueError("A non-negative schedule difference in minutes is required.")
        return self

    @property
    def sc_type(self) -> str:
        return schedule_category(self.delay_minutes, self.cancelled)


def schedule_category(minutes: float | None, cancelled: bool = False) -> str:
    """Cancellation overrides time. Exactly 300 minutes belongs to minor."""
    if cancelled:
        return "CR SC"
    if minutes is None or not 0 <= minutes <= 525600:
        raise ValueError("Invalid schedule difference.")
    return "TR SC" if minutes <= 30 else "MI SC" if minutes <= 300 else "MA SC"


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=4000)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: CaseFacts
    confirmed: bool = False


class HandoffRequest(PredictRequest):
    intent: Literal["REFUND", "REBOOKING", "OPTIONS", "COMPENSATION", "BAGGAGE", "OTHER"]
