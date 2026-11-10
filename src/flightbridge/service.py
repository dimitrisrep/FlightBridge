"""Combine language evidence, explicit confirmation and tabular predictions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from flightbridge.domain import CaseFacts
from flightbridge.nlp.data import canonical_airline
from flightbridge.nlp.extraction import extract_offline


class FlightBridge:
    def __init__(self, bundle: dict | None = None, intent_model=None):
        self.bundle = bundle
        self.intent_model = intent_model

    @classmethod
    def load(cls, directory: Path):
        # Artifacts are executable pickle data. Load only artifacts trained by you.
        bundle = joblib.load(directory / "entitlements.joblib")
        if bundle.get("schema_version") != 1 or bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("Artifact schema or scikit-learn version mismatch; retrain locally.")
        intent_path = directory / "intent.joblib"
        return cls(bundle, joblib.load(intent_path) if intent_path.exists() else None)

    def analyze(self, message: str) -> dict:
        if not message.strip() or len(message) > 4000:
            raise ValueError("Provide between 1 and 4000 characters.")
        extracted = extract_offline(message)
        facts = dict(extracted.facts)
        intent = facts.get("intent")
        intent_source = "rules" if intent else "unresolved"
        suggestion = None
        if self.intent_model is not None:
            scores = self.intent_model.predict_proba([message])[0]
            index = int(np.argmax(scores))
            suggestion = {"label": str(self.intent_model.classes_[index]), "score": float(scores[index])}
            # Rules retain precedence for negation, uncertainty and multiple requests.
            has_intent_mentions = any(
                m["category"] == "intent" for m in extracted.analysis.get("matches", [])
            )
            if not intent and not has_intent_mentions and not extracted.conflicts and scores[index] >= 0.55:
                intent, intent_source = suggestion["label"], "supervised_demo_model"
                facts["intent"] = intent
        if facts.get("airline"):
            facts["airline"] = canonical_airline(facts["airline"])
            if self.bundle:
                mapped = self.bundle["airline_type_map"].get(facts["airline"])
                if mapped:
                    facts["airline_type"] = mapped
        special = intent in {"BAGGAGE", "COMPENSATION", "OTHER"} or facts.get("disruption") in {
            "NONE",
            "OTHER",
        }
        required = ["intent", "disruption", "airline", "airline_type", "alternative_provided"]
        if not (facts.get("disruption") == "CANCELLED" and facts.get("alternative_provided") == "NO"):
            required.append("delay_minutes")
        missing = [] if special else [key for key in required if facts.get(key) is None]
        return {
            "facts": facts,
            "evidence": extracted.evidence,
            "conflicts": extracted.conflicts,
            "missing_fields": missing,
            "intent_source": intent_source,
            "model_suggestion": suggestion,
            "route": "first_line"
            if facts.get("disruption") == "NONE"
            else "specialist_support"
            if special
            else "schedule_change",
            "status": "specialist_review"
            if special
            else "clarification_required"
            if missing or extracted.conflicts
            else "confirmation_required",
            "confirmed": False,
        }

    def predict(self, facts: CaseFacts, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"status": "confirmation_required", "predictions": {}}
        if self.bundle is None:
            raise RuntimeError("No trained model is loaded.")
        airline = canonical_airline(facts.airline)
        reasons = []
        if airline not in self.bundle["known_airlines"]:
            reasons.append("airline_not_seen_in_training")
        elif self.bundle["airline_type_map"][airline] != facts.airline_type:
            reasons.append("airline_type_conflict")
        minutes = facts.delay_minutes if facts.delay_minutes is not None else 0.0
        if minutes > self.bundle["max_delay"]:
            reasons.append("duration_outside_training_range")
        if reasons:
            return {"status": "review_required", "reasons": reasons, "predictions": {}}
        frame = pd.DataFrame(
            [
                {
                    "sc_type": facts.sc_type,
                    "airline": airline,
                    "airline_type": facts.airline_type,
                    "alternative_provided": facts.alternative_provided,
                    "delay_minutes": minutes,
                }
            ]
        )
        predictions = {}
        for target, info in self.bundle["targets"].items():
            pipe = info["pipeline"]
            scores = pipe.predict_proba(frame)[0]
            index = int(np.argmax(scores))
            predictions[target] = {
                "label": str(pipe.classes_[index]),
                "model_score": float(scores[index]),
                "model": info["model_name"],
                "score_type": "uncalibrated_model_score",
            }
        return {
            "status": "agent_review",
            "sc_type": facts.sc_type,
            "predictions": predictions,
            "requires_agent_review": True,
            "dataset_sha256": self.bundle["dataset_sha256"],
        }

    def handoff(self, facts: CaseFacts, intent: str, confirmed: bool = False) -> dict:
        prediction = self.predict(facts, confirmed)
        return {
            "intent": intent,
            "confirmed_facts": facts.model_dump() if confirmed else None,
            "prediction": prediction,
            "next_step": "Verify the current airline policy and availability.",
            "actions_executed": [],
        }
