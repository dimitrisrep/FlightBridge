import json
from pathlib import Path

import numpy as np
import pytest

from flightbridge.domain import CaseFacts
from flightbridge.service import FlightBridge
from flightbridge.training import feature_groups, load_training_data, train

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("artifacts")
    report = train(ROOT / "schedule_change_dataset.csv", output)
    return FlightBridge.load(output), report


def test_real_data_excludes_inconsistent_durations():
    frame, audit = load_training_data(ROOT / "schedule_change_dataset.csv")
    assert audit["input_rows"] == 181
    assert {r["case_id"] for r in audit["excluded_rows"]} == {7, 13, 35, 72, 141, 145}
    assert len(frame) == 175


def test_holdout_has_no_repeated_predictor_combinations(trained):
    _, report = trained
    frame, _ = load_training_data(ROOT / "schedule_change_dataset.csv")
    dev = frame[frame.case_id.isin(report["development_case_ids"])]
    test = frame[frame.case_id.isin(report["holdout_case_ids"])]
    assert not set(feature_groups(dev)) & set(feature_groups(test))
    assert all(v["holdout"]["labels"] == ["NO", "YES"] for v in report["targets"].values())
    json.dumps(report)


def test_confirmation_gate_and_unknown_airline(trained):
    service, _ = trained
    facts = CaseFacts(
        airline="NOT A KNOWN CARRIER", airline_type="LEGACY", alternative_provided="YES", delay_minutes=60
    )
    assert service.predict(facts)["status"] == "confirmation_required"
    result = service.predict(facts, confirmed=True)
    assert result["predictions"] == {}
    assert result["reasons"] == ["airline_not_seen_in_training"]


def test_real_model_outputs_and_no_automatic_actions(trained):
    service, _ = trained
    airline = service.bundle["known_airlines"][0]
    facts = CaseFacts(
        airline=airline,
        airline_type=service.bundle["airline_type_map"][airline],
        alternative_provided="YES",
        delay_minutes=60,
    )
    result = service.handoff(facts, "REFUND", confirmed=True)
    assert result["actions_executed"] == []
    assert len(result["prediction"]["predictions"]) == 2
    for info in result["prediction"]["predictions"].values():
        assert info["label"] in {"YES", "NO"}
        assert np.isfinite(info["model_score"])
        assert 0 <= info["model_score"] <= 1
