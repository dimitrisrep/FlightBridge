import json
from pathlib import Path

import joblib
import pytest

from flightbridge.text_model import train_intents

ROOT = Path(__file__).resolve().parents[2]


def test_bilingual_training_round_trip_and_report(tmp_path):
    report = train_intents(ROOT / "data/intents/train.jsonl", ROOT / "data/intents/test.jsonl", tmp_path)
    assert report["training_rows"] == 48
    assert report["evaluation_rows"] == 24
    assert "synthetic" in report["dataset_type"]
    assert len(report["training_sha256"]) == 64
    model = joblib.load(tmp_path / "intent.joblib")
    assert model.predict(["I want a refund", "Χάθηκε η βαλίτσα μου"]).tolist() == ["REFUND", "BAGGAGE"]


def test_duplicate_text_is_rejected(tmp_path):
    content = json.dumps({"text": "same request", "intent": "REFUND"}) + "\n"
    training, evaluation = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    training.write_text(content, encoding="utf-8")
    evaluation.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        train_intents(training, evaluation, tmp_path / "output")
