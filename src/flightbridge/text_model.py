"""Small supervised intent baseline; bilingual fixtures are explicitly synthetic."""

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def intent_pipeline():
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True),
                        ),
                        (
                            "character",
                            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode"),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(C=4, max_iter=2000, class_weight="balanced")),
        ]
    )


def train_intents(training: Path, evaluation: Path, output: Path):
    train, test = read_jsonl(training), read_jsonl(evaluation)
    if {r["text"].casefold().strip() for r in train} & {r["text"].casefold().strip() for r in test}:
        raise ValueError("Intent train/evaluation text overlaps.")
    pipeline = intent_pipeline().fit([r["text"] for r in train], [r["intent"] for r in train])
    predicted = pipeline.predict([r["text"] for r in test])
    report = {
        "dataset_type": "authored synthetic bilingual fixtures; not a real-world benchmark",
        "training_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "training_rows": len(train),
        "evaluation_rows": len(test),
        "macro_f1": f1_score([r["intent"] for r in test], predicted, average="macro"),
        "classification_report": classification_report(
            [r["intent"] for r in test], predicted, output_dict=True, zero_division=0
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "intent.joblib")
    (output / "intent_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
