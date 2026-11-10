"""Grouped model selection with an untouched holdout and auditable row handling."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightbridge.domain import schedule_category
from flightbridge.labels import CATEGORICAL, NUMERIC, TARGETS, prepare_data
from flightbridge.nlp.data import canonical_airline

FEATURES = CATEGORICAL + NUMERIC
SEED = 42


def feature_groups(frame: pd.DataFrame) -> np.ndarray:
    """Keep identical predictor combinations together, regardless of outcome."""
    return pd.util.hash_pandas_object(frame[FEATURES].fillna("MISSING"), index=False).to_numpy()


def load_training_data(path: Path):
    frame, audit = prepare_data(pd.read_csv(path))
    exclusions = []
    for index, row in frame.iterrows():
        try:
            expected = schedule_category(row.delay_minutes, row.sc_type == "CR SC")
        except ValueError:
            expected = None
        if expected != row.sc_type:
            exclusions.append({"case_id": int(row.case_id), "reason": "inconsistent_or_missing_duration"})
    excluded = {row["case_id"] for row in exclusions}
    frame = frame[~frame.case_id.isin(excluded)].copy().reset_index(drop=True)
    frame["airline"] = frame.airline.map(canonical_airline)
    return frame, {
        "input_rows": len(pd.read_csv(path)),
        "retained_rows": len(frame),
        "excluded_rows": exclusions,
        "label_audit_counts": audit.severity.value_counts().to_dict(),
    }


def preprocessor():
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC,
            ),
        ]
    )


def candidates():
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(C=1, class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="lbfgs",
            alpha=1.0,
            max_iter=1000,
            random_state=SEED,
        ),
    }


def make_pipeline(estimator):
    return Pipeline([("preprocess", preprocessor()), ("classifier", estimator)])


def train(path: Path, output: Path) -> dict:
    frame, audit = load_training_data(path)
    groups = feature_groups(frame)
    development, holdout = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(frame, groups=groups)
    )
    assert not set(groups[development]) & set(groups[holdout])
    report = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seed": SEED,
        "data_audit": audit,
        "protocol": "25% grouped holdout; model selection on development-only grouped CV macro F1",
        "groups": "identical model-input combinations",
        "group_overlap": 0,
        "development_case_ids": frame.iloc[development].case_id.astype(int).tolist(),
        "holdout_case_ids": frame.iloc[holdout].case_id.astype(int).tolist(),
        "targets": {},
        "limitations": [
            "Small observational dataset with unresolved policy provenance.",
            "Model scores are uncalibrated, not policy guarantees.",
            "Grouping prevents exact duplicate leakage, not airline or temporal leakage.",
            "The holdout must not guide subsequent model or threshold changes.",
        ],
    }
    train_frame = frame.iloc[development]
    bundle = {
        "schema_version": 1,
        "sklearn_version": sklearn.__version__,
        "targets": {},
        "known_airlines": sorted(train_frame.airline.unique()),
        "airline_type_map": train_frame.drop_duplicates("airline")
        .set_index("airline")
        .airline_type.to_dict(),
        "max_delay": float(train_frame.delay_minutes.max()),
        "dataset_sha256": report["dataset_sha256"],
    }
    for target in TARGETS:
        dev = train_frame.dropna(subset=[target]).copy()
        test = frame.iloc[holdout].dropna(subset=[target]).copy()
        if dev[target].nunique() != 2 or test[target].nunique() != 2:
            raise ValueError(
                "Both classes are required in development and holdout; collect more labelled data."
            )
        dev_groups = feature_groups(dev)
        folds = list(
            StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
                dev, dev[target], dev_groups
            )
        )
        for left, right in folds:
            if dev.iloc[left][target].nunique() != 2:
                raise ValueError("A training fold lacks a class; collect more labelled data.")
            assert not set(dev_groups[left]) & set(dev_groups[right])
        comparisons = {}
        for name, estimator in candidates().items():
            scores = cross_val_score(
                make_pipeline(estimator),
                dev[FEATURES],
                dev[target],
                cv=folds,
                scoring="f1_macro",
                error_score="raise",
            )
            comparisons[name] = {
                "macro_f1_mean": float(scores.mean()),
                "macro_f1_std": float(scores.std()),
                "folds": scores.tolist(),
            }
        winner = max(comparisons, key=lambda n: comparisons[n]["macro_f1_mean"])
        fitted = make_pipeline(candidates()[winner]).fit(dev[FEATURES], dev[target])
        predicted = fitted.predict(test[FEATURES])
        probabilities = fitted.predict_proba(test[FEATURES])[:, list(fitted.classes_).index("YES")]
        baseline = make_pipeline(DummyClassifier(strategy="most_frequent")).fit(dev[FEATURES], dev[target])
        info = {
            "selected_model": winner,
            "development_rows": len(dev),
            "holdout_rows": len(test),
            "development_classes": dev[target].value_counts().to_dict(),
            "holdout_classes": test[target].value_counts().to_dict(),
            "cv": comparisons,
            "holdout": {
                "accuracy": accuracy_score(test[target], predicted),
                "balanced_accuracy": balanced_accuracy_score(test[target], predicted),
                "macro_f1": f1_score(test[target], predicted, average="macro"),
                "baseline_macro_f1": f1_score(
                    test[target], baseline.predict(test[FEATURES]), average="macro"
                ),
                "brier_score": brier_score_loss((test[target] == "YES").astype(int), probabilities),
                "labels": ["NO", "YES"],
                "confusion_matrix": confusion_matrix(test[target], predicted, labels=["NO", "YES"]).tolist(),
                "classification_report": classification_report(
                    test[target], predicted, output_dict=True, zero_division=0
                ),
            },
        }
        report["targets"][target] = info
        bundle["targets"][target] = {"pipeline": fitted, "model_name": winner}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "entitlements.joblib")
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
