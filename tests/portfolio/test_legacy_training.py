import importlib.util
import sys
from pathlib import Path


def test_legacy_training_script_can_be_imported_and_evaluated():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("legacy_training", root / "train_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = module.load_data().dropna(subset=["full_refund_allowed"])
        results = module.evaluate_models(
            frame[module.CATEGORICAL + module.NUMERIC], frame.full_refund_allowed
        )
        assert set(results.index) == {"Decision Tree", "Random Forest", "k-NN (k=5)"}
        assert results.attrs["n_splits"] == 5
        assert all(0 <= score <= 1 for score in results["F1-score"])
    finally:
        sys.path.remove(str(root))
