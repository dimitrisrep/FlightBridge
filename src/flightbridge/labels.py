"""Προετοιμασία ετικετών δικαιώματος από τις καταγεγραμμένες υποθέσεις.

Το solution αφορά την έκβαση ενός αιτήματος. REJECTED δεν σημαίνει NO
και στους δύο στόχους. Το αρχικό CSV δεν τροποποιείται ποτέ από αυτό το αρχείο.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "schedule_change_dataset.csv"
CATEGORICAL = ["sc_type", "alternative_provided", "airline", "airline_type"]
NUMERIC = ["delay_minutes"]
TARGETS = {
    "full_refund_allowed": "Δικαίωμα πλήρους επιστροφής χρημάτων",
    "rebooking_allowed": "Δικαίωμα επανέκδοσης εισιτηρίου",
}
REQUEST_TARGET = {"REFUND": "full_refund_allowed", "REBOOKING": "rebooking_allowed"}
# Οι τρεις αυτές στήλες δεν διαβάζονται από κανέναν κανόνα ή μοντέλο.
IGNORED_COLUMNS = ["rerouting_allowed", "codeshare_allowed", "policy_exception"]
AUDIT_COLUMNS = ["case_id", "csv_row", "severity", "field", "recorded_value", "training_value", "message"]


def prepare_data(raw):
    """Επιστρέφει (δεδομένα εκπαίδευσης, ευρήματα) χωρίς αλλαγή του raw.

    Κρατά κάθε ρητή ετικέτα YES/NO, εκτός από άμεση αντίφαση: τότε αφήνει
    μόνο τον συγκεκριμένο στόχο άγνωστο μέχρι να ελεγχθεί η καταχώριση.
    Συμπληρώνει κενές ετικέτες μόνο όταν το αποτέλεσμα τις τεκμηριώνει.
    """
    df = raw.copy().reset_index(drop=True)
    required = CATEGORICAL + NUMERIC + ["pap_request", "solution"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Λείπουν στήλες από το CSV: {sorted(missing)}")
    if "case_id" not in df:
        df["case_id"] = np.arange(1, len(df) + 1)
    if df["case_id"].isna().any() or df["case_id"].duplicated().any():
        raise ValueError("Το case_id πρέπει να είναι συμπληρωμένο και μοναδικό.")
    for column in ["rejection_reason", *TARGETS]:
        if column not in df:
            df[column] = np.nan
    text_columns = CATEGORICAL + ["pap_request", "solution", "rejection_reason", *TARGETS]
    for column in text_columns:
        df[column] = (
            df[column]
            .map(lambda v: str(v).strip().upper() if pd.notna(v) else np.nan)
            .replace({"": np.nan, "NAN": np.nan, "N/A": np.nan, "NA": np.nan})
            .astype(object)
        )

    issues = []

    def note(i, severity, field, old, new, message):
        issues.append(
            {
                "case_id": df.at[i, "case_id"],
                "csv_row": i + 2,
                "severity": severity,
                "field": field,
                "recorded_value": old,
                "training_value": new,
                "message": message,
            }
        )

    allowed = {
        "sc_type": {"TR SC", "MI SC", "MA SC", "CR SC"},
        "alternative_provided": {"YES", "NO"},
        "airline_type": {"LCC", "LEGACY"},
        "pap_request": {"REFUND", "REBOOKING"},
        "solution": {"REFUND", "FULL REFUND", "REBOOKING", "REJECTED"},
    }
    for column, values in allowed.items():
        invalid = df[column].notna() & ~df[column].isin(values)
        if invalid.any():
            raise ValueError(
                f"Άγνωστες τιμές στη στήλη {column}: {df.loc[invalid, column].unique().tolist()}"
            )
    if df[CATEGORICAL].isna().any().any():
        raise ValueError("Συμπλήρωσε τα κατηγορικά χαρακτηριστικά της περίπτωσης.")
    type_counts = df.groupby("airline")["airline_type"].nunique()
    if (type_counts > 1).any():
        raise ValueError(
            "Η ίδια εταιρεία έχει διαφορετικό airline_type: " + ", ".join(type_counts[type_counts > 1].index)
        )

    raw_delay = df["delay_minutes"].copy()
    df["delay_minutes"] = pd.to_numeric(raw_delay, errors="coerce")
    invalid_delay = raw_delay.notna() & (
        df["delay_minutes"].isna() | ~np.isfinite(df["delay_minutes"]) | df["delay_minutes"].lt(0)
    )
    for i in df.index[invalid_delay]:
        note(
            i,
            "REVIEW",
            "delay_minutes",
            raw_delay[i],
            np.nan,
            "Μη έγκυρη διάρκεια. Δώσε μη αρνητικό αριθμό λεπτών. Χρησιμοποιείται imputation.",
        )
        df.at[i, "delay_minutes"] = np.nan
    cancelled_without_alternative = (
        df["sc_type"].eq("CR SC")
        & df["alternative_provided"].eq("NO")
        & df["delay_minutes"].isna()
        & ~invalid_delay
    )
    for i in df.index[cancelled_without_alternative]:
        df.at[i, "delay_minutes"] = 0
        note(
            i,
            "INFO",
            "delay_minutes",
            np.nan,
            0,
            "Ακύρωση χωρίς εναλλακτική: 0 ως κωδικός απουσίας νέας πτήσης.",
        )

    for target in TARGETS:
        df[target + "_source"] = "explicit"
        df.loc[df[target].isna(), target + "_source"] = "unknown"
        invalid = df[target].notna() & ~df[target].isin(["YES", "NO"])
        for i in df.index[invalid]:
            note(
                i,
                "CONFLICT",
                target,
                df.at[i, target],
                np.nan,
                "Η ετικέτα πρέπει να είναι YES, NO ή κενή. Εξαιρείται μόνο αυτός ο στόχος.",
            )
            df.at[i, target] = np.nan
            df.at[i, target + "_source"] = "conflict"

    for i, row in df.iterrows():
        evidence = {}
        if row["solution"] == "REBOOKING":
            evidence["rebooking_allowed"] = "YES"
        elif row["solution"] == "FULL REFUND":
            evidence["full_refund_allowed"] = "YES"
        elif row["solution"] == "REJECTED" and row["rejection_reason"] in {"NOT ELIGIBLE", "SHORT SC"}:
            # Απορρίπτεται μόνο το συγκεκριμένο αίτημα. NO AVAIL δεν μπαίνει εδώ.
            target = REQUEST_TARGET.get(row["pap_request"])
            if target:
                evidence[target] = "NO"

        for target, expected in evidence.items():
            recorded = df.at[i, target]
            source = df.at[i, target + "_source"]
            if source == "conflict":
                continue
            if pd.isna(recorded):
                df.at[i, target] = expected
                df.at[i, target + "_source"] = "derived_from_solution"
                note(
                    i,
                    "FILLED",
                    target,
                    recorded,
                    expected,
                    "Συμπλήρωση μόνο του τεκμηριωμένου δικαιώματος από την έκβαση/αιτία απόρριψης.",
                )
            elif recorded != expected:
                df.at[i, target] = np.nan
                df.at[i, target + "_source"] = "conflict"
                note(
                    i,
                    "CONFLICT",
                    target,
                    recorded,
                    np.nan,
                    f"Το αίτημα/αποτέλεσμα τεκμηριώνει {expected}, αλλά γράφτηκε {recorded}. "
                    "Εξαιρείται μόνο αυτός ο στόχος μέχρι έλεγχο στο αρχικό CSV.",
                )
            else:
                df.at[i, target + "_source"] = "explicit_confirmed_by_solution"

        # REFUND από μόνο του δεν διευκρινίζει πλήρη ή μερική επιστροφή.
        if row["solution"] == "REFUND" and row["full_refund_allowed"] != "YES":
            note(
                i,
                "REVIEW",
                "full_refund_allowed",
                row["full_refund_allowed"],
                df.at[i, "full_refund_allowed"],
                "REFUND χωρίς full_refund_allowed=YES: έλεγξε αν ήταν πλήρες ή μερικό. "
                "Η υπάρχουσα ετικέτα δεν αλλάζει χωρίς τεκμηρίωση.",
            )

    for target in TARGETS:
        for i in df.index[df[target + "_source"].eq("unknown")]:
            note(
                i,
                "MISSING",
                target,
                np.nan,
                np.nan,
                "Άγνωστο δικαίωμα. Δεν μετατρέπεται σε NO και δεν εκπαιδεύεται αυτός ο στόχος.",
            )
    return df, pd.DataFrame(issues, columns=AUDIT_COLUMNS)


def load_data(path=DATA_FILE):
    df, audit = prepare_data(pd.read_csv(path))
    df.attrs["audit"] = audit.to_dict("records")
    return df


def save_prepared_data(df, directory=BASE_DIR):
    directory = Path(directory)
    df.drop(columns=IGNORED_COLUMNS, errors="ignore").to_csv(
        directory / "prepared_dataset.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(df.attrs["audit"], columns=AUDIT_COLUMNS).to_csv(
        directory / "data_checks.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    data = load_data()
    save_prepared_data(data)
    print("Έτοιμα: prepared_dataset.csv και data_checks.csv. Το αρχικό CSV δεν άλλαξε.")
