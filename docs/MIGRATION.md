# Source review and migration

## Inputs

| Repository | Reviewed commit | Reused components |
| --- | --- | --- |
| `dimitrisrep/Schedule_change_AI` | `2a0ef48981cc2b8cb03ce67860c07fbaef7ec2cc` | Label semantics, two separate targets, recorded dataset and model comparison concept |
| `dimitrisrep/hackathon` | `63a774dad05be6de30c1df83a0a27fca1446d67b` | English/Greek extraction, aliases, negation and explicit evidence |

## Findings

The original training file contained an uncommented explanatory line that prevented Python parsing. `evaluate_models` also returned an undefined `results` variable. Both defects are repaired; the original script remains a coursework entry point. The new package adds an independent, grouped evaluation rather than presenting the earlier CV figures as test performance.

The original hackathon language processing is rule-based. Its metrics file describes an earlier 47-row dataset while the current CSV contains 181 rows. The old metadata is not reused as evidence of current performance. No company or carrier deployment is claimed.

## Changes

`flightbridge` is a new installable package. It preserves the original label-preparation semantics, ports the parser into a package namespace and normalizes the `A3` alias consistently. It adds supervised intent experiments, canonical case validation, APIs, tests and new metrics. The source repositories are not merged mechanically and the shared CSV is not duplicated to inflate sample counts.

The existing `app.py`, report generator, figures and saved model remain legacy assets. Run `flightbridge train`, `flightbridge train-intents`, and `streamlit run demo.py` for the new application. New metrics live in `reports/` and new runtime artifacts in ignored `artifacts/`.

## Contribution provenance

The original projects belong to Dimitris Reppas. This portfolio extension, authored fixtures, documentation and tests were developed with AI assistance. The owner should review the implementation and be prepared to explain the data split, model choice, limitations and operational boundaries in an interview.
