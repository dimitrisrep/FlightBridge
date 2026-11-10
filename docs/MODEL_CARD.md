# FlightBridge model card

## Intended use

Local research and portfolio demonstrations of bilingual intake, reproducible classification and human review. Target values describe recorded case outcomes, not independently verified legal entitlements. An agent verifies current carrier policy and availability before any action.

## Models

The entitlement experiment compares a most-frequent baseline, balanced logistic regression, balanced random forest and an MLP with hidden layers `(16, 8)`, ReLU, L-BFGS and L2 regularization. Model selection uses development macro F1; a neural architecture is not assumed to be superior.

Refund selects logistic regression; rebooking selects random forest in the measured run. Both use one-hot categorical features and imputed/scaled duration. Predictors are `airline`, `airline_type`, `sc_type`, `alternative_provided`, `delay_minutes`.

The separate intent model uses word and character TF-IDF plus logistic regression. Rules take precedence where negation, uncertainty or multiple intents are detected. The classifier's fallback acceptance threshold of 0.55 is a fixed demo heuristic and has not been calibrated on passenger traffic.

## Evaluation protocol

- Retain the source CSV unchanged and preserve unknown target values.
- Exclude timing/category contradictions; record excluded case IDs.
- Group identical predictor combinations before the outer split.
- Select fixed candidates using three grouped folds on the development partition.
- Evaluate the selected pipeline once on the outer holdout; keep the serving pipeline development-only.
- Report macro F1, per-class precision/recall, confusion matrix, balanced accuracy, majority baseline and Brier score.

See `reports/entitlement_metrics.json` for case-level split manifests and all candidate CV scores. The report records software versions and the input SHA-256.

## Limitations and failure modes

Only 175 retained observations, 36 source airline identifiers before splitting, imbalanced outcomes, incomplete policy provenance and possible hidden case dependencies. The grouped split does not isolate airlines or time periods. Development-to-test performance drops markedly. A shared input combination can contain inconsistent outcomes.

Airline identity is a categorical feature, so unknown carriers are sent to review. Delays outside the development range and inconsistent airline types also receive review rather than a score. Known-carrier predictions always remain subject to agent review. The API does not verify whether the underlying airline policy is still valid.

Scores from the entitlement classifiers are uncalibrated and must not be advertised as probabilities of entitlement. The intent dataset is authored and synthetic; no production intent accuracy or language-parity claim is supported. Do not use the 0.915 synthetic intent F1 as a general NLP benchmark.

## Deployment boundaries

No access to booking systems, ticket issuance, payments or airline policy APIs. No authentication, distributed conversation store, load test or production monitoring is included. Model artifacts use pickle-based joblib and must come from a trusted local training run with the same scikit-learn version.
