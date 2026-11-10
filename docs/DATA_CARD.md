# Data card

## Recorded cases

Source: `dimitrisrep/Schedule_change_AI` at commit `2a0ef48981cc2b8cb03ce67860c07fbaef7ec2cc`. The identical raw CSV also exists in `dimitrisrep/hackathon`; it is not counted as additional training data.

The supplied CSV has 181 records and 14 columns. Its policy provenance, collection dates, permissions for public redistribution and label verification are not documented. Existing private repository access is retained; this change does not authorize public publication of operational data.

The `prepare_data` logic is carried forward from the source project. Unknown outcomes remain unknown. A rejected rebooking request does not imply that a refund is disallowed. Label contradictions quarantine only the affected target.

The new experiment excludes case IDs `7, 13, 35, 72, 141, 145` because their durations are missing or inconsistent with the stated schedule categories. It does not rewrite the original entries or infer their correct labels.

The five allowed predictors omit `pap_request`, `solution`, `rejection_reason` and all outcome columns. Outcome data is used for label preparation only. Exact predictor combinations are grouped to prevent copied inputs crossing the split. This is not a guarantee that all related cases have been identified.

## Intent fixtures

`data/intents/train.jsonl` and `test.jsonl` contain 48 and 24 separate English/Greek examples, respectively. They were authored for this portfolio implementation with AI assistance and are explicitly marked `authored_synthetic_fixture`. They contain no passenger records. Their generator is `scripts/build_intent_fixtures.py`.

Classes: refund, rebooking, options, compensation, baggage and other. Exact normalized text overlap between the files is rejected during training. Larger independently annotated data and a held-out conversational evaluation are required before any operational NLP claim.
