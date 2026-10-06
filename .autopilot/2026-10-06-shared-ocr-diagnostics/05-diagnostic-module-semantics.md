# 05 — Persist recognized modules in diagnostics

## Requirement

R04: persist `modules_used` and `field_providers` in diagnostic metadata.

## Finding

Blind acceptance found that `modules_used` in the database was populated from `HybridResult.modules_attempted`, while the pipeline already exposes `HybridResult.modules_used` for modules that returned recognition data. The database key should preserve the existing result object's meaning.

## Acceptance

- `raw_payload.modules_used` is copied from `HybridResult.modules_used`, in pipeline priority order.
- `field_providers` stays unchanged.
- No raw OCR/provider payload or passport values are added to diagnostic metadata.
- No database migration or schema change is introduced.

## Scope

- `services/passport_processing.py`

## Verification

- Do not run tests. Statically compile the changed Python module and inspect the resulting diff.
