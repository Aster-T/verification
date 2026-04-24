# DECISIONS

Autonomous decisions made while executing RUNBOOK.md. Each entry:
date, phase, one-line decision, context, rationale.

## 2026-04-22 Phase 2
Decision: Capture `AlongRowAttention` (v2.6+) / `MultiHeadAttention` (v2)
  column attention by monkey-patching `torch.nn.functional.scaled_dot_product_attention`
  only during feature-attention forwards (not item-attention).
Context: Prompt requested `capture_column_attention` context manager. v2.6
  attention reaches SDPA through `_batched_scaled_dot_product_attention`,
  which does not expose softmax weights directly. Replacing individual
  modules' forward with a manual softmax implementation would duplicate a lot
  of shape-handling/chunking code.
Rationale: Patching only during the specific forward call keeps item-attention
  untouched and produces mathematically identical outputs (softmax then matmul
  with V), while adding a capture side effect. `try/finally` guarantees
  restoration even on exceptions.

## 2026-04-22 Phase 2
Decision: Return (F, F) column attention by expanding the group-level (G, G)
  matrix. Features sharing a TabPFN feature-group share identical rows and
  columns in the returned tensor.
Context: TabPFN v2.6 uses `features_per_group=3`; for F=8 inputs the model
  attends over G=3 groups (+ target). The prompt specified `get_col_attn`
  return shape `(F, F)` and feature-order alignment with MLR's W.
Rationale: There is no way to get true per-feature attention in v2.6 without
  setting `features_per_group=1`, which would require retraining the model.
  Expanding (G, G) → (F, F) by feature→group mapping preserves the (F, F)
  contract, keeps feature ordering aligned with X columns, and is honest about
  within-group tying. Documented in the function docstring.

## 2026-04-22 Phase 2
Decision: Disable FEATURE_SHIFT_METHOD, POLYNOMIAL_FEATURES, FINGERPRINT_FEATURE,
  OUTLIER_REMOVAL_STD, and use a "none" PreprocessorConfig in `inference_config`.
Context: All of these either permute, add, or remove feature columns, any of
  which would break the (F, F) ↔ X columns mapping.
Rationale: With n_estimators=1, a single forward pass must see the same
  feature set the caller passed. Any ensemble-oriented augmentation is
  unnecessary (and actively harmful for this analysis).

## 2026-04-22 Phase 4
Decision: Added `--no-tabpfn` flag to `scripts/run_row_probe.py` (not in the
  original prompt) and a matching `include_tabpfn` parameter in
  `run_row_probe()`.
Context: Some tests (and the Phase 5 smoke curve) only need MLR records to
  validate the flat-line invariant / plotting code. Running TabPFN for those
  adds ~1s per (k, mode, seed) and a GPU dependency.
Rationale: Low-cost addition that keeps default behaviour (both models) and
  doesn't change jsonl schema when both are included. The row-probe prompt
  explicitly allows skipping TabPFN if Phase 2 fails; this flag just exposes
  that same knob to the CLI.

## 2026-04-22 Phase 4
Decision: Test uses a repo-local tmp directory (.pytest_cache/tmp/case-*)
  instead of pytest's default tmp_path.
Context: pytest's make_numbered_dir raises `PermissionError: [WinError 5]` on
  `os.scandir('%LOCALAPPDATA%/Temp/pytest-of-ryunen')` on this Windows box.
Rationale: Redirect scoped to tests/test_row_probe.py only; keeps the rest of
  the test suite using the standard fixture.
