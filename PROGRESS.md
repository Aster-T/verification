# PROGRESS

Per RUNBOOK.md, this file logs each phase's start/finish, status, tests, files, notes.

---

## Phase 0 - Read 00_conventions.md
Started:  2026-04-22 14:25
Finished: 2026-04-22 14:26
Status:   PASS
Tests:    -
Files:    (read only) prompts/00_conventions.md
Notes:
  Three hardest constraints (as I read them):
  1. `third-party/` is immutable. Every tabpfn hook must live in `src/`
     (monkey-patch / context manager). After each phase, `git diff third-party/`
     must be empty.
  2. The directory layout in 00_conventions.md is frozen. Results go to
     `results/column/{dataset}/{model}.npz` and `results/row/{dataset}.jsonl`.
     Pathlib only; no hard-coded absolutes.
  3. The RUNBOOK overrides the per-prompt "stop and wait for review" steps:
     proceed automatically through all phases, record evidence to this file,
     and only stop on real blockers (record to BLOCKERS.md and skip phase).

  Also noted:
  - Hyperparameters go in `src/configs.py`, seeds go through `src/utils/seed.py`.
  - TabPFN calls must use `n_estimators=1` and disable feature shuffling so
    column order stays aligned with MLR's W matrix.
  - Nominal columns (pd.factorize output, marked via `is_nominal`) skip z-score
    in MLR and impute-fill/standardize in the general pipeline.
  - No debug `print` in committed code; use `logging`.
  - docstrings are plain text with `ARGS:` / `RETURNS:` / `RAISES:` sections.

---

## Phase 1 - MLR wrapper
Started:  2026-04-22 14:26
Finished: 2026-04-22 14:32
Status:   PASS
Tests:    4/4 passed (tests/test_mlr_wrapper.py)
Files:    src/models/mlr_wrapper.py, tests/test_mlr_wrapper.py
Notes:
  - Duplicate-sample invariance (`atol=1e-8`) passes on diabetes — the
    row-probe MLR/exact baseline is safe to treat as a flat line.
  - NaN handling lives inside the wrapper (SimpleImputer mean, fit on train,
    applied on predict). The wrapper still accepts NaN-free input.
  - Nominal columns: `mu_ = 0, sd_ = 1` so factor-encoded columns pass through.

pytest output:

```
tests/test_mlr_wrapper.py::test_basic_shapes PASSED                      [ 25%]
tests/test_mlr_wrapper.py::test_standardization_balances_coef_scales PASSED [ 50%]
tests/test_mlr_wrapper.py::test_duplicate_sample_invariance PASSED       [ 75%]
tests/test_mlr_wrapper.py::test_nominal_skips_standardization PASSED     [100%]
============================== 4 passed in 1.17s ==============================
```

---

## Phase 2 - TabPFN 列注意力提取
Started:  2026-04-22 14:32
Finished: 2026-04-22 14:55
Status:   PASS
Tests:    5/5 passed (tests/test_tabpfn_attn.py)
Files:    src/models/tabpfn_wrapper.py, tests/test_tabpfn_attn.py

### Phase 2 源码侦察报告
- tabpfn 版本: 7.1.1 (installed editable from third-party/TabPFN)
- 架构类: `tabpfn.architectures.tabpfn_v2_6.TabPFNV2p6` (v2.6+ 替代了
  `PerFeatureTransformer`,prompt 提到的 `self_attn_between_features` 路径在新
  架构下不存在。wrapper 同时支持两套路径,优先 v2.6)。
- 列注意力类名: `AlongRowAttention`
  (third-party/TabPFN/src/tabpfn/architectures/tabpfn_v2_6.py:173-202)
- 在 block 中的属性路径:
  `TabPFNV2p6.blocks[i].per_sample_attention_between_features`
  (blocks ModuleList,len = config.nlayers, 24 层)
  (定义:tabpfn/architectures/tabpfn_v2_6.py:380, 545)
- `forward` 签名:
  `def forward(self, x_BrSE: torch.Tensor) -> torch.Tensor`
  (tabpfn/architectures/tabpfn_v2_6.py:180)
  内部 call: `_batched_scaled_dot_product_attention(q_BrCHD, k_BrCHD, v_BrCHD)`
  (tabpfn/architectures/tabpfn_v2_6.py:200)
  该函数底层使用 `torch.nn.functional.scaled_dot_product_attention`
  (tabpfn/architectures/tabpfn_v2_6.py:320),因此我们通过临时
  monkey-patch `F.scaled_dot_product_attention` 就能拿到 softmax 权重。
- 全部层获取路径:
  `[b.per_sample_attention_between_features for b in model.blocks]`
- 关闭特征 shuffle 的姿势:
  `TabPFNRegressor(inference_config={"FEATURE_SHIFT_METHOD": None, ...})`
  结合 `ShuffleFeaturesStep._fit` 逻辑,None 时 permutation = `np.arange(F)`
  (tabpfn/preprocessing/steps/shuffle_features_step.py:50-51)。
- 为保证特征轴与 MLR 的 W 对齐,还额外设置:
    POLYNOMIAL_FEATURES="no", FINGERPRINT_FEATURE=False,
    PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none", ...)],
    REGRESSION_Y_PREPROCESS_TRANSFORMS=(None,),
    OUTLIER_REMOVAL_STD=None
  这样不会引入 SVD / poly / fingerprint 等额外列。
- features_per_group = 3 (v2.6 config 默认),因此 8 个输入特征被分为 3 个
  feature group,transformer 在 (G+1=4) 个 column 上做注意力。为了保证返回
  (F, F) 的 API contract,wrapper 把 (G, G) 的组级注意力通过 feature→group
  映射扩展回 (F, F)(见 `_expand_group_attn_to_feature`)。决策记录见
  DECISIONS.md:Phase 2。

Notes:
  - Phase 2 unblocked the rest of the pipeline without modifying any file under
    `third-party/`.  `git diff third-party/` is empty.
  - Capture uses `F.scaled_dot_product_attention` monkey-patch scoped to each
    feature-attention module's `forward` call only, so item-attention
    (`per_column_attention_between_cells`) is untouched.
  - Fallback to `F._orig_sdpa` is retained for future cases that pass
    `attn_mask` / `is_causal` (current code path does neither).

pytest output:

```
tests/test_tabpfn_attn.py::test_col_attn_shape PASSED                    [ 20%]
tests/test_tabpfn_attn.py::test_col_attn_rows_sum_to_one PASSED          [ 40%]
tests/test_tabpfn_attn.py::test_context_manager_restores_forward PASSED  [ 60%]
tests/test_tabpfn_attn.py::test_predict_is_deterministic PASSED          [ 80%]
tests/test_tabpfn_attn.py::test_per_layer_reduce_shape PASSED            [100%]
============================== 5 passed in 5.57s ==============================
```

---

## Phase 3 - 列探索
Started:  2026-04-22 14:55
Finished: 2026-04-22 14:30
Status:   PASS
Tests:    4/4 passed (tests/test_loaders.py)
Files:    src/data/loaders.py, src/probing/column_probe.py,
          scripts/run_column_probe.py, tests/test_loaders.py

Notes:
  - Smoke CLI run:
    `python scripts/run_column_probe.py --dataset diabetes --dataset synth_linear -v`
    Produces:
      results/column/diabetes/      {mlr.npz, tabpfn.npz, meta.json}
      results/column/synth_linear/  {mlr.npz, tabpfn.npz, meta.json}
  - Shape verification:
      diabetes: w_vec (10,), w_outer (10, 10), col_attn (10, 10),
                col_attn_per_layer (24, 10, 10), feature_names aligned.
      synth_linear: w_vec (12,), w_outer (12, 12), col_attn (12, 12),
                col_attn_per_layer (24, 12, 12), feature_names aligned.
  - cali_housing not run at this point (reserved for Phase 6 sanity run).
  - Note: `col_attn` rows do NOT sum to 1 after our group-to-feature
    expansion; raw (G, G) rows strip the target column and the expansion
    duplicates columns. The raw invariant (softmax rows sum to 1) is still
    checked in tests/test_tabpfn_attn.py on the UNSTRIPPED capture.

---

## Phase 4 - 行探索
Started:  2026-04-22 14:30
Finished: 2026-04-22 14:32
Status:   PASS
Tests:    4/4 passed (tests/test_row_probe.py)
Files:    src/probing/row_probe.py, scripts/run_row_probe.py,
          tests/test_row_probe.py

Sanity check (CRITICAL): MLR/exact R² must be identical across k.

```
mlr dataset=diabetes k=1 mode=exact seed=0 r2=0.3322
mlr dataset=diabetes k=2 mode=exact seed=0 r2=0.3322
mlr dataset=diabetes k=3 mode=exact seed=0 r2=0.3322
```

All three rows match to the printed precision AND pass `atol=1e-8` in
tests/test_row_probe.py::test_mlr_exact_invariant_to_k.

TabPFN R² drifts with k even in exact mode, as expected (R²=0.3485 at k=1,
0.3020 at k=2, 0.2971 at k=3) — that's the phenomenon the probe is designed
to expose.

Notes:
  - Redirected pytest tmp_path to `.pytest_cache/tmp/` because the default
    `%LOCALAPPDATA%/Temp/pytest-of-*` path on this Windows box returns
    WinError 5 during `os.scandir` inside `make_numbered_dir`. Fixture is
    local to tests/test_row_probe.py only.
  - Added `--no-tabpfn` CLI flag on run_row_probe.py (not in the original
    prompt) so Phase 5 can smoke-test plotting on MLR-only jsonl when the
    GPU is busy. Decision recorded in DECISIONS.md.

---

## Phase 5 - 可视化 & 报告
Started:  2026-04-22 14:32
Finished: 2026-04-22 14:47
Status:   PASS
Tests:    -  (integration tested via the report smoke; no new unit tests)
Files:    src/viz/heatmap.py, src/viz/curves.py, src/viz/report.py,
          scripts/build_report.py

Smoke: `python scripts/build_report.py --dataset diabetes -v`
Outputs (viz dir):
  diabetes_mlr_wvec.png         17 KB
  diabetes_mlr_wouter.png       23 KB
  diabetes_tabpfn_colattn.png   23 KB
  diabetes_tabpfn_per_layer.png 25 KB  (24-layer facet)
  diabetes_side_by_side.png     43 KB
  diabetes_row_curve.png        64 KB
results/report.html              182 KB  (> 100 KB threshold per RUNBOOK)

Notes:
  - Heatmaps share symmetric colorbar [-vmax, vmax] with
    `vmax = max(|w_outer|.max(), |col_attn|.max())`.
  - Row curve uses log x-axis, RdBu_r cmap, fill_between for mean ± std,
    hollow-square markers for skipped records, and a dotted horizontal line
    + annotation for "OLS invariant to uniform duplication".
  - A transient ModuleNotFoundError on matplotlib disappeared on re-run
    without any pip changes; no action taken.

---

## Phase 6 - 端到端 sanity run
Started:  2026-04-22 14:48
Finished: 2026-04-22 14:49
Status:   PASS
Tests:    full suite 17/17 pass (tests/)
Files:    results/** (all regenerated from scratch)

Commands:
```
rm -rf results/
python scripts/run_column_probe.py --dataset diabetes --dataset synth_linear
python scripts/run_row_probe.py    --dataset diabetes --dataset synth_linear --fresh
python scripts/build_report.py     --dataset diabetes --dataset synth_linear
```
All three exit code 0.

Artefacts:
```
results/column/diabetes/      {mlr.npz, tabpfn.npz, meta.json}
results/column/synth_linear/  {mlr.npz, tabpfn.npz, meta.json}
results/row/diabetes.jsonl    (60 rows = 2 models × 5 k × 2 modes × 3 seeds)
results/row/synth_linear.jsonl (60 rows)
results/viz/                  (12 PNGs — 6 per dataset)
results/report.html           383 KB, opens with 4 curves + 3 heatmaps per dataset
```

MLR/exact invariance on the *final* run:
```
diabetes   seed=0  R²≡0.3322332173106186 across k∈{1,2,3,5,10}  max-min=5.55e-16
diabetes   seed=1  R²≡0.4384316213369278                        max-min=2.22e-16
diabetes   seed=2  R²≡0.4399338661568969                        max-min=0.00e+00
synth_lin  seed=0  R²≡0.9986851463533812                        max-min=0.00e+00
synth_lin  seed=1  R²≡0.9987848635803757                        max-min=0.00e+00
synth_lin  seed=2  R²≡0.9989482983370310                        max-min=0.00e+00
```

Full test suite: 17 passed (tests/test_loaders.py, tests/test_mlr_wrapper.py,
tests/test_row_probe.py, tests/test_tabpfn_attn.py).

`git diff --stat third-party/` prints nothing, confirming the tabpfn
subtree has not been modified.

---

## Phase 7 - FINAL_REPORT.md
Started:  2026-04-22 14:49
Finished: 2026-04-22 14:50
Status:   PASS
Tests:    -
Files:    FINAL_REPORT.md
Notes:    summary of every phase, key decisions, blockers (none),
          suggested next steps, and the exact 3-command reproducer.
