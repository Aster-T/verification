# 最终报告

RUNBOOK.md 的 Phase 0 → Phase 7 全部执行完毕,共 17 个单测全绿,`third-party/`
未被修改,`results/report.html` 可直接打开。

## Phase 状态总览

| Phase | Name                          | Status | Tests    | Notes |
|-------|-------------------------------|--------|----------|-------|
| 0     | Read 00_conventions.md        | PASS   | -        | 三条硬约束记录在 PROGRESS.md |
| 1     | MLR wrapper                   | PASS   | 4/4      | 样本复制不变性 atol=1e-8 通过 |
| 2     | TabPFN 列注意力提取             | PASS   | 5/5      | 通过 SDPA monkey-patch 捕获 |
| 3     | 列探索                        | PASS   | 4/4      | diabetes + synth_linear 产物齐备 |
| 4     | 行探索                        | PASS   | 4/4      | MLR/exact 不变性 R² max-min=0 |
| 5     | 可视化 & 报告                 | PASS   | -        | 12 PNG + report.html (383 KB) |
| 6     | 端到端 sanity run             | PASS   | 17/17 全量 | 三条命令 exit 0 |
| 7     | 本文件                        | PASS   | -        | - |

## 生成的关键文件

源码(按 prompt 要求的路径):
- `src/models/mlr_wrapper.py` — `MLRWithW`,支持 is_nominal 跳过标准化
- `src/models/tabpfn_wrapper.py` — `TabPFNWithColAttn` + `capture_column_attention`
- `src/data/loaders.py` — `load_dataset(name, seed)`,支持三个数据集
- `src/probing/column_probe.py` — `run_column_probe`
- `src/probing/row_probe.py` — `duplicate_context`, `run_row_probe`
- `src/viz/heatmap.py`, `src/viz/curves.py`, `src/viz/report.py`
- `scripts/run_column_probe.py`, `scripts/run_row_probe.py`, `scripts/build_report.py`

数据产物(Phase 6 端到端重跑结果):
```
results/column/{diabetes,synth_linear}/
  mlr.npz        w_vec, w_outer, feature_names
  tabpfn.npz     col_attn, col_attn_per_layer (L=24), feature_names
  meta.json      seed, n_tr, n_te, n_features, sklearn/tabpfn 版本,tabpfn_status
results/row/
  diabetes.jsonl         60 行
  synth_linear.jsonl     60 行
results/viz/             12 PNG
results/report.html      383 KB(内嵌 base64 图片,自包含)
```

运行日志与元数据:
- `PROGRESS.md` — 每 phase 的 start/finish/status/tests/files/notes
- `DECISIONS.md` — 自主决策(主要是 Phase 2 的 SDPA patch 选择、
  (G,G)→(F,F) 扩展、Phase 4 的 `--no-tabpfn` 开关)
- `BLOCKERS.md` — 空(无 3 次重试失败的场景)

## 决策摘要 (详见 DECISIONS.md)

Phase 2 — TabPFN 相关:
- 选择 **临时 monkey-patch `torch.nn.functional.scaled_dot_product_attention`**,
  且只在 feature-attn 模块 forward 生效。比"手写完整注意力"路径少写数百行重复
  代码,同时数值与原模型完全一致。
- 发现 tabpfn 7.1.1 实际加载 `TabPFNV2p6` 架构而不是 prompt 设想的
  `PerFeatureTransformer`,attention 路径变为
  `model.blocks[i].per_sample_attention_between_features`。wrapper 同时兼容新旧
  路径。
- v2.6 的 `features_per_group=3` 意味着 transformer 对列的注意力是"每组一个
  token",所以原生形状是 `(G, G)`。为了守住 `(F, F)` 的 API 合约,在 wrapper
  里按 feature→group 映射把组级注意力扩展回特征级。同组特征的行/列一致,已在
  docstring 中明确说明。
- 通过 `inference_config={FEATURE_SHIFT_METHOD: None, POLYNOMIAL_FEATURES:"no",
  FINGERPRINT_FEATURE: False, PREPROCESS_TRANSFORMS: [PreprocessorConfig(
  "none")], OUTLIER_REMOVAL_STD: None, REGRESSION_Y_PREPROCESS_TRANSFORMS:
  (None,)}` 关掉一切会增删或打乱特征的 step,保证 attn 轴与 X 列一一对应。

Phase 4:
- 在 `run_row_probe` / `scripts/run_row_probe.py` 中加了 `include_tabpfn` /
  `--no-tabpfn` 开关,便于 MLR-only 冒烟测试。默认仍然跑两模型。
- `tests/test_row_probe.py` 本地重写 `tmp_path` fixture,绕开本机
  `%LOCALAPPDATA%/Temp/pytest-of-*` 的 WinError 5 权限问题。

## 阻塞点 (详见 BLOCKERS.md)

None. 没有任何测试在 3 次重试后仍失败,没有触发 BLOCKERS。

## 建议的下一步

- **用户需要人工验证的**:
  - 打开 `results/report.html`,核对:
    - 每个数据集的三张热力图 colorbar 是否共享且对称。
    - 行探索曲线是否 4 条(MLR/exact, MLR/jitter, TabPFN/exact, TabPFN/jitter)。
    - MLR/exact 是否确实水平(diabetes seed=0 下全部 k R²=0.3322332173106186)。
  - `col_attn` 经过 (G,G)→(F,F) 扩展后同组特征行列相同,是否符合下游分析的
    意图;如果希望保持组级分辨率,可在 `get_col_attn` 加一个
    `reduce="group"` 分支。
- **已知遗留问题**:
  - cali_housing 数据集没有跑完整 pipeline(RUNBOOK Phase 3 fallback 允许跳过
    以防显存不足,Phase 6 只跑了 diabetes + synth_linear)。如果需要,可以
    `python scripts/run_column_probe.py --dataset cali_housing`,context
    =1600,应该在 10k 上限内安全运行。
  - `tabpfn.__version__ == 7.1.1`,`sklearn` 版本被写入
    `results/column/*/meta.json`。若要更老的 tabpfn v2 行为(features_per_group
    =1,注意力天然是 `(F, F)`),需要降级到 `pip install tabpfn==2.x`,
    wrapper 的 v2 分支已经兼容。
  - `row_probe` 没有实现 `--resume`;每次跑都 append(配 `--fresh` 清空)。如
    实验规模扩大可以加 `existing_keys_jsonl` 去重(`src/utils/io.py` 已提供
    这一 helper)。

## 可复现命令

```
rm -rf results/
python scripts/run_column_probe.py --dataset diabetes --dataset synth_linear
python scripts/run_row_probe.py    --dataset diabetes --dataset synth_linear --fresh
python scripts/build_report.py     --dataset diabetes --dataset synth_linear
```

在当前环境下三条命令耗时合计约 40 秒(CUDA 可用,TabPFN v7.1.1 editable 安装)。
