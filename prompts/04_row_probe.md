# 04 - 行探索主流程

**前置**：已读 `00_conventions.md`，已完成 `02`（MLR wrapper）。`01` 若还没做完也可以先跑 MLR 分支。

## 目标

对每个数据集，把训练集（TabPFN 语境下的 context）整体复制 k 倍，k ∈ `{1, 2, 3, 5, 10}`，固定测试集不变，记录 MLR 与 TabPFN 的 R²、RMSE、MAE 随 k 的变化。

同时支持两种复制模式：
- `exact`：`np.tile(X_tr, (k, 1))`，完全重复。
- `jitter`：复制后加极小高斯噪声 `N(0, 1e-6)`，打破完全共线。

## 交付物

- `src/probing/row_probe.py`
  - `duplicate_context(X, y, k: int, mode: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]`
  - `run_row_probe(dataset: str, out_path: Path, k_list: list[int], modes: list[str], seeds: list[int]) -> None`
    对每个 `(model, k, mode, seed)` 组合产出一行 JSON，写入 `out_path / f"{dataset}.jsonl"`（append 模式，跑多次要先清空或用 `--resume`）。
- `scripts/run_row_probe.py`，CLI 参数：
  - `--dataset`（可多次）
  - `--k-list` 逗号分隔，默认 `1,2,3,5,10`
  - `--modes` 逗号分隔，默认 `exact,jitter`
  - `--seeds` 逗号分隔，默认 `0,1,2`
  - `--out`，默认 `results/row`
  - `--fresh`，若指定则先删除目标 jsonl
- `tests/test_row_probe.py`
  - MLR 在 `exact` 模式下，对同一 (dataset, seed) 不同 k 的 R² 必须完全相同（`atol=1e-8`）。这是框架层 sanity check，**不通过就不要继续**。
  - `duplicate_context(k=1, mode="exact")` 返回的数组与原数组相等。
  - `jitter` 模式下两次调用（不同 rng state）结果不一致。

## JSONL schema

每行一个 JSON，字段顺序固定：

```json
{"dataset": "diabetes", "model": "mlr", "k": 3, "mode": "exact",
 "seed": 0, "n_ctx": 1323, "n_te": 89,
 "r2": 0.4801, "rmse": 54.23, "mae": 43.11,
 "fit_sec": 0.02, "predict_sec": 0.01}
```

`n_ctx` 是复制后的 context 大小（= `k * n_tr_original`）。

## `scripts/run_row_probe.py` 骨架

```python
# scripts/run_row_probe.py
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from probing.row_probe import run_row_probe  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]

def _csv_strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--k-list", type=_csv_ints, default=[1, 2, 3, 5, 10])
    p.add_argument("--modes", type=_csv_strs, default=["exact", "jitter"])
    p.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=REPO / "results" / "row")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for ds in args.dataset:
        jsonl_path = args.out / f"{ds}.jsonl"
        if args.fresh and jsonl_path.exists():
            jsonl_path.unlink()
        run_row_probe(ds, jsonl_path, args.k_list, args.modes, args.seeds)


if __name__ == "__main__":
    main()
```

## 关键约束

1. **TabPFN context 上限**：v2 大约 10k 样本。复制后超过上限直接 skip 该组合，并往 jsonl 写一条 `{"skipped": true, "reason": "n_ctx > 10000", ...}`。不要硬撑导致 OOM。
2. **NaN 保留原则**：不要在 row_probe 里预填 NaN；MLR 侧若数据含 NaN，在 `MLRWithW` 内用 `SimpleImputer(strategy="mean")` 单独处理，而不是污染数据源。
3. **评估固定**：所有 `(k, mode, seed)` 下 `X_te, y_te` 必须完全一致——只在 context 上做复制。实现时先 split，再在 train 上循环。
4. **时间记录**：`fit_sec` / `predict_sec` 用 `time.perf_counter()` 量，用于后续分析 context 翻倍的计算代价。

## 步骤

1. 实现 `duplicate_context` + 单测三条。**停**。
2. 实现 `run_row_probe` 的 MLR 分支，跑 `diabetes` 一次，把生成的 jsonl 头 20 行贴回。**重点验证 sanity check**：过滤 `model=mlr, mode=exact, seed=0` 的记录，`r2` 列应对所有 k 完全相同。**停**。
3. 加 TabPFN 分支，跑 `diabetes` 一次，把分模型的 R² 随 k 变化贴回（简单 print 即可）。**停**。
4. 写 `scripts/run_row_probe.py`，在命令行跑 `--dataset diabetes --fresh`，验证 `--fresh` 生效。

## 非目标

- 不做 CD (Critical Difference) 检验（后续）。
- 不画曲线（归 `05`）。
- 不在本 prompt 内尝试 "subsample to size N_k" 对照组——那是下一轮扩展。
