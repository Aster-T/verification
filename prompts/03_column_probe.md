# 03 - 列探索主流程

**前置**：已读 `00_conventions.md`，已完成 `01`、`02`。

## 目标

对每个数据集分别拟合 MLR 和 TabPFN，导出：
- MLR 的 `w_vec`、`w_outer`
- TabPFN 的 `col_attn (F, F)`（`reduce="mean"` 跨层）和 `col_attn_per_layer (L, F, F)`

只存数值，**不画图**（归 `05`）。

## 交付物

- `src/data/loaders.py`
  - `load_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[bool]]`
    返回 `(X_tr, y_tr, X_te, y_te, feature_names, is_nominal)`。
  - 支持 `name ∈ {"diabetes", "cali_housing", "synth_linear"}`，所有数据集来源写进 `configs.CONFIG["datasets"]`。
- `src/probing/column_probe.py`
  - `run_column_probe(dataset: str, out_dir: Path, seed: int) -> None`
    输出到 `out_dir / dataset /` 下的文件：
    - `mlr.npz`，字段 `w_vec`, `w_outer`, `feature_names`
    - `tabpfn.npz`，字段 `col_attn`, `col_attn_per_layer`, `feature_names`
    - `meta.json`，记录 `seed`, `n_tr`, `n_te`, `n_features`, 模型版本
- `scripts/run_column_probe.py`
  - 纯 Python 入口，CLI 参数 `--dataset`（可多次传） `--out` `--seed`。
  - 开头通过 `sys.path` 把 `<repo>/src` 和 `<repo>/third-party/tabpfn/src` 加入 import 路径（Windows/Linux 都能跑）。

## `scripts/run_column_probe.py` 骨架

```python
# scripts/run_column_probe.py
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from probing.column_probe import run_column_probe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", required=True,
                   help="可多次指定，例如 --dataset diabetes --dataset cali_housing")
    p.add_argument("--out", type=Path, default=REPO / "results" / "column")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for ds in args.dataset:
        run_column_probe(ds, args.out, args.seed)


if __name__ == "__main__":
    main()
```

按这个结构写。不要在 script 里做业务逻辑，全部交给 `probing/column_probe.py`。

## 约束

- `load_dataset` 必须返回 numpy，不是 pandas。`feature_names` 单独返回。
- TabPFN 拟合前 `y_tr` 要 `float64`。
- `meta.json` 记录 `tabpfn.__version__` 和 `sklearn.__version__`，便于复现。
- **特征顺序一致性**：`mlr.npz` 和 `tabpfn.npz` 的 `feature_names` 必须完全相同（同序同长度），否则 `05` 的热力图没法对齐。在 `run_column_probe` 末尾加一条 assert。

## 步骤

1. 实现 `load_dataset`，三个数据集各写一个单测（shape 正确、`is_nominal` 长度匹配、训练/测试无交集）。**停**。
2. 实现 `run_column_probe` 的 MLR 分支，跑 `diabetes` 看产物。**停**。
3. 加上 TabPFN 分支，跑 `diabetes` 看产物。验证 feature_names 对齐。**停**。
4. 写 `scripts/run_column_probe.py`，在命令行跑一次 `--dataset diabetes --dataset synth_linear`，把生成的文件树贴回。

## 非目标

- 不画热力图。
- 不做统计检验。
- 不 ensemble 多 seed（单 seed 即可；多 seed 聚合等有结果后再考虑）。
