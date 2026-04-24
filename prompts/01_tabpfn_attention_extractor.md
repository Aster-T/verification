# 01 - TabPFN 列注意力提取器

**前置**：已读 `00_conventions.md`。

## 背景

TabPFN v2 的 `PerFeatureTransformer` 每层有"列注意力"（跨特征的 attention，形状 `(B, H, F, F)`）和"行注意力"（跨样本的 attention，形状 `(B, H, N, N)`）。本项目只需要列注意力。

默认实现走 `F.scaled_dot_product_attention`，不返回权重。需要通过 monkey-patch 捕获 softmax 后的权重张量。

## 目标

提供一个 TabPFN 回归封装 + 一个上下文管理器，使得：

```python
from src.models.tabpfn_wrapper import TabPFNWithColAttn

model = TabPFNWithColAttn(device="cuda", seed=0).fit(X_tr, y_tr)
y_pred = model.predict(X_te)
col_attn = model.get_col_attn()   # np.ndarray, shape (F, F), 已跨 batch/head/layer 聚合
```

且调用结束后 tabpfn 的 forward 必须还原，不能污染后续调用。

## 交付物

- `src/models/tabpfn_wrapper.py`
  - `class TabPFNWithColAttn`
    - `__init__(device: str = "cuda", seed: int = 0)`
    - `fit(X: np.ndarray, y: np.ndarray) -> Self`
    - `predict(X: np.ndarray) -> np.ndarray`
    - `get_col_attn(reduce: str = "mean") -> np.ndarray`，shape `(F, F)`。`reduce ∈ {"mean", "last", "per_layer"}`。`per_layer` 时返回 `(L, F, F)`。
  - `@contextmanager capture_column_attention(tabpfn_inner_model)`
- `tests/test_tabpfn_attn.py`
  - 加载 `sklearn.datasets.load_diabetes`，取前 200 样本 / 8 特征。
  - 断言 `get_col_attn().shape == (8, 8)`。
  - 断言每行和 ≈ 1（`np.allclose(attn.sum(-1), 1.0, atol=1e-4)`）。
  - 断言上下文管理器退出后，再调用一次 `predict` 不抛异常、`_captured` 已清空。
  - 断言 `n_estimators=1` 下预测是确定性的（同一 seed 两次 predict 结果一致）。

## 硬约束

- **不修改 `third-party/tabpfn/` 任何文件**。所有 hook 必须在 `src/models/tabpfn_wrapper.py` 内完成。
- 强制 `n_estimators=1`，并在 wrapper 内关闭任何形式的特征置换 / 子采样，使得 `col_attn` 的行列下标与输入 X 的列顺序严格一致。若 TabPFN 构造器不直接暴露相关开关，必须通过修改 `model.model_processors_` 之类的内部字段实现，并在代码里注释"为何这样做 + 对应源码位置"。
- 上下文管理器必须 `try/finally` 还原 forward，哪怕推理中途抛异常。
- 所有捕获的 attention tensor `.detach().cpu()`，禁止把 CUDA tensor 跨函数传出去。

## 步骤

### 步骤 1：源码侦察（不写代码，只做报告）

读以下文件，在对话里回贴：

1. `third-party/tabpfn/src/tabpfn/model/transformer.py`（或对应的 Per-Feature transformer 定义文件）
   - 找到列注意力层的**类名**、在 `PerFeatureTransformer` 里的**属性路径**（例如 `layers[i].self_attn_between_features`）、`forward` 的**签名**。
   - 贴出关键代码片段（不超过 30 行），标注行号。
2. `third-party/tabpfn/src/tabpfn/regressor.py`（或等价入口）
   - 找到 `TabPFNRegressor.fit` / `.predict` 调用内部模型的位置。
   - 确认 `n_estimators`、特征 shuffle、子采样是由哪个字段 / 方法控制的。

**交付物**：一段不含代码改动的文字报告，列清楚：
- 列注意力模块类名 = ?
- 其 forward 签名 = ?
- 通过什么路径能拿到所有层的列注意力模块列表？
- 关闭特征 shuffle 的正确姿势 = ?

**停下来等我 review**。我确认路径无误后再开步骤 2。

### 步骤 2：上下文管理器

仅实现 `capture_column_attention`。包含：

- 全局或闭包变量 `_captured: list[tuple[int, torch.Tensor]]`，元素是 `(layer_idx, attn_weights)`。
- 对每个列注意力层，替换其 `forward` 为手动实现的版本：手动算 `softmax(Q @ K^T / sqrt(d))`，把权重 append 到 `_captured`，再与 V 相乘返回。注意要保留原 forward 的其他参数与返回行为（例如 residual、dropout 位置）。
- 退出时逐个还原。

写最小测试：构造随机输入直接跑一次 forward，断言 `_captured` 非空且形状合理。先不接 TabPFNRegressor。

**停下来等我 review**。

### 步骤 3：TabPFNWithColAttn 外壳

实现 `fit / predict`，并在 `__init__` 里写死 `n_estimators=1` + 关闭特征置换。
`fit` 之后保留 inner model 引用。`predict` 内用 `capture_column_attention` 包裹，把捕获结果存到 `self._last_attn`。

`get_col_attn(reduce)` 从 `self._last_attn` 做聚合：
- 形状是 `list[(layer_idx, tensor of shape (B, H, F, F))]`。
- `reduce="mean"`：先跨 B、H 取 mean → `(L, F, F)`，再跨 L 取 mean → `(F, F)`。
- `reduce="last"`：只取最后一层，跨 B、H mean → `(F, F)`。
- `reduce="per_layer"`：跨 B、H mean → `(L, F, F)`。

**停下来等我 review**。

### 步骤 4：完整单元测试

实现 `tests/test_tabpfn_attn.py`，在 CPU 跑完（若 CUDA 不可用自动降级到 CPU）。把 pytest 输出完整贴回对话。

## 非目标

- 不处理行注意力（留给后续任务）。
- 不画热力图（归 `05`）。
- 不做多 seed / 多 ensemble 聚合（明确 `n_estimators=1`）。
- 不考虑分类任务，只支持回归。

## 完成标准

- `pytest tests/test_tabpfn_attn.py -v` 全绿。
- 能在 REPL 里执行开头 "目标" 段给的示例代码并得到正确 shape 的 `col_attn`。
- `git diff third-party/` 为空。
