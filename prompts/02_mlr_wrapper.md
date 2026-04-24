# 02 - MLR 封装 + W 矩阵导出

**前置**：已读 `00_conventions.md`。

## 目标

封装 `sklearn.linear_model.LinearRegression`，在标准化后的特征空间拟合，使其权重 `coef_` 具有跨特征可比性；同时提供一个与 TabPFN 列注意力同形状的 rank-1 代理矩阵 `W ⊗ W`，方便并排可视化。

## 交付物

- `src/models/mlr_wrapper.py`
  - `class MLRWithW`
    - `__init__(standardize: bool = True)`
    - `fit(X: np.ndarray, y: np.ndarray, is_nominal: list[bool] | None = None) -> Self`
    - `predict(X: np.ndarray) -> np.ndarray`
    - `get_W() -> dict`，返回 `{"w_vec": np.ndarray shape (F,), "w_outer": np.ndarray shape (F, F), "feature_names": list[str] | None}`
- `tests/test_mlr_wrapper.py`
  - 见下方"测试清单"。

## 关键约束

1. **标准化**：默认对连续列做 `(x - mu) / sd`；`is_nominal=True` 的列**跳过标准化**（这些列通常是 `pd.factorize` 出的整数编码，z-score 会给 MLR 引入伪线性序关系）。
2. 标准化参数 `mu_`、`sd_` 必须在 `fit` 时锁定，`predict` 用同一套参数，不能在 predict 时重新估计。
3. `w_outer = np.outer(w_vec, w_vec)`。显式注释这是 rank-1 代理，不等价于特征交互。

## 测试清单

- **基本功能**：`load_diabetes` 上拟合，`get_W()["w_vec"].shape == (10,)`，`w_outer.shape == (10, 10)`。
- **标准化正确性**：构造 `X` 让第 0 列方差远大于其他列，比较 `standardize=True` 和 `False` 下的 `w_vec`，断言前者各维尺度更接近（用 `abs(w).max() / abs(w).min()` 比值作为量度，前者应显著更小）。
- **样本复制不变性**（sanity check，极其重要）：
  ```
  m1 = MLRWithW().fit(X, y)
  X2, y2 = np.tile(X, (3,1)), np.tile(y, 3)
  m2 = MLRWithW().fit(X2, y2)
  assert np.allclose(m1.get_W()["w_vec"], m2.get_W()["w_vec"], atol=1e-8)
  ```
  这条测试也会被 `04` 的行探索复用——它保证 "OLS 对均匀复制样本的不变性" 成立。若失败，行探索的 MLR baseline 就不是水平线，整个实验逻辑坍塌。
- **nominal 跳过**：构造 `X`，第 0 列取值 `{0,1,2}` 且标记为 nominal；断言 `mu_[0] == 0` 且 `sd_[0] == 1`。

## 步骤

1. 实现 `MLRWithW` 基本结构（标准化 + fit + predict），写前两项测试并通过。**停**。
2. 加 `get_W()` 返回 dict，写第三项（样本复制不变性）测试并通过。**停**。
3. 加 nominal 跳过逻辑，写第四项测试并通过。把 `pytest -v` 输出贴回。

## 非目标

- 不做 Ridge / Lasso / ElasticNet 变体。
- 不做 p-value、置信区间、t 检验。
- 不画图。
