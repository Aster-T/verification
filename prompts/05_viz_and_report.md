# 05 - 可视化与 HTML 报告

**前置**：已读 `00_conventions.md`，已完成 `03`（列探索产物）和 `04`（行探索 jsonl）。

## 目标

把 `03`、`04` 的数值产物渲染成图与一份单文件 HTML 报告，渲染过程**完全不接触模型**——只读 `.npz` / `.jsonl`。这样调图不用重跑实验。

## 交付物

- `src/viz/heatmap.py`
  - `plot_column_heatmaps(dataset: str, column_dir: Path, out_dir: Path) -> Path`
    产出：
    - `{dataset}_mlr_wvec.png`（1D bar + 横向热力条）
    - `{dataset}_mlr_wouter.png`（`(F, F)` 热力图）
    - `{dataset}_tabpfn_colattn.png`（`(F, F)` 热力图）
    - `{dataset}_side_by_side.png`（三张并排，**同一 colorbar 范围**）
- `src/viz/curves.py`
  - `plot_row_curves(dataset: str, jsonl_path: Path, out_dir: Path) -> Path`
    产出 `{dataset}_row_curve.png`：x 轴 k，y 轴 R²，按 (model, mode) 分 4 条线，每条线跨 seed 画 mean ± std（`fill_between`）。
- `src/viz/report.py`
  - `build_report(datasets: list[str], results_root: Path, out_html: Path) -> Path`
    生成单文件 HTML，内嵌所有 PNG（base64）或相对路径（二选一，默认 base64 便于分享）。
- `scripts/build_report.py`，CLI 参数 `--dataset`（可多次）、`--results-root`（默认 `results`）、`--out`（默认 `results/report.html`）。

## 热力图规范

1. **同一数据集内的三张热力图共享 colorbar 范围**：`vmax = max(|w_outer|.max(), |col_attn|.max())`，`vmin = -vmax`。
2. cmap 用 `RdBu_r`（对称双色），对角线不做特殊处理（MLR 的 `w_outer` 对角线就是 `w_i²`，TabPFN 列注意力对角线一般较大，这是正常的）。
3. 横纵坐标 tick 用 `feature_names`，长名字旋转 45°。
4. 标题写清楚聚合方式，例如 `TabPFN col-attn (mean over B/H/L, n_estimators=1)`。

## 曲线图规范

1. 四条线：`MLR/exact`, `MLR/jitter`, `TabPFN/exact`, `TabPFN/jitter`。
2. `MLR/exact` 在理论上是水平线，**在图上显式标注**（加一条虚线横贯 + annotation "OLS 对均匀复制不变"）。若实测不水平，图里要能一眼看出来。
3. x 轴 log 刻度（k 跨度 1~10 不算大，但 log 轴更稳当），y 轴线性。
4. skipped 记录（`04` 里 TabPFN 超 context 上限）用空心方块标注在对应 k 上，不连线。

## HTML 报告结构

```
<h1>TabPFN vs MLR: Rows & Columns Probing</h1>
<nav>  数据集目录跳转  </nav>

<section id="diabetes">
  <h2>diabetes</h2>

  <h3>列探索</h3>
  <img src="...side_by_side.png">
  <details>
    <summary>逐层 TabPFN attention</summary>
    <img src="...per_layer.png">
  </details>

  <h3>行探索</h3>
  <img src="...row_curve.png">
  <table>  ...JSONL 的聚合表格：按 (model, mode, k) mean±std ...  </table>
</section>
```

单文件自包含，不依赖外部 CSS/JS。`<style>` 写在 `<head>`。

## 步骤

1. 实现 `plot_column_heatmaps`，在 `diabetes` 上跑通，把 4 张 png 贴回对话（Claude Code 如果能贴图就贴，否则列路径 + 断言 shape / dtype）。**停**。
2. 实现 `plot_row_curves`，在 `diabetes` 上跑通，验证 MLR/exact 线确实水平。若不水平，停下来排查 `02` 或 `04`。**停**。
3. 实现 `build_report` + `scripts/build_report.py`，生成一份包含 `diabetes` + `synth_linear` 的报告。把 HTML 前 100 行（结构骨架，不含 base64）贴回。

## 非目标

- 不做交互式可视化（d3 / plotly），纯 matplotlib 静态图即可。
- 不做跨数据集聚合（例如全局 leaderboard）——本项目目前只关心单数据集内的对比。
- 不在本 prompt 引入 LaTeX 渲染 / 论文级别排版。

## 完成标准

- `python scripts/build_report.py --dataset diabetes --dataset synth_linear` 一次成功。
- 打开 `results/report.html`，三张热力图同 colorbar、曲线图有 4 条线、MLR/exact 是明显水平线。
