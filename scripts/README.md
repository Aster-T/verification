# scripts/

CLI 入口集合。每个脚本都是 `if __name__ == "__main__": main()` 的薄包装,
真正的逻辑在 `src/` 下。

| 类型 | 脚本 |
|---|---|
| 数据准备 | [`export_datasets.py`](#export_datasetspy)、[`infer_meta.py`](#infer_metapy) |
| 数据观察 | [`plot_feature_distributions.py`](#plot_feature_distributionspy)（特征分布 box plot） |
| 实验 | [`run_column_probe.py`](#run_column_probepy)、[`run_row_probe.py`](#run_row_probepy) |
| 一键扫一遍 | [`run_row.py`](#run_rowpy) |
| 汇报 | [`rebuild_reports.py`](#rebuild_reportspy)（重生成 PNG）、[`serve_report.py`](#serve_reportpy)（起服务看图） |

> LOO 不再有独立脚本 —— 用 `run_row_probe.py --split-mode loo`,逐点预测 CSV
> 会自动落到 `results/<dataset>/row/predictions_*.csv`。

---

## 约定

- **Python 解释器**:用项目自带环境的 `python`(必须能 import `tabpfn`,即指向 `pip install -e third-party/TabPFN` 安装的解释器)
- **运行目录**:在项目根下运行
- **`-v` / `--verbose`**:所有脚本都接受,打开后级别从 WARNING 变为 INFO
- **数据集 3 种来源**,都能通过 `--dataset <name>` 引用:
  - `src/configs.py::CONFIG["datasets"]` 里的内置项
  - OpenML(走下方共享的 `--openml-*` 参数)
  - 本地 CSV(自动发现:`datasets/<name>/meta.json` 存在就注册为 `local_csv`)

### 统一的产物布局

```
results/
  feature_distributions/          # plot_feature_distributions.py 的产物
    <dataset>.png                 #   每数据集一张:每列 box plot + target
  sigma_<σ>/[jitter_<scale>/]     # σ-分区 + 可选 jitter_scale 子分区
    <dataset>/                    # LOO + column probe 区(test_size 无关)
      column/                     #   run_column_probe.py 的产物
        mlr.npz                   #     w_vec (F,), w_outer (F,F), feature_names
        tabpfn.npz                #     col_attn (F,F), col_attn_per_layer (L,F,F)
        meta.json
      row/                        #   run_row_probe.py 的 LOO 产物
        metrics.jsonl             #     每个 (model, split, k, mode, seed) 一条聚合
        predictions_<combo>.csv   #     每条非 skip 记录配一份逐点预测
                                  #     <combo> = <model>_<split>_<mode>_k<k>_s<seed>
                                  #     列: id, y_true, y_pred, residual
      viz/                        #   rebuild_reports.py 的产物 (PNG)
        side_by_side.png          #     MLR w / w⊗w / TabPFN col-attn 三联
        tabpfn_per_layer.png      #     TabPFN 每层 attention 网格
        row_curve.png             #     nRMSE vs n_ctx (4 线 combined)
        row_curve_<model>.png     #     单模型 (exact + jitter)
        row_curve_<model>_<mode>.png   #  4 个 facet 子图
    test_size_<ts>/<dataset>/     # proportional 区(每 test_size 独立)
      row/metrics.jsonl
      viz/row_curve*.png
```

`jitter_<scale>/` 子层只在非默认(`per_col_std`)时出现;`absolute` 路径保持
和老版本一致。

> 不再有静态 `report.html` —— 起 `scripts/serve_report.py` 直接看 live 报告。

### 共享的 OpenML 参数(所有实验/汇报脚本)

| 参数 | 含义 |
|---|---|
| `--openml-id ID[:NAME]` | 临时指定一个 OpenML `data_id`,可重复。`560` → 注册为 `openml_560`;`560:bodyfat` → 注册为 `bodyfat` |
| `--openml-preset NAME` | 从 `datasets/openml.json` 读一个 preset(key 即注册名),可重复 |
| `--openml-all` | 加载 `datasets/openml.json` 里**全部** preset |
| `--openml-config PATH` | 覆盖默认 preset 路径(默认 `datasets/openml.json`) |
| `--openml-subsample N` | 子采样上限。`0` 或负 = 不限;默认不限。preset 条目的 `subsample` 优先于此 |

preset 文件格式:

```json
{
  "_comment": "'_' 开头的 key 被忽略",
  "bodyfat": {"id": 560, "subsample": 2000, "description": "..."},
  "house_prices": {"id": 41021}
}
```

---

## `export_datasets.py`

把已注册数据集**物化**到 `datasets/<name>/data.csv` + `meta.json`。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset NAME` | - | 已注册数据集名,可重复 |
| `--openml-*` | - | 共享参数 |
| `--out PATH` | `datasets/` | 输出根 |
| `--seed INT` | `0` | 用于 `synth_linear` 生成 / OpenML 子采样 |
| `-v` | off | 详细日志 |

```bash
python scripts/export_datasets.py --dataset diabetes
python scripts/export_datasets.py --openml-id 560:bodyfat --openml-subsample 2000
python scripts/export_datasets.py --openml-all
```

---

## `infer_meta.py`

从 `data.csv` 反推 `meta.json`。用于投放自备数据集。

| 参数 | 默认 | 含义 |
|---|---|---|
| `csv`(位置参数) | 必填 | CSV 路径,meta.json 生成到同目录 |
| `--target-col NAME` | 最后一列 | target 列名 |
| `--source STR` | `user_supplied` | 来源备注 |
| `--nominal a,b,c` | `""` | 强制把这些数值列标为 nominal |
| `--no-factorize` | off | 遇到字符串列报错退出(默认就地因子化) |
| `--seed INT` | `null` | 如 CSV 来自某 seed,记录下来 |
| `--overwrite` | off | 覆盖已有 meta.json |
| `-v` | off | 详细日志 |

```bash
python scripts/infer_meta.py datasets/ship/data.csv
python scripts/infer_meta.py datasets/ship/data.csv --target-col draft
python scripts/infer_meta.py datasets/ship/data.csv --nominal engine_type,flag
```

关键自动行为:target 必须数值;字符串特征默认就地因子化并回写 CSV。

---

## `plot_feature_distributions.py`

每个数据集出一张 PNG,把每个特征当列做 box plot,用来在跑 probe 之前快速看
列尺度差异、异常值、target 长尾等。文本列改成 top-N value-count 的横向柱状
图。产物落在 `results/feature_distributions/<dataset>.png`,跟 σ / test_size /
jitter_scale 都没关系——它是 dataset 级别的事实。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset NAME` | - | 已注册数据集名,可重复 |
| `--openml-*` | - | 共享参数 |
| `--local-all` | off | 自动扫 `datasets/<name>/meta.json` 全量入列 |
| `--out PATH` | `results/feature_distributions/` | 输出根 |
| `--seed INT` | `0` | 给 OpenML 子采样 / `make_regression` 用 |
| `--cols-per-row N` | `4` | 子图网格宽度 |
| `--top-categorical N` | `10` | 文本列只画 top-N 高频值 |
| `--no-target` | off | 不画 target 子图(默认会画,琥珀色与特征蓝区分) |
| `-v` | off | 详细日志 |

```bash
# 单个数据集
python scripts/plot_feature_distributions.py --dataset diabetes

# 本地数据集全量 + 所有 OpenML preset
python scripts/plot_feature_distributions.py --local-all --openml-all
```

数值列子图底部带 `n=… μ=… ± …  [min, max]`。数值 nominal 列(整数类别码)
仍画 box,标题加 `[nominal]` 后缀。`scripts/serve_report.py` 启动后会自动把
这些 PNG 渲染在 "特征分布" 区(夹在 "宏指标" 和 gallery 之间),由
**dataset 筛选**驱动,与 σ/test_size/scale 解耦。

字体:已配置 CJK 回退(`Microsoft YaHei` / `PingFang SC` / `Noto Sans CJK SC`
/ `WenQuanYi Zen Hei` / `SimHei`),含中文列名(如 `船号`)的数据集不会
出现 missing-glyph 方块。

---

## `run_column_probe.py`

训练 MLR + TabPFN,导出系数外积与列注意力。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset NAME` | - | 已注册数据集名,可重复 |
| `--openml-*` | - | 共享参数 |
| `--out PATH` | `results/` | 根目录,产物到 `<out>/<dataset>/column/` |
| `--seed INT` | `0` | 种子 |
| `-v` | off | 详细日志 |

```bash
python scripts/run_column_probe.py --dataset diabetes --dataset synth_linear
python scripts/run_column_probe.py --openml-preset bodyfat
```

产物:`results/<dataset>/column/{mlr.npz, tabpfn.npz, meta.json}`。

---

## `run_row_probe.py`

行探针:把训练 context 复制 k 倍,衡量 nRMSE/R²/RMSE/MAE 随 k 的变化。
两种 split 模式:按比例切 或 LOO。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset NAME` | - | 已注册数据集名,可重复 |
| `--openml-*` | - | 共享参数 |
| `--k-list "2,3,5,10"` | `2,3,5,10` | context 复制倍数列表。加 `1` 表示 no-duplication 基线 |
| `--modes "exact,jitter"` | `exact,jitter` | `exact` = 原样复制;`jitter` = 加 N(0, 1e-6) 微噪声 |
| `--seeds "0,1,2"` | `0,1,2` | 影响切分 / 合成数据生成 / jitter rng |
| `--split-mode` | `proportional` | `proportional`:train/test 切分,复制 train,预测 test。`loo`:留一,每折复制 N-1 行,聚合 N 次预测 |
| `--no-tabpfn` | off | 只跑 MLR(冒烟用) |
| `--fresh` | off | 清空 `<out>/<dataset>/row/` 再开始 |
| `--out PATH` | `results/` | 根目录,产物到 `<out>/<dataset>/row/` |
| `-v` | off | 详细日志 |

```bash
# 经典:所有 mode/seed 跑一遍
python scripts/run_row_probe.py --dataset diabetes --fresh

# LOO,只 MLR(秒级)
python scripts/run_row_probe.py --dataset ship --split-mode loo --no-tabpfn \
    --seeds 0 --fresh

# 冒烟:单 k、单 seed、单 mode
python scripts/run_row_probe.py --dataset diabetes --k-list 2 --seeds 0 \
    --modes exact --no-tabpfn --fresh

# 所有 preset
python scripts/run_row_probe.py --openml-all --fresh
```

### 产物

```
results/<dataset>/row/
  metrics.jsonl                                  # 每个 combo 一条聚合记录
  predictions_<model>_<split>_<mode>_k<k>_s<seed>.csv
```

**`metrics.jsonl` 记录 schema(17 字段)**

```json
{
  "dataset": "...", "model": "mlr|tabpfn",
  "split_mode": "proportional|loo",
  "k": 2, "mode": "exact|jitter", "seed": 0,
  "n_ctx": 706,         // 每次模型调用的 context 样本数
  "n_query": 89,        // 每次模型调用的 query 样本数 (loo 下 = 1)
  "n_folds": 1,         // 聚合的折数 (proportional = 1, loo = N)
  "n_features": 10,
  "y_query_std": 71.6,  // nrmse 分母
  "nrmse": 0.8172,      // 主指标,跨数据集可比
  "r2": 0.3322, "rmse": 58.5, "mae": 46.2,
  "mape": 0.124,        // 相对误差 mean(|y_true-y_pred|/|y_true|),仅在 y_true!=0 上聚合;全 0 时为 null
  "mape_n": 89,         // 实际进入 mape 的样本数(y_true!=0 的行数)
  // 仅在 ship-all / ship-selected 上出现:油船子集(船号 ∈ 9 艘)上的相对误差
  "mape_tanker": 0.083, "mape_tanker_n": 12,
  "fit_sec": 0.001, "predict_sec": 0.0
}
```

skip 条目(**只**在 fit / predict 实际抛异常时产生,不再有预判阈值)写成
`{"skipped": true, "reason": "<ExceptionType>: <msg>", <base fields>}`,
**不**配 CSV。

**`predictions_<combo>.csv` 列**

| 列 | 含义 |
|---|---|
| `id` | query 点在 query 集里的行索引(proportional 下是 test 集下标 0..n_te-1;loo 下是完整数据集下标 0..N-1) |
| `y_true` | 真实值 |
| `y_pred` | 预测值 |
| `residual` | `y_true - y_pred` |

### 性能提示

- TabPFN 对 n_ctx 没有硬上限;只有在 fit / predict 真正抛异常(通常是 OOM)
  时才 skip,skip 记录里的 `reason` 带真实 exception type
- LOO + TabPFN:成本约 N × |k_list| × |modes| × |seeds| 次 fit。**小数据集才用**

---

## `rebuild_reports.py`

扫一遍 `results/sigma_*/` 子目录，对每个有 `row/metrics.jsonl` 或
`column/mlr.npz` 的位置重新生成 viz/ 下的 PNG。**不**跑模型，**不**生成 HTML。
改了 `src/viz/curves.py` / `src/viz/heatmap.py` 想刷图就跑这个。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--results-root PATH` | `results/` | 读产物的根 |
| `-v` | off | 详细日志 |

```bash
python scripts/rebuild_reports.py
```

---

## `serve_report.py`

起一个 HTTP server，提供单页报告：可筛选（σ / test_size / dataset / 图类型）、
可"加入对比"挑图横向并排、表格按需展开懒加载。**不**写任何静态文件。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--port` | `8000` | 端口 |
| `--bind` | `127.0.0.1` | 绑定地址。`0.0.0.0` = 暴露到 LAN |
| `--results-root PATH` | `results/` | 报告读取的根目录 |
| `-v` | off | 详细日志 |

```bash
python scripts/serve_report.py
# 然后浏览器打开 http://localhost:8000/
```

页面区块自上而下:Filters → 宏指标 → **特征分布** → Gallery → Data tables;
右侧固定 Compare 栏。"特征分布" 区只看 `dataset` 筛选,把
`results/feature_distributions/<dataset>.png` 直接列出来,点图进 lightbox
全屏放大,也可以 `+ Compare` 钉到右栏跟其他图并排比。

Endpoints（前端会自己用，调试时可手动访问）：

| 路径 | 说明 |
|---|---|
| `GET /` | 单页前端（HTML + 内联 CSS/JS） |
| `GET /manifest.json` | 启动时扫一遍 `results/` 得到的所有可用 (σ, test_size, dataset, chart, jitter_scale) 元组,以及 `feature_distributions[]` |
| `GET /results/<rel>` | 透传 `results/` 下的文件（PNG） |
| `GET /table?jsonl=<rel>` | 按需聚合一份 `metrics.jsonl`，返回每 (split, model, mode, k) 一行的 JSON |

---

## `run_row.py`

一次性扫一遍所有 (σ, test_size) 组合，跑 LOO + proportional + 自动重生成
PNG。配置在脚本顶部（`FRESH` / `LOCAL_DATASETS` / `OPENML_DATASETS` /
`JITTER_SIGMAS` / `TEST_SIZES` / `SEEDS` / `K_LIST`），无 CLI 参数。

```bash
python scripts/run_row.py
# 跑完后：
python scripts/serve_report.py
```

---

## 端到端典型 pipeline

```bash
PY=python   # 或绝对路径,指向你装了 tabpfn (editable) 的 conda env 的 python

# 1. 自备数据集(可选)
$PY scripts/infer_meta.py datasets/ship/data.csv

# 2. 列探针 + 行探针
$PY scripts/run_column_probe.py --dataset diabetes --dataset ship
$PY scripts/run_row_probe.py    --dataset diabetes --dataset ship --fresh

# 3. 把 PNG 刷一遍 + 起服务看
$PY scripts/rebuild_reports.py
$PY scripts/serve_report.py
```

浏览器打开 http://localhost:8000/ 看结果。
