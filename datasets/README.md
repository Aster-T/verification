# datasets/

本目录存放项目使用的数据集的**本地快照**,用于离线复现、审查、以及投放自备
数据。pipeline 既能从 sklearn / OpenML / `make_regression` 取数,也能**直接
读本目录** —— 只要 `datasets/<name>/meta.json` 存在,
`load_dataset_full(name, seed)` 会自动按 `loader=local_csv` 注册并加载。

## 目录结构

```
datasets/
  README.md                       # 本文件
  <dataset_name>/
    data.csv                      # 最后一列是 target,其他列是特征
    meta.json                     # 元信息(见下)
```

每个数据集独占一个子目录,子目录名与 `src/configs.py` 中 `CONFIG["datasets"]`
的 key 一致(例如 `diabetes`, `synth_linear`, `cali_housing`, `openml_560`)。

## `data.csv` 约定

- UTF-8,带表头
- **最后一列是回归目标**,列名见 `meta.json.target_col`(一般叫 `target` 或 `y`)
- 其他列即特征,列名对应 `meta.json.feature_names`
- 类别型特征已预先因子化为整数(对应 `meta.json.is_nominal[i] == true`);
  缺失值写作空单元格(`pandas.read_csv` 会读成 NaN)
- 本项目是回归项目,**不接受分类目标**

## `meta.json` 字段

```json
{
  "name":             "diabetes",
  "source":           "sklearn.datasets.load_diabetes",
  "generator_params": null,
  "target_col":       "target",
  "feature_names":    ["age", "sex", "bmi", ...],
  "is_nominal":       [false, false, false, ...],
  "n_rows":           442,
  "n_features":       10,
  "seed":             0,
  "exported_at":      "2026-04-24",
  "sha256":           "ab12...89cd"
}
```

- `source`:sklearn 函数、OpenML 链接、或 `"user_supplied"`
- `generator_params`:合成数据集填入 `make_regression` 参数,其他填 `null`
- `seed`:生成/子采样时使用的 seed(OpenML 的子采样也依赖这个)
- `sha256`:对 `data.csv` 整个文件取的十六进制摘要,用于验证未被篡改

## 如何导出内置数据集到本目录

```bash
python scripts/export_datasets.py --dataset diabetes --dataset synth_linear
python scripts/export_datasets.py --dataset cali_housing
```

## OpenML 数据集批量管理

在 `datasets/openml.json` 维护一份常用 OpenML 数据集清单(key 即注册名):

```json
{
  "bodyfat":      {"id": 560,   "subsample": 2000, "description": "..."},
  "house_prices": {"id": 41021}
}
```

`_` 开头的 key 会被忽略(方便写注释)。之后任何 CLI 脚本都可以:

```bash
# 临时加一个,自动名 openml_560
python scripts/export_datasets.py --openml-id 560

# 临时加一个,自定义名
python scripts/export_datasets.py --openml-id 560:my_bodyfat

# 从 datasets/openml.json 里挑一个
python scripts/export_datasets.py --openml-preset bodyfat

# 从 datasets/openml.json 里挑多个
python scripts/export_datasets.py --openml-preset bodyfat --openml-preset house_prices

# 把 datasets/openml.json 里的全部跑一遍
python scripts/export_datasets.py --openml-all

# 用别的配置文件
python scripts/export_datasets.py --openml-config path/to/other.json --openml-all
```

4 个 CLI(`export_datasets` / `run_column_probe` / `run_row_probe` /
`build_report`)都共享这套 `--openml-*` 参数。

**subsample 优先级**:preset 条目里的 `subsample` > CLI `--openml-subsample`
> 默认**不设上限**。CLI 传 `--openml-subsample 0` 或任何非正整数也表示"不限"。
行探针本身**不再**限制 n_ctx;但 TabPFN v2 在 n_ctx ≳ 10k 时代价陡增,若
显存吃紧建议显式给个 `--openml-subsample 2000` 或在 preset 里写
`"subsample": 2000` 作为输入层的软控制。

每次导出会覆盖目标子目录下的 `data.csv` 和 `meta.json`。

## 如何投放自备数据集

1. `mkdir datasets/<my_dataset>`
2. 把 `data.csv` 放进去,**最后一列**是回归目标(或用 `--target-col` 覆盖)
3. 自动生成 meta.json:`python scripts/infer_meta.py datasets/<my_dataset>/data.csv`
   (或手写 meta.json,字段同上)
4. 直接用:任何 `--dataset <my_dataset>` 都会走 `loader=local_csv` 自动注册

`src/configs.py::get_dataset_cfg(name)` 在 CONFIG miss 时会扫 `datasets/<name>/`
自动注册,不需要手动改 CONFIG。

## .gitignore 策略

默认**整个 `datasets/*` 都不进 git**(见 repo 根 `.gitignore`),只有
`README.md` 和 `openml.json` 两个配置文件被保留。原因:
- sklearn / OpenML 数据有各自 license,未必适合二次分发
- 数据可由 `export_datasets.py` + `openml.json` 随时重建

如果有可公开的自备数据,手动 `git add -f datasets/<name>/` 即可破例入库。
