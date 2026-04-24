# datasets/

本目录存放项目使用的数据集的**本地快照**,用于离线复现、审查、以及投放自备
数据。**不是** pipeline 的强制数据源 —— `src/data/loaders.py` 默认仍从
sklearn / OpenML / `make_regression` 取数,本目录为可选镜像。

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

5 个 CLI(`export_datasets` / `run_column_probe` / `run_row_probe` / `run_loo`
/ `build_report`)都共享这套 `--openml-*` 参数。

**subsample 优先级**:preset 条目里的 `subsample` > CLI `--openml-subsample`
> 默认**不设上限**。CLI 传 `--openml-subsample 0` 或任何非正整数也表示"不限"。
跑 TabPFN 时(v2 context 上限 ~10k)建议显式给个 `--openml-subsample 2000`
或在 preset 里写 `"subsample": 2000`。

每次导出会覆盖目标子目录下的 `data.csv` 和 `meta.json`。

## 如何投放自备数据集

1. `mkdir datasets/<my_dataset>`
2. 准备 `data.csv`,**最后一列**是回归目标
3. 手写 `meta.json`(字段同上,`source` 填 `"user_supplied"`,`sha256` 可选)
4. (可选)如果要让 pipeline 直接用,再在 `src/configs.py` 注册一条
   `"loader": "local_csv"` 的 entry —— 当前 loaders.py **尚未实现** 这个分支,
   接入时再加即可

## .gitignore 策略

本目录下的 `data.csv` 默认**不进 git**(见 repo 根 `.gitignore`),因为:
- sklearn / OpenML 数据有各自的 license,未必适合二次分发
- 合成数据可由 `export_datasets.py` 随时重建

`meta.json` **进 git**,作为"此快照曾存在过"的记录。
