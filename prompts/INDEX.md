# Prompts Index

本目录是 `D:\projects\verification\prompts\` 的完整内容。

## 文件清单

| 文件 | 作用 | 产出 |
|---|---|---|
| `00_conventions.md` | 全局约定(路径、编码风格、禁止事项) | 无,仅被其他 prompt 引用 |
| `01_tabpfn_attention_extractor.md` | TabPFN 列注意力 monkey-patch 提取器 | `src/models/tabpfn_wrapper.py` |
| `02_mlr_wrapper.md` | MLR 封装 + W 矩阵导出 | `src/models/mlr_wrapper.py` |
| `03_column_probe.md` | 列探索主流程(拟合 + 导出 npz) | `src/probing/column_probe.py` + `scripts/run_column_probe.py` |
| `04_row_probe.md` | 行探索主流程(context 复制 + 评估) | `src/probing/row_probe.py` + `scripts/run_row_probe.py` |
| `05_viz_and_report.md` | 热力图、曲线、HTML 报告 | `src/viz/*.py` + `scripts/build_report.py` |
| `RUNBOOK.md` | **总指挥**,全自动连续执行 | 指挥 CC 按序跑 00→07 |

## 两种用法

### 全自动(推荐)

项目根目录开 CC,发:
```
请严格按 @prompts/RUNBOOK.md 执行。
```

RUNBOOK 会 override 01~05 中所有"停下来等 review"的指令,让 CC 连续跑完。
产物:`FINAL_REPORT.md`, `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md`(若有)。

### 半自动(手动 review 每步)

不读 RUNBOOK,直接发:
```
请先读 @prompts/00_conventions.md,然后执行 @prompts/02_mlr_wrapper.md 的步骤 1。
```

每步完成后 CC 会停下等你说"继续"。

## 前置依赖

本 prompts 预设以下文件已存在:
- `src/configs.py`
- `src/utils/io.py`
- `src/utils/seed.py`

若还没放,见对话早期提供的模板。

## 执行顺序(RUNBOOK 内部规则)

```
Phase 0: 读 00
Phase 1: 执行 02 (MLR,无 GPU 依赖,先验证底座)
Phase 2: 执行 01 (TabPFN,源码侦察是关键点)
Phase 3: 执行 03 (列探索,依赖 01+02)
Phase 4: 执行 04 (行探索,依赖 02,TabPFN 分支依赖 01)
Phase 5: 执行 05 (可视化,只读文件)
Phase 6: 端到端 sanity run
Phase 7: 生成 FINAL_REPORT.md
```

若 Phase 2 (TabPFN) 失败,Phase 3~5 会自动降级为仅 MLR 版本,不零产出。
