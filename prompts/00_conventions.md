# 00 - 项目全局约定

本文件是所有后续 prompt（01~05）的前置依赖。每份 prompt 开头都默认你已经读过这里。

## 目录布局

```
verification/
├── prompts/              # 本目录，给 Claude Code 的分步任务书
├── scripts/              # Python 入口脚本，面向命令行
├── src/                  # 库代码，所有模块在这里
│   ├── configs.py
│   ├── data/
│   ├── models/
│   ├── probing/
│   ├── viz/
│   └── utils/
├── third-party/
│   └── tabpfn/           # 通过 `pip install -e ./third-party/tabpfn` 安装
└── results/
    ├── column/{dataset}/{model}.{npy,png}
    └── row/{dataset}.jsonl
```

## 必须遵守

1. **不修改 `third-party/` 下任何文件**。需要 hook tabpfn 行为时，只能在 `src/` 用 monkey-patch / 上下文管理器。
2. **路径统一用 `pathlib.Path`**。不允许字符串拼路径，不允许写死任何机器特定的绝对路径（无论 Windows 盘符还是 POSIX 挂载点）。
3. **随机种子统一走 `src/utils/seed.py` 的 `set_seed(seed)`**，一次性设定 numpy / torch / python random / cuda。
4. **配置集中在 `src/configs.py`**，dict-based，不使用 yaml / hydra。所有超参必须可从 CONFIG 里查到，禁止散落在函数默认值里。
5. **结果文件**：
   - 数值矩阵存 `.npy`，字段语义写在同目录 `README.md` 或 docstring 里。
   - 标量指标存 `.jsonl`，一行一条记录，字段顺序稳定，便于 `pandas.read_json(lines=True)`。
6. **测试**：每个新增模块必须有对应的 `tests/test_<module>.py`，至少一个 sanity check 级别的 assert。测试用小数据（`n < 500`），能在 CPU 跑完。
7. **docstring 用纯文本，不用 Markdown 语法**。章节用大写标签，例如 `ARGS:` / `RETURNS:` / `RAISES:`。

## 编码细节

- Python 3.10+，类型注解用 `X: np.ndarray` 这种内建语法，不用 `from typing import List` 风格。
- NaN 保留给 TabPFN 和 tree-based 模型，**不在通用预处理里做 imputation**。MLR 侧需要时单独处理。
- Nominal 编码列（`pd.factorize` 出来的）**不做 z-score**。数据加载器必须返回 `is_nominal: list[bool]` 标记。
- TabPFN 所有调用显式设 `n_estimators=1` 并关闭特征 shuffle，便于与 MLR 的 W 做特征顺序对齐。见 `01` 的源码核对步骤。

## 分步交付协议（重要）

每份 prompt 的「步骤」章节是严格线性的。对每一步：

1. CC 只完成当前这一步，交付物落到指定文件。
2. 把关键证据（测试输出 / 源码片段 / 数值）贴回对话，**然后停下来等我说「继续」**。
3. 我没说继续之前，不要跨步骤提前实现。
4. 不要顺手重构与当前步骤无关的代码。

## 禁止事项

- 禁止 scope creep：prompt 里没写的不要做。比如 `02` 不要画图，`03` 不要做统计显著性。
- 禁止猜测 tabpfn 内部 API。凡是涉及 tabpfn 内部类/属性的代码，必须先在 `third-party/tabpfn/src/` 下找到定义并在提交里注明源文件相对路径 + 行号。
- 禁止把临时调试用的 `print` 留在提交里。需要日志就用 `logging`。
