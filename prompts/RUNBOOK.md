# RUNBOOK - 全自动连续执行

本文件是 `prompts/00~05` 之上的总指挥。把本文件完整内容发给 CC,它会按顺序跑完全部 phase,不中途停下等 review。

---

## 总览

```
Phase 0   读取 00_conventions.md              (仅读,不写代码)
Phase 1   执行 02_mlr_wrapper.md               → src/models/mlr_wrapper.py
Phase 2   执行 01_tabpfn_attention_extractor.md → src/models/tabpfn_wrapper.py
Phase 3   执行 03_column_probe.md              → src/probing/column_probe.py + scripts/
Phase 4   执行 04_row_probe.md                 → src/probing/row_probe.py + scripts/
Phase 5   执行 05_viz_and_report.md            → src/viz/*.py + scripts/build_report.py
Phase 6   端到端 sanity run                    → 产出真实 results/ 文件
Phase 7   写 FINAL_REPORT.md                   → 汇总所有状态
```

为什么先 02 后 01:02 的 MLR 没有 GPU / tabpfn 依赖,是整条链路里最稳的一环。先打通 02 相当于先把"测试流水线、pathlib、configs、io、seed 是否都接得上"这件事验证掉,后面出问题更容易定位。

---

## 全局运行规则 (全 phase 适用)

### 自主决策授权

对以下情况,**不必停下询问用户,自己决定并记录到 `DECISIONS.md`**:

- 代码实现细节:变量命名、注释、格式、私有辅助函数拆分
- 测试用例的具体构造方式(只要覆盖 prompt 列出的断言点)
- 当前文件内的小规模 refactor
- 错误消息的具体文案
- prompt 里描述的 API 与 tabpfn 实际源码存在细微差异时,选择功能等价的最接近实现
- matplotlib/seaborn 的样式细节(色板、字号、margin)

对以下情况,**必须停下并写入 `BLOCKERS.md`,然后跳过当前 phase 进入下一个**:

- 某测试连续 3 次修改后仍失败
- 需要修改 `third-party/` 下任何文件才能通过
- 需要 `pip install` 一个 `requirements.txt` 中不存在的新包
- tabpfn 源码结构与 prompt 描述相差过大,无法通过同构实现绕过

### 测试失败重试策略

对每个失败的测试,最多 **3 轮修改-重试**:

```
Attempt 1: 分析 pytest 输出 → 修改代码 → 重跑
Attempt 2: 失败则重新读相关源码 → 修改 → 重跑
Attempt 3: 失败则考虑是测试本身写错了 → 修改测试或代码 → 重跑
Attempt 4+: 不要继续。把完整错误栈 + 每轮的 diff 写入 BLOCKERS.md,
           标记该 phase 为 PARTIAL,进入下一 phase。
```

### 禁止事项

- ❌ 不要 `pip install` 任何包。需要的包假定已在环境里。缺了就记录到 BLOCKERS.md。
- ❌ 不要修改 `third-party/`。每完成一个 phase 跑 `git diff third-party/` 确认为空。
- ❌ 不要在提交代码里留 `print`。调试完删掉,需要日志用 `logging`。
- ❌ 不要在 RUNBOOK 要求之外创建 "bonus" 文件(例如 README、额外的 notebook)。

### 进度文件 (CC 必须维护)

- **PROGRESS.md**:每进入/完成一个 phase 时 append 一段。格式:
  ```
  ## Phase N - <name>
  Started:  2026-04-22 14:03
  Finished: 2026-04-22 14:27
  Status:   PASS | PARTIAL | SKIPPED
  Tests:    6/6 passed
  Files:    src/models/mlr_wrapper.py, tests/test_mlr_wrapper.py
  Notes:    (任何偏离 prompt 的地方)
  ```

- **DECISIONS.md**:记录所有自主决策。格式:
  ```
  ## <date> Phase N
  Decision: <一句话结论>
  Context:  <prompt 原文 / 源码差异>
  Rationale:<为什么这样做>
  ```

- **BLOCKERS.md**:记录 3 次重试仍失败的情况。格式:
  ```
  ## Phase N - <test name>
  Full error: <完整 pytest 输出>
  Attempts:
    1. <改了什么> → <失败原因>
    2. ...
    3. ...
  Suspected root cause: <猜测>
  ```

三个文件都放在 `<repo_root>/` 下(与 `prompts/` 同级)。

### override 子 prompt 的停顿指令

`00~05` 中所有出现以下措辞的地方,**按下列规则重新解释**:

| 子 prompt 原文 | RUNBOOK 下的解释 |
|---|---|
| "停下来等我 review" | 通过测试后立即进入下一步 |
| "把 ... 贴回对话" | 写入 `PROGRESS.md` 对应 phase 段 |
| "我确认后再开步骤 X" | 直接开步骤 X |
| "把 pytest 输出贴回" | 保存完整输出到 `PROGRESS.md`,继续 |

---

## Phase 0 - 读取约定

**Prompt**: `@prompts/00_conventions.md`

**Action**: 只读,不写代码。在 PROGRESS.md 写一段 "Phase 0" 记录,列出你理解的三条最硬的约束(目录不改、third-party 不动、分步但全自动)。

**Pass 条件**: PROGRESS.md 中有 Phase 0 段落。

---

## Phase 1 - MLR wrapper

**Prompt**: `@prompts/02_mlr_wrapper.md`

**Depends on**: Phase 0

**执行方式**: 按 prompt 步骤 1→2→3 连续做。每步写完立即跑对应测试,通过后直接进入下一步,**不停下**。

**Pass 条件**:
- `pytest tests/test_mlr_wrapper.py -v` 全绿
- 特别验证"样本复制不变性"测试通过(`atol=1e-8`)
- `src/models/mlr_wrapper.py` 存在且包含 `class MLRWithW`

**Fallback (partial pass)**: 若"nominal 跳过"测试(第 4 条)失败,其他通过,可标记 PARTIAL 继续——后续 phase 用的数据集(diabetes / cali_housing / synth_linear)默认都是连续变量,不走 nominal 分支。

---

## Phase 2 - TabPFN 列注意力提取

**Prompt**: `@prompts/01_tabpfn_attention_extractor.md`

**Depends on**: Phase 0 (Phase 1 失败不影响本 phase)

**执行方式**:
1. **步骤 1 的源码侦察不贴给用户,直接写入 `PROGRESS.md` 的 Phase 2 段**,格式:
   ```
   ### Phase 2 源码侦察报告
   - tabpfn 版本: X.Y.Z
   - 列注意力类名: <classname>
   - 定义位置: third-party/tabpfn/src/.../file.py:LINE_NO
   - forward 签名: def forward(self, ...)
   - 全部层的获取路径: model.<path>
   - 关闭特征 shuffle 的姿势: <method>
   ```
2. 侦察写完立即进入步骤 2(上下文管理器实现)。
3. 步骤 2→3→4 连续做,每步跑测试,通过即前进。

**Pass 条件**:
- `pytest tests/test_tabpfn_attn.py -v` 全绿
- `git diff third-party/` 输出为空
- `src/models/tabpfn_wrapper.py` 包含 `class TabPFNWithColAttn` 和 `capture_column_attention`

**Fallback**: 若本 phase PARTIAL 或 SKIPPED:
- Phase 3(列探索)仍然要跑,但只产出 MLR 分支的 `mlr.npz`
- Phase 4(行探索)仍然要跑,但只产出 `model="mlr"` 的记录
- Phase 5(可视化)仍然要跑,热力图只画 MLR 侧(`w_vec` + `w_outer`),TabPFN 侧跳过并在 HTML 报告标注"skipped due to Phase 2 failure"

---

## Phase 3 - 列探索

**Prompt**: `@prompts/03_column_probe.md`

**Depends on**: Phase 1 (必需), Phase 2 (可选,见 fallback)

**执行方式**: 步骤 1→4 连续。测试每步都跑,最后通过 `scripts/run_column_probe.py` 在命令行跑一次 `--dataset diabetes --dataset synth_linear`,产物验证。

**Pass 条件**:
- 三个数据集的 loader 单测通过
- `results/column/diabetes/` 和 `results/column/synth_linear/` 下都有 `mlr.npz`, `tabpfn.npz`(Phase 2 跳过时只有 mlr.npz), `meta.json`
- `feature_names` 在 mlr.npz 和 tabpfn.npz 中完全一致(assert)

**Fallback**: 若 cali_housing 因 context 大 / 内存问题失败,跳过它,保留 diabetes 和 synth_linear。

---

## Phase 4 - 行探索

**Prompt**: `@prompts/04_row_probe.md`

**Depends on**: Phase 1 (必需), Phase 2 (可选)

**执行方式**: 步骤 1→4 连续。**务必优先验证 sanity check**:步骤 2 之后必须在 PROGRESS.md 贴一段证据,证明 `model=mlr, mode=exact` 在不同 k 下 R² 完全相等。若不等,立即停止本 phase(不重试),写 BLOCKERS.md。这是整条实验的底线。

命令行 sanity run: `--dataset diabetes --fresh --k-list 1,2,3 --seeds 0`(小规模先跑通),再跑 `--k-list 1,2,3,5,10 --seeds 0,1,2`。

**Pass 条件**:
- `pytest tests/test_row_probe.py -v` 全绿
- `results/row/diabetes.jsonl` 存在,行数 = `len(models) × len(k_list) × len(modes) × len(seeds)`(Phase 2 失败时 `len(models)=1`)
- Sanity check 数值验证通过

**Fallback**: 若 TabPFN 某些 `(k, mode, seed)` 组合 OOM 或超 context 上限,写 `{"skipped": true, ...}` 记录,不终止整个 phase。

---

## Phase 5 - 可视化 & 报告

**Prompt**: `@prompts/05_viz_and_report.md`

**Depends on**: Phase 3(列探索产物)、Phase 4(行探索产物)

**执行方式**: 步骤 1→3 连续。只吃文件,不碰模型。

**Pass 条件**:
- 每个数据集都生成 `side_by_side.png` + `row_curve.png`
- `results/report.html` 生成,文件大小 > 100KB(证明 base64 图片嵌入成功)
- MLR/exact 曲线目视水平(容差 R²[k=1] 与 R²[k=10] 差异 < 1e-6)

**Fallback**: 若某个数据集的产物缺失,跳过该数据集的可视化,在报告里写占位块。

---

## Phase 6 - 端到端 sanity run

**不对应任何 prompt**,是整条 pipeline 的最终回放。

**Action**:
```bash
# 1. 清空旧产物
rm -rf results/

# 2. 按序跑三条命令
python scripts/run_column_probe.py --dataset diabetes --dataset synth_linear
python scripts/run_row_probe.py    --dataset diabetes --dataset synth_linear --fresh
python scripts/build_report.py     --dataset diabetes --dataset synth_linear

# 3. 验证最终产物
ls -la results/
```

把三条命令的完整输出写到 `PROGRESS.md` 的 Phase 6 段。

**Pass 条件**: 三条命令全部 exit code 0,`results/report.html` 存在。

---

## Phase 7 - FINAL_REPORT.md

在 repo 根生成 `FINAL_REPORT.md`,内容:

```markdown
# 最终报告

## Phase 状态总览
| Phase | Status | Tests | Duration |
|-------|--------|-------|----------|
| 0     | PASS   | -     | -        |
| 1     | PASS   | 6/6   | 12s      |
| ...

## 生成的关键文件
- src/models/mlr_wrapper.py
- src/models/tabpfn_wrapper.py
- ...

## 决策摘要 (详见 DECISIONS.md)
- Phase 2: 选择通过 xxx 方式关闭特征 shuffle,因为 ...
- ...

## 阻塞点 (详见 BLOCKERS.md)
- (若无,写 "None")

## 建议的下一步
- 用户需要人工验证: ...
- 已知遗留问题: ...

## 可复现命令
(把 Phase 6 的三条命令贴在这里)
```

---

## 开始执行

现在按 Phase 0 → Phase 7 顺序开始,中途不要停下询问。完成 Phase 7 后输出一句:

```
=== RUNBOOK COMPLETE. 详见 FINAL_REPORT.md ===
```
