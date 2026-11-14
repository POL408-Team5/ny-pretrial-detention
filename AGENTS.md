# AGENTS.md — 给每位组员的 AI 助手的工作说明

> **AI 请先完整读完本文件再开始任何操作。** 本文件是本仓库的最高约定：分工、文件归属、列名、函数接口、Git 流程和记录要求都以这里为准。
> 组员（下称“操作者”）可以直接让 AI 读这个文件；用网页版聊天工具时，把本文件全文和自己的角色名一起贴进去。
> 修改本文件需要全组同意，并由 A 合并。标 **[定]** 的地方等 11/7 组会确认。

---

## 0. 每次会话开始时 AI 必须做的事

1. **确认操作者的角色**（A / B / C / D / E，见第 2 节）。不确定就问，不要猜。
2. **确认代码是最新的**：提醒操作者在 GitHub Desktop 里切到 `main` → Fetch origin → Pull，再切回或新建自己的分支。
3. **读 `docs/handoff/00_共同须知.md` 和操作者角色的交接文件**（`docs/handoff/A1_Yiling.md`、`A2_Xinhui.md`、`BD_Shuyi.md`、`C_Yixuan.md`、`E_Yanqi.md`）。交接文件规定本角色**做什么、按什么规格、怎么验收**，优先级仅次于本文件；两者冲突时以本文件为准并告诉 A。
4. **读 `CHANGELOG.md` 最近的记录**，看交接文件末尾的状态表，了解做到哪里、接口有没有变。
5. 用中文向操作者说明本次打算做什么、会改哪些文件、预计产出什么，**得到确认后再动手**。

## 1. 项目概要

- **课程**：Northwestern SE_POL 408，Fall 2026，五人小组期末项目，按 HOML 第 2 章流程完成，通过 GitHub 提交。评分看分析流程和概念是否正确；汇报时每人都要能解释任何一段代码。
- **问题**：纽约市提审法官设保释或收押的决定，有多少能由案件和前科预测？加入法院、法官后多预测多少？模型误差在不同种族之间是否一致？
- **决策者与时点**：提审法官，被捕后约 24 小时内的提审。**特征只能用提审当下已有的信息。**
- **数据**：NY UCS Pretrial Release Data，一行一个匿名“被告—案卷”，纽约市五县，2021–2024 提审年。
- **任务**：二分类，`y_detain` = 设保释或收押（1）vs 无条件释放 / 非金钱条件释放（0）。**它不是“实际被羁押”。**
- **三组特征**：`CASE` ⊂ `CASE_COURT` ⊂ `CASE_COURT_JUDGE`（见 `src/config.py`）。性别和种族一样只用于审计；补充特征组 `CASE_PLUS_GENDER`（CASE + 性别）只用来看性别能多解释多少，不参加嵌套比较。
- **模型**：Logistic、LASSO Logistic、决策树、随机森林、XGBoost；规则基准。
- **划分**：2021–2022 训练（交叉验证调参）；2023 验证（选模型、定阈值）；2024 测试（11/16 冻结后只跑一次）。
- **解释与公平性**：SHAP；按种族的校准、FPR / FNR；法院 / 法官残差（匿名）。
- **不做**：因果推断、评价保释改革、法官实名排名、全州分析。Q3（未到庭）和聚类等主结果跑通后再定。
- **节点**：11/22 数据分析 deck · 12/2–3 汇报 · 12/4 提交仓库、slides、AI Use Statement。

## 2. 分工与文件归属

**每个文件只有一个负责人。AI 只能创建或修改操作者所属角色的文件。** 需要别人的文件改动时，AI 帮操作者写一段 Issue 或群消息，由负责人去改。

| 角色 | 负责 | 只属于该角色的文件 | 不做 |
|---|---|---|---|
| **A 数据与特征**（Yiling Long：样本、特征、仓库；Xinhui Zong：代理变量说明、deck 页，见交接文件） | 下载与版本记录、样本筛选、标签、匿名 `judge_id`、三组特征、泄露检查、年份划分、样本流程表、Table 1、仓库管理、`AI_Use_Statement.md` 汇总 | `src/config.py`、`src/sample.py`、`src/features.py`、`src/build_dataset.py`、`notebooks/01_build_sample.ipynb`、`notebooks/02_features.ipynb`、`tests/test_features.py`、`data/`（含原始数据）、`docs/`（含 `docs/handoff/`，各人只改自己交接文件的状态表）、`scripts/`、`README.md`、`AGENTS.md`、`CLAUDE.md`、`.gitignore`、`requirements.txt` | 不训练模型 |
| **B 基础模型**（Shuyi Niu，建议） | 训练框架 `fit_and_predict`、预处理管线；Logistic、LASSO、决策树；规则基准 | `src/models.py`、`src/rules.py`、`notebooks/05_models_baseline.ipynb` | 不改特征和指标函数 |
| **C 地点与公平**（Yixuan Wen，建议） | 法院 / 法官描述；公平性分析；法院 / 法官残差图 | `src/fairness.py`、`notebooks/03_court_judge.ipynb`、`notebooks/08_fairness.ipynb` | 不训练模型，不公开法官姓名 |
| **D 评估**（Shuyi Niu，建议） | 指标函数、模型比较总表、信息增益与 bootstrap 区间、2024 最终测试；Q3（若做） | `src/evaluate.py`、`notebooks/07_compare_test.ipynb`、`notebooks/10_q3_fta.ipynb` | 不改训练框架 |
| **E 复杂模型与解释**（Yanqi Wu，建议） | EDA 发现页；RF、XGBoost；SHAP；聚类（若做）；deck 整合 | `src/models_ensemble.py`、`src/explain.py`、`notebooks/04_eda.ipynb`、`notebooks/06_models_ensemble.ipynb`、`notebooks/09_shap.ipynb`、`notebooks/11_clustering.ipynb` | 不另写训练流程 |

每个人在什么时候交什么，见 `docs/分工与时间线.md`。

容易重叠的地方：

- **训练框架**只在 `src/models.py`（B）写一次。E 在 `src/models_ensemble.py` 只定义 RF / XGBoost 和参数网格，然后**调用** `fit_and_predict`。
- **指标函数**只在 `src/evaluate.py`（D）写一次；B、C、E 都调用，不重写。
- **测试集**只有 D 在冻结后运行一次；其他人只看 `train_oof` 和 `valid`。
- **描述统计**：A 只做样本流程表和 Table 1；deck 的其他描述图由 E 画；法院 / 法官图由 C 画。
- **共享文件**：`CHANGELOG.md` 每人只改自己的小节；`AI_Use_Statement.md` 每人只追加行；`requirements.txt` 要加包时在 PR 里说明，由 A 合并。

## 3. 统一约定

### 3.1 代码风格

- Python 3.11；包以 `requirements.txt` 为准。
- 变量、列、函数一律 `snake_case` 英文；注释、docstring、notebook 里的 markdown 说明用英文（是提交物）。
- **路径、年份、阈值、随机种子只从 `src/config.py` 读取**，不在别处写死。随机种子 `RANDOM_STATE = 408`。
- 中间文件用 parquet，存到 `data/processed/`（不上传）。
- 可复用代码写进 `src/` 的函数；notebook 只负责调用、展示和说明。notebook 开头：
  ```python
  %load_ext autoreload
  %autoreload 2
  import sys; sys.path.append("..")
  from src import config
  ```

### 3.2 列名

`src/sample.py` 读入时把所有原始列名转成 `snake_case`：小写，空格和连字符换成 `_`。例：`Release Decision at Arraign` → `release_decision_at_arraign`，`Obey_Court_Conditions-Family_Offense` → `obey_court_conditions_family_offense`。**之后所有代码只用 snake_case 名字。**

### 3.3 数据接口（列名一经发布不得随意改）

**① `data/processed/analytic.parquet` — A 生成，一行一个案卷**

包含 S3 之后（纽约市、去掉当庭结案和 Unknown）的所有行；主样本用 `main_sample` 筛选。

| 列 | 类型 | 说明 |
|---|---|---|
| `case_id` | str | 主键，= `internal_case_id`（十六进制字符串，如 `0xF7…`，不是数字） |
| `arr_cycle_id` | float | 逮捕周期，可缺失 |
| `arraign_year` | int | 提审年 |
| `split` | str | `train` / `valid` / `test` |
| `y_detain` | int 0/1 | 主标签 |
| `nominal_bail` | int 0/1 | 设保释且首次保释金额都 ≤ $1 |
| `bail_posted_at_arraign` | int 0/1 | 提审当天交保 |
| `released` | int 0/1 | ROR、NMR 或提审当天交保 |
| `y_warrant`、`y_rearrest` | int 0/1 | Q3 用 |
| `excl_case_type`、`dup_cycle` | int 0/1 | 样本筛选标记 |
| `main_sample` | bool | 通过所有主样本规则 |
| `judge_id` | str | 匿名法官编号（`J001` 起），按 2021–22 编码；低频为 `other`，之后新出现为 `unseen`。**选模型（2023 验证）用这一列** |
| `judge_id_final` | str | 同上，但按 2021–2023 编码。**冻结后重新拟合、预测 2024 时用这一列代替 `judge_id`** |
| `dcjs_missing` | int 0/1 | 前科字段缺失（没匹配到 DCJS，不等于无前科） |
| `age_fixed` | int 0/1 | 原始 `age_at_arrest` = 0（占位值），已用 `age_at_crime` 替换（或设为缺失） |
| `age_group` | str | 公平性分组 |
| `race`、`ethnicity`、`gender` | str | **只用于公平性分析**（`gender` 另外只允许出现在补充特征组 `CASE_PLUS_GENDER`） |
| 其余原始列 | — | snake_case 原名；`judge_name` 已删除 |

另输出 `reports/sample_flow.csv`（只含计数，可上传）：每一步筛选后的行数，按年份分列。

**② 预测结果 — B 生成 `predictions_baseline.parquet`，E 生成 `predictions_ensemble.parquet`，列完全相同**

| 列 | 说明 |
|---|---|
| `case_id`、`arraign_year` | |
| `model` | `logit` / `lasso` / `tree` / `rf` / `xgb` / `rule` |
| `feature_set` | `CASE` / `CASE_COURT` / `CASE_COURT_JUDGE` |
| `target` | 默认 `y_detain` |
| `split` | `train_oof`（训练集交叉验证的样本外预测）/ `valid` / `test` |
| `y_true`、`y_prob` | |

训练好的模型由 `fit_and_predict` 保存为 `models/{target}_{feature_set}_{model}.joblib`（不上传）。

### 3.4 防泄露规则

`config.LEAKAGE_BLOCKLIST` 和 `config.FAIRNESS_GROUPS` 中的列**不得**进入主特征集（`gender` 例外地只能出现在 `CASE_PLUS_GENDER`）：释放决定及其组成部分、保释金额、交保信息、检方请求、辩护人类型（记录的是最近一次律师，含提审后信息）、出庭次数、逮捕令、处置、判决、羁押天数、再逮捕、种族、族裔、法官姓名、所有标签和样本标记。`features.check_leakage()` 由 B 在训练前调用。**AUC 异常高（> 0.95）先怀疑泄露。**

### 3.5 函数接口

```python
# src/sample.py (A)
def load_raw(years=None) -> pd.DataFrame            # 读 data/raw/，列名转 snake_case，加 arraign_year
def build_sample(raw) -> tuple[pd.DataFrame, pd.DataFrame]   # (analytic 行, sample_flow)

# src/features.py (A)
FEATURE_SETS, CATEGORICAL, NUMERIC                   # 从 config 派生
def check_leakage(cols: list[str]) -> None          # 发现违规列就 raise
def get_X(df, feature_set: str) -> pd.DataFrame

# src/build_dataset.py (A)
def main() -> None                                  # python -m src.build_dataset

# src/models.py (B)
def split_by_year(df) -> dict[str, pd.DataFrame]    # 只用 main_sample；返回 train / valid / test
def make_preprocessor(feature_set: str)
def fit_and_predict(df, model_name, estimator, param_grid, feature_set,
                    target=config.TARGET_Q1, run_test=False, scale=True) -> pd.DataFrame
BASELINE_MODELS = {"logit": (...), "lasso": (...), "tree": (...)}

# src/models_ensemble.py (E)
ENSEMBLE_MODELS = {"rf": (...), "xgb": (...)}

# src/evaluate.py (D)
def metrics(y_true, y_prob) -> dict                 # roc_auc, avg_precision, brier
def bootstrap_ci(y_true, y_prob, metric, n=config.N_BOOTSTRAP) -> tuple
def calibration_table(y_true, y_prob, bins=10) -> pd.DataFrame
def load_predictions() -> pd.DataFrame
def comparison_table(pred_df, split="valid") -> pd.DataFrame

# src/fairness.py (C)
def group_metrics(pred_df, df, group_col, threshold) -> pd.DataFrame   # n、标签比例、校准、FPR、FNR

# src/explain.py (E)
def shap_summary(feature_set, model_name, df) -> ...
```

上游没交付时，先用**随机生成的同结构小数据**写好自己的函数并测试，接口就绪后直接换。

### 3.6 修改接口的流程

列名、函数签名、`config` 常量、特征集一旦发布，修改须：(1) 开 Issue，标签 `interface`；(2) 群里通知受影响的人；(3) A 更新本文件第 3 节并在 `CHANGELOG.md` 记录。

## 4. 数据使用规则

- **原始数据在仓库里**：数据是公开的，为方便全组，`data/raw/NYS for Web <year>.csv.gz`（gzip 压缩，每个 23–31 MB）直接提交，pandas 可直接读。未压缩的 `.csv`（160–210 MB，超过 GitHub 100 MB 上限）不提交。只有 A 能更新原始数据；更新时同步 `data/README.md` 里的下载日期和 sha256。
- **处理后的数据和模型不上传**：`data/processed/`、`models/` 被忽略，每人用 `python -m src.build_dataset` 本地生成。不要用 `git add -f`。
- 全组用**同一版**原始文件：官方每半年更新，旧案信息会变。
- 数据虽然公开且匿名，**输出里不要出现单个案卷记录**（如 `df.head()`）；**不要出现法官姓名**；样本量小于 `MIN_GROUP_N` 的分组不报告。
- **不要凭记忆写类别值。** 用到 `release_decision_at_arraign`、`case_type` 等字段的取值前，先打印 `value_counts()` 或查 `docs/` 里的字典，向操作者展示后再写代码。

## 5. Git 与提交流程

1. 每个任务一个分支，命名 `角色字母/任务`，如 `a/sample-v1`、`b/baseline-models`。不直接改 `main`。
2. 提交由**操作者在 GitHub Desktop 里完成**。AI 负责给出 Commit Summary（英文祈使句，≤ 60 字符）和 Description（英文，2–4 行）。
3. Push 后开 Pull Request，找 reviewer 审：Yiling ↔ Xinhui 互审；Shuyi → Yanqi；Yanqi → Shuyi；Yixuan → Yiling。审过后合并。
4. 合并前确认：干净环境能运行 `python -m src.build_dataset` 和自己的 notebook；没有数据文件；`CHANGELOG.md` 已更新。

## 6. 记录要求（每个 PR）

1. **`CHANGELOG.md`**：在自己角色小节最上方加一行：`- YYYY-MM-DD · 姓名 · 做了什么 · 涉及文件 · PR #编号`；接口变动行首加 `[接口]`。
2. **`AI_Use_Statement.md`**：使用了 AI 就追加一行（日期、姓名、工具、做了什么、怎么核对的、改掉或否掉了什么）。课程要求每份提交写明至少一处需要修正或拒绝的 AI 输出。
3. AI 在会话结束前把这两行写好给操作者确认。

## 7. AI 向操作者解释的方式

- **全程用中文解释**；代码、变量名、命令保持英文。
- 每完成一步，按这个格式说明：**做了什么** / **为什么这样做** / **怎么看结果** / **需要你确认**。
- 需要取舍时（阈值、编码、参数）列出选项和利弊，**让操作者决定**。
- 不确定的事直接说不确定，并说去哪里核对（`docs/` 里的字典和说明、课程文件、找 A 或相应负责人）。法律和文献事实在引用前必须查到原文。
- 会话结束时总结：完成了什么、改了哪些文件、下一步、要通知谁。

## 8. AI 不能做的事

- 改不属于操作者角色的文件，或替其他角色完成任务。
- 不走 3.6 流程就修改已发布的列名、函数签名、`config` 常量。
- 上传数据；用 `git add -f` 绕过 `.gitignore`；在输出中展示单个案卷或法官姓名。
- 编造类别值、数据结果、法律条款或文献。
- 在冻结前运行 2024 测试集；直接 push 到 `main`；删除别人的分支和文件。
- 把 `LEAKAGE_BLOCKLIST` 中的变量放进特征集。

## 9. 当前进度（A 维护，每次交付后更新）

**2026-11-07**

- 已完成：换题；方案 v2（`docs/项目方案_v2.md`）；`src/config.py` 按 v2 更新（snake_case 列名、三组特征、泄露黑名单、样本规则常量）；仓库建立。
- 已下载并提交（gzip）：2021–2025 原始文件；字典、数据说明、处置层级（11/7 重新下载的字典与 11/6 版完全相同）。
- 每人的任务、规格、验收标准和状态：`docs/handoff/`。
- **11/7 已完成 A1-01 至 A1-04**：2021–2025 原始数据已入库（2025 暂不使用）；`python -m src.build_dataset` 生成 `analytic.parquet`（502,989 行，主样本 449,474：训练 193,196 / 验证 120,457 / 测试 135,821）和 `reports/sample_flow.csv`。各人可以开始 EDA。
- **11/8 已合并**：BD-01~04（Shuyi：`models.py`、`evaluate.py`、`rules.py`、notebook 05）、C-01（Yixuan：notebook 03、`fairness.py` 中的描述函数）、A2-05 初稿（Xinhui：notebook 02 第 6–7 节）。
- **整体复现**：在仓库根目录、`pol408` 环境下运行 `bash scripts/run_all.sh`（构建数据 → 全部测试 → 按编号重跑全部 notebook）。
- **11/9 已合并**：E-01 EDA（Yanqi：notebook 04、`reports/e01_eda_report.md`）。第一阶段（EDA）全部完成。
- **11/13 已合并**：BD-05（Shuyi：logit、LASSO、决策树 × 三组特征及性别补充组，`reports/bd05_*`）。验证集 AUC 0.920–0.924，规则基准 0.847。notebook 05 现在重训 12 个模型，`run_all.sh` 全程约 45 分钟。
- **11/13 A1**：敏感性检查完成（`reports/a106_sensitivity_report.md`，结论不变）；法律背景与文献核实（`docs/背景与文献核实.md`）。
- 下一步：E-02 11/14；BD-06 选模型、C-02 11/15；11/16 冻结并跑 2024 测试；11/19 deck 各页交 Yanqi；11/22 提交。
- 其他角色现在可以做：用假数据先写好自己的函数（接口见 3.3、3.5）。
