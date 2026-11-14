# 更新记录（CHANGELOG）

每个 PR 合并前，在**自己角色的小节最上方**加一行（最新的在上）：

`- YYYY-MM-DD · 姓名 · 做了什么（一句话） · 涉及文件 · PR #编号`

接口变动（列名、函数签名、config 常量、特征集）在行首加 `[接口]`。每人只改自己的小节，避免合并冲突。

## 里程碑（A 维护）

| 版本 | 日期 | 内容 |
|---|---|---|
| v0.1 | 11/7 | 换题；方案 v2；仓库结构、分工、AI 工作说明 |
| v0.2 | 11/10 | `analytic.parquet` 可生成，全组本地跑通 |
| v0.3 | 11/16 | 模型冻结；2024 测试 |
| v0.4 | 11/22 | 数据分析 deck |
| v1.0 | 12/4 | 期末提交 |

## A 数据与特征（Yiling Long、Xinhui Zong）

- 2026-11-13 · Yiling Long · [接口] A1-06：`build_sample` 加两个开关参数（默认取 config，主样本不变）；`config.SENSITIVITY_VARIANTS`；`build_dataset --sensitivity`；新增 `src/sensitivity.py` 和验证集结果；A1-07 法律背景与文献核实 · `src/`、`reports/a106_*`、`reports/sample_flow_*`、`docs/背景与文献核实.md`、`README.md` · 本地合并
- 2026-11-13 · Yiling Long · 合并 BD-05；建模截止后移（E-02 11/14，BD-06、C-02 11/15，冻结 11/16）；交接文件补课件 Week 6 要点（XGBoost 早停用训练年切片、permutation importance 用测试集、测试报告政策阈值下的混淆矩阵）和不加 SVM 的说明 · `docs/`、`AGENTS.md` · 本地合并
- 2026-11-09 · Yiling Long · 合并 E-01（PR #5）；按数据字典核实 `bail_made_indicator` 的含义并更新 EDA 报告第 5 节措辞；更新进度与共同须知 · `reports/e01_eda_report.md`、`docs/`、`AGENTS.md` · 本地合并
- 2026-11-08 · Yiling Long · 合并 BD-01~04、C-01、A2-05 初稿；notebook 05 加性别增量的配对 bootstrap 区间；新增 `scripts/run_all.sh` 一键复现；更正 C 交接中与测试年规则冲突的要求；更新进度与共同须知 · `scripts/`、`notebooks/05_models_baseline.ipynb`、`docs/`、`AGENTS.md`、`README.md` · 本地合并
- 2026-11-07 · Xinhui Zong · Started A2-05 with training-only proxy-variable summaries and reviewed feature code · `notebooks/02_features.ipynb` · PR #—

- 2026-11-07 · Yiling Long · [接口] 组内决定：性别移入审计变量（FAIRNESS_GROUPS），新增补充特征组 `CASE_PLUS_GENDER`；`check_leakage` 加 `allow` 参数；测试增至 13 个；待决事项 2–4 已定 · `src/config.py`、`src/features.py`、`tests/`、`notebooks/02_features.ipynb`、`docs/` · PR #—
- 2026-11-07 · Yiling Long · 回应 Xinhui 审阅：测试年不再计算任何结果统计（diagnostics、notebook 01、Table 1）并在 data/README 记录；Table 1 加中位数、计数和小格抑制；notebook 01 改为读已生成文件；前科时点无法从官方文档确认，列为 D14 · `src/sample.py`、`notebooks/01–02`、`reports/table1.csv`、`data/README.md`、`docs/` · PR #—
- 2026-11-07 · Yiling Long · [接口] 完成 A2-01 至 A2-04（代 Xinhui）：`src/features.py`（`get_X`、`check_leakage`、`NUMERIC`/`CATEGORICAL`）与测试；notebook 02（Table 1、缺失与异常、跨年一致性）；年龄 0 修正（新列 `age_fixed`） · `src/features.py`、`tests/`、`notebooks/02_features.ipynb`、`reports/table1.csv`、`src/sample.py` · PR #—
- 2026-11-07 · Yiling Long · [接口] 加入 2024（测试）和 2025（暂不用）原始数据；新增 `judge_id_final`（按 2021–23 编码，用于最终测试）；逮捕周期去重改为先保留较早年份；读入时先筛纽约市以降低内存；核实官方字典无更新 · `src/sample.py`、`src/config.py`、`notebooks/01_build_sample.ipynb`、`data/`、`docs/` · PR #—
- 2026-11-07 · Yiling Long · 完成样本与数据构建：`python -m src.build_dataset` 生成 analytic.parquet（351,996 行，主样本 313,653）和样本流程表；notebook 01 记录名义保释、逮捕周期、法官编码（2023 年 30% 为新法官）与缺失检查 · `src/sample.py`、`src/build_dataset.py`、`notebooks/01_build_sample.ipynb`、`reports/sample_flow.csv`、`data/README.md`、`docs/handoff/` · PR #—
- 2026-11-07 · Yiling Long · [接口] 建立新仓库：方案 v2、AGENTS.md、README、data/README；config 改为 snake_case 列名，定义三组特征、样本规则常量和泄露黑名单 · `docs/项目方案_v2.md`、`AGENTS.md`、`README.md`、`data/README.md`、`src/config.py`、`.gitignore`、`requirements.txt` · PR #—

## B 基础模型

- 2026-11-12 · Shuyi Niu · 完成 BD-05：三个简单模型 × 三组主特征及三组性别补充对照；保存训练 OOF/2023 验证预测、调参和矩阵维度汇总；42 项测试及独立复算通过，未使用 2024 测试 · `src/models.py`、`tests/test_models.py`、`notebooks/05_models_baseline.ipynb`、`reports/bd05_*`、`docs/handoff/BD_Shuyi.md` · 待审核

- 2026-11-08 · Shuyi Niu · 完成 BD-01~04：`fit_and_predict`、预处理、规则基准、指标与 bootstrap 函数及测试；冒烟测试 2023 验证 AUC 0.920（规则 0.846），性别补充组 +0.001 · `src/models.py`、`src/rules.py`、`src/evaluate.py`、`tests/test_models.py`、`tests/test_evaluate.py`、`notebooks/05_models_baseline.ipynb` · 合并于 11/8（Yiling 补记）

## C 地点与公平

- 2026-11-08 · Yixuan Wen · 完成 C-01：训练年五县/法官比例、100/200 案门槛覆盖、分层 P10/P50/P90、指控权重×暴力标记间接标准化；未来年份仅做无结果的法官覆盖与案件量汇总；导出 7 张汇总表和 2 张图，notebook 整本运行、21 项测试（含 8 项专项）及独立复算通过 · `src/fairness.py`、`tests/test_fairness.py`、`notebooks/03_court_judge.ipynb`、`reports/c01_*.csv`、`reports/figures/c01_*.png` · PR #3

## D 评估

## E 复杂模型与解释

- 2026-11-08 · Yanqi Wu · Expanded training-only E-01 EDA with an 18-feature inventory, level distributions, feature relationships, year-to-year drift, small-cell suppression, judge permutation checks, and a review report · `notebooks/04_eda.ipynb`, `reports/e01_*.csv`, `reports/e01_eda_report.md`, `reports/figures/e01_*.png` · PR #5
