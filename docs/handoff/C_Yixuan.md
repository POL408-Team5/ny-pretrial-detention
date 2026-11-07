# C 法院 / 法官与公平性（Yixuan Wen）— 交接文件

> 给 AI：先读 `AGENTS.md` 和 `docs/handoff/00_共同须知.md`。你的操作者是 Yixuan，角色 C。
> 你只能改：`src/fairness.py`、`notebooks/03_court_judge.ipynb`、`notebooks/08_fairness.ipynb`、`tests/test_fairness.py`、本文件末尾的状态表。

## 你需要知道的

- 法官只用 `judge_id`（`J001`…、`other`、`unseen`），由 Yiling 在 `sample.py` 里生成。**任何输出、图、slides 都不能出现法官姓名**；少于 `config.MIN_GROUP_N` 的格子不报告。
- 我们只描述差异，不说“法官造成了差异”或“存在歧视”：法官分配不是随机的（轮值），差异可能反映案件分配。
- 种族、族裔只用于分组评估，不进模型。
- 探索只用训练年（2021–22）；2023 / 2024 只看新法官占比这类总量。

## 任务

### C-01 `notebooks/03_court_judge.ipynb`（11/8）
1. 五县 `y_detain`：总体，以及在 `top_severity_at_arraign`（重罪 / 轻罪）内
2. 法官案件量分布（训练年主样本）：多少法官 ≥ 100 案、≥ 200 案，覆盖多少比例的案件；对 `config.MIN_JUDGE_CASES` 给建议
3. 法官间羁押率差异：案件量 ≥ 门槛的法官，P10 / P50 / P90；**在重罪和轻罪内分别算**；再做一个粗略的“案件结构调整”：每位法官的实际率 − 按其案件的指控权重 × 暴力标记分格的全体平均率（间接标准化）
4. 2023、2024 年由 `unseen` 法官审理的案件占比（A1 已算出 2023 年约 30%，按月从 2 月起上升；请在法官层面进一步描述：新法官多少人、各自案件量、羁押率是否与老法官不同）
5. 产出 1–2 张可放进 deck 的图（如法官羁押率的毛毛虫图，按 `judge_id`），与 Yanqi 商量谁的图做 EDA 发现页

### C-02 `src/fairness.py`（11/11–13）
```python
def choose_threshold(y_prob_valid, y_true_valid) -> float
    # 默认规则：使 2023 年预测为 1 的比例 = 2023 年实际 y_detain 比例 [定]
def group_metrics(pred_df, df, group_col, threshold) -> pd.DataFrame
    # pred_df 与 df 按 case_id 合并；每组：n、实际比例、平均预测概率、校准差（平均预测 − 实际）、AUC、
    # 固定阈值下的 FPR、FNR、预测为 1 的比例；n < MIN_GROUP_N 的组不输出
def residuals(pred_df, df, by: str) -> pd.DataFrame
    # 按 by（county_name 或 judge_id）汇总：实际比例 − 平均预测概率（用 CASE 组模型），附 bootstrap 区间
```
指标函数调用 Shuyi 的 `src/evaluate.py`，不重写。写 `tests/test_fairness.py`（小样本手算核对 FPR / FNR）。

### C-03 公平性分析（验证集 11/14；测试集 11/16–18）
`notebooks/08_fairness.ipynb`：按 `race`、`ethnicity`、`gender`、`age_group` 的 `group_metrics`；分组校准图；`CASE` 与 `CASE_COURT_JUDGE` 两个模型都做，看加入法官后组间差距是否变化。测试集结果要等 Shuyi 11/15 跑完。

### C-04 残差（11/16–18）
县和法官的残差图（`CASE` 组最终模型）：哪些法官比“同样案件的平均预测”系统性偏高或偏低，附区间；只显示 `judge_id`。

### C-05 deck（11/19）
公平性一页（分组表 + 校准图）、残差一页；措辞只描述差异。

## 状态（只由 Yixuan 更新）

| 任务 | 截止 | 状态 | 日期 | PR |
|---|---|---|---|---|
| C-01 notebook 03 | 11/8 | todo | | |
| C-02 fairness.py + 测试 | 11/13 | todo | | |
| C-03 公平性分析 | 11/18 | todo | | |
| C-04 残差 | 11/18 | todo | | |
| C-05 deck 页 | 11/19 | todo | | |
