# A2 特征（Xinhui Zong）— 交接文件

> 给 AI：先读 `AGENTS.md` 和 `docs/handoff/00_共同须知.md`。你的操作者是 Xinhui，角色 A（数据与特征）中负责特征的部分。
> 你只能改：`src/features.py`、`notebooks/02_features.ipynb`、本文件末尾的状态表。需要改 `src/config.py`（特征清单、泄露黑名单、阈值）时，写清楚改什么、为什么，交给 Yiling。

## 你需要知道的

- 输入：`data/processed/analytic.parquet`（Yiling 11/7 晚交付；之前用假数据写函数）。只用 `main_sample == True` 的行建模。
- 特征清单在 `config.FEATURE_SETS`，你负责把它变成可直接给模型用的接口，并检查数据问题。
- 所有探索**只用训练年（2021–22）**；验证、测试年只看分布是否相似。

## 任务

### A2-01 `src/features.py`（11/8）
```python
FEATURE_SETS = config.FEATURE_SETS
NUMERIC = ["prior_vfo_cnt", "prior_nonvfo_cnt", "prior_misd_cnt", "pend_vfo", "pend_nonvfo",
           "pend_misd", "supervision", "dcjs_missing", "age_at_arrest"]
CATEGORICAL = [c for c in all features if c not in NUMERIC]
def check_leakage(cols: list[str]) -> None
    # 任一列在 config.LEAKAGE_BLOCKLIST 中、或以 "y_" 开头、或在 config.FAIRNESS_GROUPS 中 → raise ValueError 并列出
def get_X(df, feature_set: str) -> pd.DataFrame
    # 返回该组特征列（先 check_leakage）；类别列缺失填 "Missing"、统一转 str；
    # 二元 Y/N 列保留为类别；数值列保留 NaN（填补在 B 的预处理管线里，只在训练集拟合）
```
验收：`check_leakage(config.FEATURE_SETS["CASE_COURT_JUDGE"])` 不报错；`check_leakage(["race"])`、`check_leakage(["representation_type"])`、`check_leakage(["y_detain"])` 都报错；`get_X` 输出无 object 类型的 NaN。写一个 `tests/test_features.py`（pytest）覆盖这些。

### A2-02 Table 1（11/8）
`notebooks/02_features.ipynb` 第 1 部分：主样本按 `split` 分列，每个特征的分布（类别：比例；数值：均值、中位数、缺失率）和 `y_detain` 比例。另存 `reports/table1.csv`（只含汇总数）。

### A2-03 缺失与异常检查（11/8）
- `dcjs_missing`、`race == "Unknown"`、`arrest_type` 缺失、`top_severity_at_arraign` 缺失：按县和年份的比例；缺失组的 `y_detain` 是否明显不同
- `age_at_arrest` 异常值（< 16 或 > 90）的数量；建议处理方式交给 Yiling 定
- `top_charge_weight_at_arraign` 为 `V`、`I` 或缺失的行数

### A2-04 字典核对（11/8，等 Yiling 下载新版字典和 2024 文件后）
- 新旧字典逐字段对比：新增 / 删除字段、类别值变化（特别是 Release Decision 是否有 “Nominal bail-set”）、定义变化
- 2024 年文件列名、类别值是否与 2021–23 一致
- 结果写成表放进 notebook 02，并在群里告诉 Yiling 哪些需要改样本规则

### A2-05 代理变量说明（11/12–13）
为老师的问题“哪个输入是代理变量”准备一页：前科数（受各社区执法强度影响）、逮捕类型（DAT 与拘捕取决于警方）、年龄、性别各可能代理什么；每条要有数据上的佐证（例如前科数按种族的分布），文献主张需找到原文。

### A2-06 deck（11/16–19）
特征与泄露规则一页（三组特征是什么、哪些列被排除、为什么），代理变量一页。

## 状态（只由 Xinhui 更新）

| 任务 | 截止 | 状态 | 日期 | PR |
|---|---|---|---|---|
| A2-01 features.py + 测试 | 11/8 | todo | | |
| A2-02 Table 1 | 11/8 | todo | | |
| A2-03 缺失与异常 | 11/8 | todo | | |
| A2-04 字典核对 | 11/8 | todo | | |
| A2-05 代理变量 | 11/13 | todo | | |
| A2-06 deck 页 | 11/19 | todo | | |
