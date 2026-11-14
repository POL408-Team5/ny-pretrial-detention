# A1 数据（Yiling Long）— 交接文件

> 给 AI：先读 `AGENTS.md` 和 `docs/handoff/00_共同须知.md`。你的操作者是 Yiling，角色 A（数据），同时是仓库管理者。
> 你只能改 A 的文件：`src/config.py`、`src/sample.py`、`src/build_dataset.py`、`notebooks/01_build_sample.ipynb`、`data/`、`README.md`、`AGENTS.md`、`CLAUDE.md`、`.gitignore`、`requirements.txt`、`docs/handoff/`、`docs/项目方案_*.md`、`docs/分工与时间线.md`。`src/features.py` 和 notebook 02 归 Xinhui。

## 你交给下游的东西

`analytic.parquet` 是所有人的输入。**周六（11/7）晚必须能用 `python -m src.build_dataset` 生成**，列定义见 `AGENTS.md` 3.3。

## 任务

### A1-01 `src/sample.py`（11/7）
```python
def load_raw(years=None) -> pd.DataFrame
    # 读 data/raw/NYS for Web {year}.csv.gz（不存在时退回 .csv；都没有就 warning 并跳过该年）
    # encoding="utf-8-sig", low_memory=False；列名转 snake_case；加 arraign_year（由 first_arraign_date 解析）
    # 检查各年列名一致，不一致就 raise 并列出差异
def build_sample(raw) -> tuple[pd.DataFrame, pd.DataFrame]   # (analytic 行, sample_flow)
```
步骤（每步记录行数，按年份分列）：
1. S1 纽约市五县；S2 去 `Disposed at arraign`；S3 去 `Unknown` → 之后的行全部保留在 analytic 中
2. 标签：`y_detain`、`nominal_bail`（`Bail-set` 且 `BAIL_AMOUNT_COLS` 中所有非缺失值 ≤ `NOMINAL_BAIL_MAX`，且至少一个非缺失）、`bail_posted_at_arraign`、`released`、`y_warrant`（`Y`→1）、`y_rearrest`（`rearrest` 非 `No Arrest` 且非缺失→1；缺失保留 NaN）
3. 标记：`excl_case_type`（不在 `CASE_TYPES_MAIN`）；`dcjs_missing`（`prior_vfo_cnt` 缺失）
4. `dup_cycle`：在通过 S4、S5 的行里，按 `arr_cycle_id` 分组，保留 `top_charge_weight_at_arraign` 最重的一行（按 `SEVERITY_ORDER`，平局取 `case_id` 最小），其余标 1；`arr_cycle_id` 缺失的行标 0
5. `main_sample` = 未被 S4、S5（若 `EXCLUDE_NOMINAL_BAIL`）、S6（若 `DEDUP_ARREST_CYCLE`）排除
6. `judge_id`：只用**训练年主样本**统计每位法官的案件数；按案件数降序编号 `J001`…；少于 `MIN_JUDGE_CASES` 的为 `other`；验证 / 测试年里训练年没出现的法官为 `unseen`。之后删除 `judge_name`。对照表**不保存**
7. `age_group`（`AGE_BINS`、`AGE_LABELS`）；`split`（按 `arraign_year`）；`case_id` = `internal_case_id`
8. 字符串列 `strip()`，空字符串转缺失

验收：2022 年 S3 之后 117,294 行；`case_id` 唯一；`y_detain` 在主样本里无缺失；输出中没有 `judge_name`。

### A1-02 `src/build_dataset.py`（11/7）
`main()`：`load_raw` → `build_sample` → 写 `config.ANALYTIC_FILE` 和 `config.SAMPLE_FLOW_FILE`；打印各年主样本行数和 `y_detain` 比例。

### A1-03 `notebooks/01_build_sample.ipynb`（11/8）
样本流程表；各年 `y_detain`（主样本 / 含名义保释）；名义保释比例；同一 `arr_cycle_id` 跨年份的数量（跨训练 / 验证 / 测试的要报告并决定处理）；法官编码统计（编号法官数、`other`、`unseen` 案件占比）。

### A1-04 数据版本（11/7–8）
下载 2024 年文件和最新版字典 → gzip 进 `data/raw/` → 更新 `data/README.md`（行数、下载日期、sha256）→ 群里通知。

### A1-05 周一邮件（11/9）
汇总 E、C、B+D 的 EDA 结果，起草给老师的邮件（换题、问题、数据、2–3 个发现、想请教的问题），群里过一遍后发出。

### A1-06 敏感性数据（11/10–14）
用 config 开关生成：名义保释计为 1、不去重两个版本；交给 Shuyi 用 `fit_and_predict` 跑同一个最终模型。

11/13 完成：`python -m src.build_dataset --sensitivity` 生成两个版本；`python -m src.sensitivity [--model 名]` 在 2023 年上比较（模型存 `models/sensitivity/`，不覆盖主模型）。结果见 `reports/a106_sensitivity_report.md`：去重几乎无影响（只多 81 个案卷）；名义保释计为 1 使 AUC 降约 0.04，但三组特征的排序和法官增量不变。

### A1-07 deck（11/16–19）
问题与背景、数据与样本流程、局限；核实法律背景和文献（`docs/项目方案_v2.md` 附录）。

### A1-08 提交（12/4）
汇总 `AI_Use_Statement.md` 的 Summary；干净环境复现。

## 状态（只由 Yiling 更新）

| 任务 | 截止 | 状态 | 日期 | PR |
|---|---|---|---|---|
| A1-01 sample.py | 11/7 | done | 11/7 | |
| A1-02 build_dataset.py | 11/7 | done | 11/7 | |
| A1-03 notebook 01 | 11/8 | done | 11/7 | |
| A1-04 2024 数据与新字典 | 11/8 | done | 11/7 | |
| A1-05 给老师的邮件 | 11/9 | done（改为 summary 附件） | 11/10 | |
| A1-06 敏感性数据 | 11/14 | done（数据 + logit 验证集结果；最终模型冻结后可重跑） | 11/13 | 本地合并 |
| A1-07 deck 页 | 11/19 | 法律与文献已核实（`docs/背景与文献核实.md`）；页面待做 | 11/13 | |
| A1-08 提交 | 12/4 | todo | | |
