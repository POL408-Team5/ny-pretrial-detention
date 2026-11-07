# 更新记录（CHANGELOG）

每个 PR 合并前，在**自己角色的小节最上方**加一行（最新的在上）：

`- YYYY-MM-DD · 姓名 · 做了什么（一句话） · 涉及文件 · PR #编号`

接口变动（列名、函数签名、config 常量、特征集）在行首加 `[接口]`。每人只改自己的小节，避免合并冲突。

## 里程碑（A 维护）

| 版本 | 日期 | 内容 |
|---|---|---|
| v0.1 | 11/7 | 换题；方案 v2；仓库结构、分工、AI 工作说明 |
| v0.2 | 11/10 | `analytic.parquet` 可生成，全组本地跑通 |
| v0.3 | 11/15 | 模型冻结；2024 测试 |
| v0.4 | 11/22 | 数据分析 deck |
| v1.0 | 12/4 | 期末提交 |

## A 数据与特征（Yiling Long、Xinhui Zong）

- 2026-11-07 · Yiling Long · [接口] 完成 A2-01 至 A2-04（代 Xinhui）：`src/features.py`（`get_X`、`check_leakage`、`NUMERIC`/`CATEGORICAL`）与测试；notebook 02（Table 1、缺失与异常、跨年一致性）；年龄 0 修正（新列 `age_fixed`） · `src/features.py`、`tests/`、`notebooks/02_features.ipynb`、`reports/table1.csv`、`src/sample.py` · PR #—
- 2026-11-07 · Yiling Long · [接口] 加入 2024（测试）和 2025（暂不用）原始数据；新增 `judge_id_final`（按 2021–23 编码，用于最终测试）；逮捕周期去重改为先保留较早年份；读入时先筛纽约市以降低内存；核实官方字典无更新 · `src/sample.py`、`src/config.py`、`notebooks/01_build_sample.ipynb`、`data/`、`docs/` · PR #—
- 2026-11-07 · Yiling Long · 完成样本与数据构建：`python -m src.build_dataset` 生成 analytic.parquet（351,996 行，主样本 313,653）和样本流程表；notebook 01 记录名义保释、逮捕周期、法官编码（2023 年 30% 为新法官）与缺失检查 · `src/sample.py`、`src/build_dataset.py`、`notebooks/01_build_sample.ipynb`、`reports/sample_flow.csv`、`data/README.md`、`docs/handoff/` · PR #—
- 2026-11-07 · Yiling Long · [接口] 建立新仓库：方案 v2、AGENTS.md、README、data/README；config 改为 snake_case 列名，定义三组特征、样本规则常量和泄露黑名单 · `docs/项目方案_v2.md`、`AGENTS.md`、`README.md`、`data/README.md`、`src/config.py`、`.gitignore`、`requirements.txt` · PR #—

## B 基础模型

## C 地点与公平

## D 评估

## E 复杂模型与解释
