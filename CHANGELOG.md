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

- 2026-11-07 · Yiling Long · [接口] 建立新仓库：方案 v2、AGENTS.md、README、data/README；config 改为 snake_case 列名，定义三组特征、样本规则常量和泄露黑名单 · `docs/项目方案_v2.md`、`AGENTS.md`、`README.md`、`data/README.md`、`src/config.py`、`.gitignore`、`requirements.txt` · PR #—

## B 基础模型

## C 地点与公平

## D 评估

## E 复杂模型与解释
