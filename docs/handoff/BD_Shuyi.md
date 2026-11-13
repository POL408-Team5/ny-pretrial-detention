# B+D 建模框架、简单模型与评估（Shuyi Niu）— 交接文件

> 给 AI：先读 `AGENTS.md` 和 `docs/handoff/00_共同须知.md`。你的操作者是 Shuyi，角色 B（基础模型）和 D（评估）。
> 你只能改：`src/models.py`、`src/rules.py`、`src/evaluate.py`、`notebooks/05_models_baseline.ipynb`、`notebooks/07_compare_test.ipynb`、`notebooks/10_q3_fta.ipynb`、`tests/test_models.py`、`tests/test_evaluate.py`、本文件末尾的状态表。

## 你需要知道的

- 你写的 `fit_and_predict` 和指标函数是全组共用的：E 调用它训练 RF / XGBoost，C 调用指标函数做公平性。**签名一旦发布不要随便改**（改要走 `AGENTS.md` 3.6）。
- 输入：`analytic.parquet` 中 `main_sample == True` 的行；特征用 Xinhui 的 `features.get_X(df, feature_set)`。
- 2024 测试集**只在 11/16（周一）冻结后由你跑一次**。

## 任务

### BD-01 `src/models.py`（11/8，先用假数据写）
```python
def split_by_year(df) -> dict[str, pd.DataFrame]
    # 只保留 main_sample；按 config.SPLIT_COL 返回 {"train", "valid", "test"}
def make_preprocessor(feature_set: str, scale: bool = True) -> ColumnTransformer
    # 类别：OneHotEncoder(handle_unknown="ignore")；数值：SimpleImputer(strategy="constant", fill_value=0)（dcjs_missing 已标记缺失）+ 可选 StandardScaler
def fit_and_predict(df, model_name, estimator, param_grid, feature_set,
                    target=config.TARGET_Q1, run_test=False, scale=True) -> pd.DataFrame
    # 1) 训练集：Pipeline(preprocessor, estimator)，GridSearchCV(StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE), scoring="roc_auc")
    # 2) 训练集样本外预测：cross_val_predict（最优参数）→ split="train_oof"
    # 3) 最优参数在全训练集上拟合 → 预测 2023 → split="valid"
    # 4) run_test=True 时：最优参数在 2021–2023 上重新拟合 → 预测 2024 → split="test"（[定]，11/15 前确认）
    # 5) 保存模型 models/{target}_{feature_set}_{model_name}.joblib；返回 AGENTS.md 3.3 ② 格式的长表
BASELINE_MODELS = {
    "logit": (LogisticRegression(max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10]}),
    "lasso": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000), {"clf__C": [0.001, 0.01, 0.1, 1]}),
    "tree":  (DecisionTreeClassifier(random_state=config.RANDOM_STATE), {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [50, 200]}),
}
```
训练集约 20 万行；网格先小，跑太慢可以先在 20% 子样本上调参（写明）。

### BD-02 `src/rules.py`（11/8）
`rule_scores(df)`：只用 `top_charge_weight_at_arraign` × `top_charge_at_arraign_violent_felony_ind` 分格，概率 = 训练年该格的 `y_detain` 比例（格子样本 < 100 的并入同等级）。输出与 `fit_and_predict` 同格式，`model="rule"`，`feature_set="CASE"`。

### BD-03 `src/evaluate.py`（11/8）
```python
def metrics(y_true, y_prob) -> dict              # roc_auc, avg_precision, brier, n, base_rate
def calibration_table(y_true, y_prob, bins=10) -> pd.DataFrame   # 分位数分箱：平均预测、实际比例、n
def bootstrap_ci(y_true, y_prob, metric, n=config.N_BOOTSTRAP, seed=config.RANDOM_STATE) -> tuple   # (估计, 2.5%, 97.5%)
def paired_bootstrap_diff(y_true, p_a, p_b, metric, n=...) -> tuple   # 两组特征之差，同一组重抽样
def load_predictions() -> pd.DataFrame           # 合并 baseline 与 ensemble 文件
def comparison_table(pred_df, split="valid") -> pd.DataFrame   # 行：model；列：feature_set × 指标
```
写 `tests/test_evaluate.py`：完美预测 AUC = 1；随机预测 AUC ≈ 0.5；Brier 与 sklearn 一致。

### BD-04 冒烟测试（11/8 晚）
`notebooks/05_models_baseline.ipynb`：规则基准 + `logit` × `CASE`，再加 `logit` × `CASE_PLUS_GENDER`（补充特征组），报告 `train_oof` 和 `valid` 的 AUC、AP、Brier，以及加入性别后的 AUC 变化。预期 AUC 约 0.8；**> 0.95 先怀疑泄露**，查 `features.check_leakage`。把结果数字发群里，周一邮件会用。

### BD-05 三个简单模型 × 三组特征（11/10–13）
输出 `data/processed/predictions_baseline.parquet`（另加 `CASE_PLUS_GENDER` 作为补充组）；notebook 05 报告验证集结果。

### BD-06 比较与选模型（11/15 周日前）
`notebooks/07_compare_test.ipynb`：合并 E 的预测，验证集比较表（五模型 + 规则 × 三组特征）；`CASE` → `CASE_COURT` → `CASE_COURT_JUDGE` 的 AUC 增量（配对 bootstrap 区间），**同时报告只含已编码法官（`judge_id` 以 J 开头）的案件**——2023 年约 30% 是 `unseen` 法官；校准图。全组据此选最终模型。

不加 SVM：线性 SVM 在 `CASE` 上验证集 AUC 0.918 / AP 0.658，logit 为 0.920 / 0.663，没有更好，而且不直接给概率（阈值和校准都要用概率）。deck 里一句话说明即可。

### BD-07 冻结后测试（11/16，只跑一次）
`run_test=True` 跑选定模型 × 三组特征（重新拟合用 2021–2023，`CASE_COURT_JUDGE` 组用 `config.JUDGE_COL_FINAL` 代替 `judge_id`）；测试集比较表（AUC、AP、Brier）、信息增益图（附区间）、校准图，以及**政策阈值（C-02 在 2023 年定的阈值）下的混淆矩阵**（课件 Week 6 的测试步骤），存 `reports/`。措辞按“预测”写，不写成因果效应。跑完在群里通知，之后不再改模型。

### BD-08 Q3（11/23 之后，视进度）
在 `released == 1` 的人中预测 `y_warrant`；注意观察窗口到处置为止长短不一、未结案件。开工前先和 Yiling 定样本。

### BD-09 deck（11/16–19）
划分与验证设计、模型比较表、信息增益图、校准图；准备回答“给决策者看 impurity 还是 permutation importance、为什么”。

## 状态（只由 Shuyi 更新）

| 任务 | 截止 | 状态 | 日期 | PR |
|---|---|---|---|---|
| BD-01 models.py | 11/8 | done | 11/8 | merged 11/8 |
| BD-02 rules.py | 11/8 | done | 11/8 | merged 11/8 |
| BD-03 evaluate.py + 测试 | 11/8 | done | 11/8 | merged 11/8 |
| BD-04 冒烟测试 | 11/8 | done | 11/8 | merged 11/8 |
| BD-05 简单模型 × 三组 | 11/13 | done（11/13 合并） | 2026-11-12 | 本地合并 |
| BD-06 比较与选模型 | 11/15 | todo | | |
| BD-07 测试 | 11/16 | todo | | |
| BD-08 Q3 | 视进度 | todo | | |
| BD-09 deck 页 | 11/19 | todo | | |
