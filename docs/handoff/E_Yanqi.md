# E 复杂模型、SHAP、EDA 与 deck 整合（Yanqi Wu）— 交接文件

> 给 AI：先读 `AGENTS.md` 和 `docs/handoff/00_共同须知.md`。你的操作者是 Yanqi，角色 E。
> 你只能改：`src/models_ensemble.py`、`src/explain.py`、`notebooks/04_eda.ipynb`、`notebooks/06_models_ensemble.ipynb`、`notebooks/09_shap.ipynb`、`notebooks/11_clustering.ipynb`、`slides/`、本文件末尾的状态表。

## 你需要知道的

- 训练流程只在 Shuyi 的 `src/models.py` 里：你只定义 RF / XGBoost 和参数网格，**调用** `fit_and_predict`，不另写训练循环。框架要改找 Shuyi。
- 老师对旧项目 EDA 的批评是“没有展示与关键变量有关的有意思的发现”。这次 EDA **必须有一个能指向行动的发现**。
- 探索只用训练年（2021–22）。种族只做描述，不下“歧视”结论。

## 任务

### E-01 `notebooks/04_eda.ipynb`（11/8）
1. `y_detain` 按：年份、`top_severity_at_arraign`、`top_charge_weight_at_arraign`、暴力重罪、`arraign_charge_category`（前 12 类）、前科三项、待决、监管、`dat_wo_ws_prior_to_arraign`、`age_group`、`gender`、`arrest_type`
2. 按 `race` 的原始羁押率，以及在“重罪 / 轻罪 × 有无暴力前科”分层内的比率（只描述）
3. 补充：主样本中设保释（不含名义保释）后从未交上的比例（`bail_made_indicator` 缺失）——注意这一列在泄露黑名单中，只能做描述，不能做特征
4. **一个指向行动的发现**：候选“同样的指控和前科，不同法官的羁押率差多少”（与 Yixuan 的 C-01 分工：她算法官分布，你做成发现页的叙述和图），或“指控与前科能解释多少（与 Shuyi 的冒烟测试 AUC 呼应）”
5. 统一图的风格（字体、配色、来源注 “Source: NY UCS Pretrial Release Data, 2021–22 NYC arraignments”），后面 deck 直接用；图存 `reports/figures/`

### E-02 `src/models_ensemble.py`（11/14 周六前）
```python
ENSEMBLE_MODELS = {
    "rf":  (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=config.RANDOM_STATE),
            {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [20, 100]}),
    "xgb": (XGBClassifier(n_estimators=400, tree_method="hist", eval_metric="logloss", random_state=config.RANDOM_STATE),
            {"clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8]}),
}
```
`notebooks/06_models_ensemble.ipynb`：两个模型 × 三组特征调用 `fit_and_predict(..., scale=False)`，输出 `data/processed/predictions_ensemble.parquet`（列与 baseline 完全相同）。太慢时先在子样本上调参并写明。XGBoost 用 early stopping 定树的数量时，早停用的验证片**从训练年（2021–22）里切**，不能用 2023 验证集（课件 Week 6）。

### E-03 SHAP 与 permutation importance（11/16–18）
`src/explain.py`：
```python
def shap_summary(feature_set, model_name, df, n_sample=5000) -> pd.DataFrame
    # 读 models/y_detain_{feature_set}_{model_name}.joblib；在验证集抽样上算 TreeExplainer；
    # one-hot 后的列按原始特征汇总平均 |SHAP|
def permutation_importance_table(feature_set, model_name, df) -> pd.DataFrame
    # sklearn.inspection.permutation_importance，scoring="roc_auc"，按原始特征置换；
    # 冻结前在 2023 验证集上试跑，报告用的结果在 11/16 冻结后用 2024 测试集算（课件 Week 6）
```
`notebooks/09_shap.ipynb`：最终模型的 SHAP 汇总图、前三个特征的依赖图、permutation importance 对比 impurity importance（为老师的问题准备）。准备回答“模型最依赖的三个输入”（是预测上的依赖，不是因果）。另外对 `CASE_PLUS_GENDER` 的 XGBoost 做一次 SHAP，报告性别在 SHAP 里的排名（全组据此决定性别最终是否进模型）。

### E-04 deck 整合（11/19–21）
收各人页面（11/19），统一风格，按 `docs/项目方案_v2.md` 第 7 节结构排；11/20–21 全组过一遍。**11/22 提交。**

### E-05 聚类（11/23 之后，视进度）
在 `CASE` 特征上对案件聚类（不含决定和结果），报告每类的羁押率、县间和法官间差异。

## 状态（只由 Yanqi 更新）

| 任务 | 截止 | 状态 | 日期 | PR |
|---|---|---|---|---|
| E-01 EDA 与发现页 | 11/8 | done（11/9 合并） | 2026-11-08 | #5 |
| E-02 RF / XGBoost × 三组 | 11/14 | todo | | |
| E-03 SHAP 与 permutation | 11/18 | todo | | |
| E-04 deck 整合 | 11/21 | todo | | |
| E-05 聚类 | 视进度 | todo | | |
