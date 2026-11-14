"""RF and XGBoost specifications for the shared development-only workflow (E-02).

Use ``models.fit_and_predict(..., scale=False)``. The agreed grids are tuned
within the training years; the validation year is never an early-stopping set.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config


class TrainOnlyEarlyStoppingXGBClassifier(XGBClassifier):
    """Choose tree count inside each fit, then refit on all rows supplied to it.

    The shared Pipeline fits its preprocessor on the outer training fold. This
    estimator sees only that transformed fold and holds out 20% of its rows for
    log-loss early stopping. Outer CV holdouts and the development validation
    year never reach ``fit``. After selecting a tree count, a new booster fits
    every supplied row, including the internal early-stopping slice.

    Constructor ``n_estimators`` remains the search ceiling after fitting, so
    sklearn clones repeat the same policy rather than inheriting a tree count
    chosen on a different fold. This remains an XGBClassifier for downstream
    TreeExplainer / get_booster consumers. External eval_set is rejected.
    """

    _validation_fraction = 0.2

    def fit(self, X, y, *, eval_set=None, verbose=False):
        if eval_set is not None:
            raise ValueError("Early-stopping data must be split inside the supplied training fold.")
        y = np.asarray(y)
        if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("E-02 early stopping requires two nonmissing binary classes.")
        if self.early_stopping_rounds is None or self.early_stopping_rounds < 1:
            raise ValueError("Set a positive early_stopping_rounds patience.")
        if self.n_estimators is None or self.n_estimators < 1:
            raise ValueError("Set a positive n_estimators search ceiling.")
        fit_rows, stop_rows = train_test_split(
            np.arange(len(y)), test_size=self._validation_fraction,
            stratify=y, random_state=self.random_state,
        )
        def rows(index):
            return X.iloc[index] if hasattr(X, "iloc") else X[index]

        params = self.get_params(deep=False)
        probe = XGBClassifier(**params)
        probe.fit(rows(fit_rows), y[fit_rows],
                  eval_set=[(rows(stop_rows), y[stop_rows])], verbose=False)
        selected = int(probe.best_iteration) + 1
        refit = XGBClassifier(**{
            **params, "n_estimators": selected, "early_stopping_rounds": None,
        })
        refit.fit(X, y, verbose=False)
        # Copy fitted state only; preserve constructor parameters for clone().
        self._Booster = refit.get_booster()
        self.n_classes_ = refit.n_classes_
        self._Booster.set_attr(best_iteration=str(selected - 1))
        self.evals_result_ = probe.evals_result()
        self.selected_n_estimators_ = selected
        self.early_stopping_best_score_ = float(probe.best_score)
        self.early_stopping_rounds_trained_ = int(probe.get_booster().num_boosted_rounds())
        self.early_stopping_fit_n_ = len(fit_rows)
        self.early_stopping_valid_n_ = len(stop_rows)
        self.refit_n_ = len(y)
        self.stopped_early_ = self.early_stopping_rounds_trained_ < self.n_estimators
        return self


ENSEMBLE_MODELS = {
    "rf": (
        RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
        {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [20, 100]},
    ),
    "xgb": (
        TrainOnlyEarlyStoppingXGBClassifier(
            n_estimators=2000, early_stopping_rounds=50,
            tree_method="hist", eval_metric="logloss",
            n_jobs=8, random_state=config.RANDOM_STATE,
        ),
        {"clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1],
         "clf__subsample": [0.8]},
    ),
}
