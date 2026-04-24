"""
Multiple Linear Regression wrapper with exportable W matrix.

PUBLIC API:
  MLRWithW.fit(X, y, is_nominal=None) -> Self
  MLRWithW.predict(X) -> np.ndarray
  MLRWithW.get_W() -> dict with keys w_vec, w_outer, feature_names

TEXT COLUMNS:
  fit() accepts X as numeric ndarray OR as object ndarray holding a mix of
  strings and numbers. String columns are factorized per-column via
  pd.factorize(sort=True) at fit time, the category→code mapping is stored,
  and predict() reapplies the same mapping (unseen categories map to NaN,
  which the SimpleImputer then fills).

STANDARDIZATION:
  Continuous columns: subtract mu_, divide by sd_ (sd_ clamped away from 0).
  Nominal columns (is_nominal[i] == True, OR a column that was factorized):
  mu_ forced to 0, sd_ forced to 1, i.e. the column is passed through
  untouched. z-scoring integer-coded categoricals would imply a spurious
  linear ordering.

NAN:
  If the caller passes NaNs in X (or unseen categories appear at predict
  time), SimpleImputer(strategy="mean") fills them so the linear fit is
  well-defined. TabPFN upstream never sees this imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


class MLRWithW:
    def __init__(self, standardize: bool = True) -> None:
        """
        ARGS:
          standardize: if True, continuous columns are z-scored using
            train-time mu / sd. Nominal columns are always passed through.
        """
        self.standardize = standardize
        self._lr: LinearRegression | None = None
        self._imputer: SimpleImputer | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.is_nominal_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        # column_index -> list[str] of categories (predict-time uses these).
        self._cat_mappings_: dict[int, list[str]] | None = None

    @staticmethod
    def _column_is_text(col: np.ndarray) -> bool:
        """True iff any element in the column is a Python str."""
        for v in col:
            if isinstance(v, str):
                return True
        return False

    def _encode_object_X(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Turn an object-dtype X into float64, factorizing text columns.

        Stores / re-uses self._cat_mappings_. Unseen categories at predict
        time become NaN (later filled by SimpleImputer).
        """
        n, f = X.shape
        X_num = np.empty((n, f), dtype=np.float64)
        if fit:
            self._cat_mappings_ = {}
        cat_maps = self._cat_mappings_ or {}
        for j in range(f):
            col = X[:, j]
            if fit:
                text_col = self._column_is_text(col)
            else:
                text_col = j in cat_maps
            if text_col:
                if fit:
                    codes, uniques = pd.factorize(col, sort=True)
                    cats = [str(u) for u in uniques.tolist()]
                    cat_maps[j] = cats
                    X_num[:, j] = codes.astype(np.float64)
                    X_num[codes < 0, j] = np.nan
                else:
                    mapping = {c: float(i) for i, c in enumerate(cat_maps[j])}
                    for i, v in enumerate(col):
                        if isinstance(v, str):
                            X_num[i, j] = mapping.get(v, np.nan)
                        elif v is None or (isinstance(v, float) and np.isnan(v)):
                            X_num[i, j] = np.nan
                        else:
                            try:
                                X_num[i, j] = float(v)
                            except (TypeError, ValueError):
                                X_num[i, j] = np.nan
            else:
                for i, v in enumerate(col):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        X_num[i, j] = np.nan
                    else:
                        try:
                            X_num[i, j] = float(v)
                        except (TypeError, ValueError):
                            X_num[i, j] = np.nan
        if fit:
            self._cat_mappings_ = cat_maps
        return X_num

    def _coerce_to_float(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X)
        if X.dtype == object:
            return self._encode_object_X(X, fit=fit)
        return X.astype(np.float64, copy=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_nominal: list[bool] | None = None,
        feature_names: list[str] | None = None,
    ) -> "MLRWithW":
        """
        Fit the linear regression.

        ARGS:
          X: (n, f) array. Numeric ndarray OR object ndarray with strings in
             nominal columns. String columns are factorized internally.
          y: (n,) float array.
          is_nominal: optional list of length f. Columns flagged True skip
            standardization. Columns auto-detected as text are ALWAYS treated
            as nominal regardless of this flag.
          feature_names: optional list of length f, forwarded to get_W().
        """
        self._cat_mappings_ = None
        X = self._coerce_to_float(X, fit=True)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, f = X.shape

        if is_nominal is None:
            nominal = np.zeros(f, dtype=bool)
        else:
            nominal = np.asarray(is_nominal, dtype=bool)
            if nominal.shape != (f,):
                raise ValueError(
                    f"is_nominal length {nominal.shape[0]} != n_features {f}"
                )
        # Any factorized text column is also nominal (no linear scaling).
        if self._cat_mappings_:
            for j in self._cat_mappings_:
                nominal[j] = True
        self.is_nominal_ = nominal
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        if np.isnan(X).any():
            self._imputer = SimpleImputer(strategy="mean")
            X = self._imputer.fit_transform(X)
        else:
            self._imputer = None

        if self.standardize:
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=0)
            sd = np.where(sd > 0, sd, 1.0)
            mu[self.is_nominal_] = 0.0
            sd[self.is_nominal_] = 1.0
        else:
            mu = np.zeros(f, dtype=np.float64)
            sd = np.ones(f, dtype=np.float64)
        self.mu_ = mu
        self.sd_ = sd

        Xs = (X - mu) / sd
        self._lr = LinearRegression()
        self._lr.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("MLRWithW.predict called before fit.")
        X = self._coerce_to_float(X, fit=False)
        if self._imputer is not None:
            X = self._imputer.transform(X)
        elif np.isnan(X).any():
            # Unseen categories or NaNs at predict time even though train
            # had none: fill with column means from training.
            col_mean = np.where(np.isfinite(self.mu_), self.mu_, 0.0)
            nan_mask = np.isnan(X)
            X = X.copy()
            X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
        Xs = (X - self.mu_) / self.sd_
        return self._lr.predict(Xs)

    def get_W(self) -> dict:
        """
        Export the learned coefficients.

        RETURNS:
          dict with keys:
            w_vec:   (f,) float, coef_ in standardized feature space.
            w_outer: (f, f) float, np.outer(w_vec, w_vec). This is a RANK-1
                     proxy for pairwise feature importance; it is NOT a
                     learned interaction term and should not be interpreted
                     as one. Its role is to be shape-comparable with the
                     TabPFN column attention matrix (F, F).
            feature_names: list[str] | None.
        """
        if self._lr is None:
            raise RuntimeError("MLRWithW.get_W called before fit.")
        w_vec = np.asarray(self._lr.coef_, dtype=np.float64).ravel()
        w_outer = np.outer(w_vec, w_vec)
        return {
            "w_vec": w_vec,
            "w_outer": w_outer,
            "feature_names": self.feature_names_,
        }
