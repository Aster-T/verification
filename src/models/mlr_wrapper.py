"""
Multiple Linear Regression wrapper with exportable W matrix.

PUBLIC API:
  MLRWithW.fit(X, y, is_nominal=None) -> Self
  MLRWithW.predict(X) -> np.ndarray
  MLRWithW.get_W() -> dict with keys w_vec, w_outer, feature_names

STANDARDIZATION:
  Continuous columns: subtract mu_, divide by sd_ (sd_ clamped away from 0).
  Nominal columns (is_nominal[i] == True): mu_ forced to 0, sd_ forced to 1,
  i.e. the column is passed through untouched. z-scoring integer-coded
  categoricals would imply a spurious linear ordering.

  mu_ and sd_ are locked at fit time; predict uses the same parameters.

NAN:
  Per 00_conventions.md NaN is not imputed in a generic preprocessing step.
  If the caller passes NaNs in X, SimpleImputer(strategy="mean") is applied
  *inside* MLRWithW (fit on train, applied on predict) so the linear fit is
  well-defined. TabPFN / tree models upstream never see this imputation.
"""

from __future__ import annotations

import numpy as np
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
          X: (n, f) float array. May contain NaN (mean-imputed internally).
          y: (n,) float array.
          is_nominal: optional list of length f. Columns flagged True skip
            standardization.
          feature_names: optional list of length f, forwarded to get_W().
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, f = X.shape

        if is_nominal is None:
            self.is_nominal_ = np.zeros(f, dtype=bool)
        else:
            self.is_nominal_ = np.asarray(is_nominal, dtype=bool)
            if self.is_nominal_.shape != (f,):
                raise ValueError(
                    f"is_nominal length {self.is_nominal_.shape[0]} != n_features {f}"
                )
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
        X = np.asarray(X, dtype=np.float64)
        if self._imputer is not None:
            X = self._imputer.transform(X)
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
