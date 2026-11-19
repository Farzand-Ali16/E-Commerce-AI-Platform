"""
Reusable preprocessing for customer segmentation (K-Means).

One object does everything in a FIXED order, so training and the future web API
treat a new customer exactly the same way:

    1. fill missing values   (median learned from the data)
    2. transform skewed columns (log1p or sqrt)
    3. clip extreme values   (mean +/- clip_sigma * std, learned from the data)
    4. StandardScaler        (mean 0, std 1)

Everything it "learns" (medians, clip limits, scaler) is learned in fit() and
only APPLIED in transform(). It is saved with joblib in Stage 4.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class ClusteringPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features, log_cols=(), sqrt_cols=(), impute_cols=(), clip_sigma=4.0):
        self.features = features
        self.log_cols = log_cols
        self.sqrt_cols = sqrt_cols
        self.impute_cols = impute_cols
        self.clip_sigma = clip_sigma

    # -- helpers ---------------------------------------------------------
    def _check(self, X):
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = X[list(self.features)].astype(float).copy()
        if np.isinf(X.to_numpy()).any():
            raise ValueError("Input contains infinite values")
        return X

    def _impute_and_transform(self, X):
        for c in self.impute_cols:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.log_cols:
            if (X[c] < 0).any():
                raise ValueError(f"{c} has negative values, cannot apply log1p")
            X[c] = np.log1p(X[c])
        for c in self.sqrt_cols:
            if (X[c] < 0).any():
                raise ValueError(f"{c} has negative values, cannot apply sqrt")
            X[c] = np.sqrt(X[c])
        return X

    # -- sklearn API -----------------------------------------------------
    def fit(self, X, y=None):
        X = self._check(X)
        self.medians_ = {c: float(X[c].median()) for c in self.impute_cols}   # NaN is ignored by median()
        self.n_imputed_ = {c: int(X[c].isna().sum()) for c in self.impute_cols}
        Xt = self._impute_and_transform(X)
        mean, std = Xt.mean(), Xt.std(ddof=0)
        self.clip_low_ = mean - self.clip_sigma * std
        self.clip_high_ = mean + self.clip_sigma * std
        self.n_clipped_ = ((Xt < self.clip_low_) | (Xt > self.clip_high_)).sum().astype(int).to_dict()
        self.scaler_ = StandardScaler().fit(Xt.clip(self.clip_low_, self.clip_high_, axis=1))
        return self

    def transform(self, X):
        X = self._check(X)
        Xt = self._impute_and_transform(X).clip(self.clip_low_, self.clip_high_, axis=1)
        out = pd.DataFrame(self.scaler_.transform(Xt), columns=list(self.features), index=X.index)
        return out

    def transform_before_scaling(self, X):
        """Same as transform() but stops before StandardScaler (useful for plots)."""
        X = self._check(X)
        return self._impute_and_transform(X).clip(self.clip_low_, self.clip_high_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.features)
