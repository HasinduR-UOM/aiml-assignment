"""SHAP and LIME helpers that work on any fitted Pipeline from modeling.py.

SHAP (SHapley Additive exPlanations)
  * Tree models -> shap.TreeExplainer on the model inside the pipeline (exact,
    fast). One-hot / missing-indicator columns are summed back to the original
    feature so plots show "district", not 25 dummy columns.
  * Other models (Ridge, SVR, KNN) -> model-agnostic permutation
    explainer on the raw features.
LIME (Local Interpretable Model-agnostic Explanations)
  * Fits a small weighted linear model around ONE prediction by perturbing
    the inputs. Categorical features are label-encoded for LIME and decoded
    before calling the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import CAT_FEATURES

TREE_TYPES = ("RandomForest", "GradientBoosting", "XGB", "LGBM","DecisionTree", "ExtraTrees")


class RawCodec:
    """Label-encode categorical columns so explainers can work on a numeric matrix."""

    def __init__(self, X: pd.DataFrame, features: list[str]):
        self.features = features
        self.cats = {c: sorted(X[c].dropna().astype(str).unique()) for c in features if c in CAT_FEATURES}

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.features].copy()
        for c, levels in self.cats.items():
            Z[c] = Z[c].astype(str).map({v: i for i, v in enumerate(levels)}).fillna(-1)
        return Z.astype(float).to_numpy()

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(A, columns=self.features)
        for c, levels in self.cats.items():
            idx = np.clip(np.round(df[c]).astype(int), 0, len(levels) - 1)
            df[c] = [levels[i] for i in idx]
        return df

    @property
    def cat_idx(self):
        return [self.features.index(c) for c in self.cats]


def _final(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def is_tree_pipeline(model) -> bool:
    return any(t in type(_final(model)).__name__ for t in TREE_TYPES)


def _group(values: np.ndarray, cols: list[str], features: list[str]) -> np.ndarray:
    """Sum transformed-column SHAP values back to original features."""
    out = np.zeros((values.shape[0], len(features)))
    for j, col in enumerate(cols):
        src = col.replace("missingindicator_", "")
        for i, f in enumerate(features):
            if src == f or (f in CAT_FEATURES and src.startswith(f + "_")):
                out[:, i] += values[:, j]
                break
    return out


def shap_explanation(model, X: pd.DataFrame, features: list[str],
                     background: pd.DataFrame | None = None, max_evals: int = 800) -> shap.Explanation:
    """Return a shap.Explanation over the ORIGINAL features for rows of X."""
    X = X[features]
    if is_tree_pipeline(model):
        pre, est = model.named_steps["pre"], _final(model)
        Xt = pre.transform(X)
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xt)
        base = np.repeat(np.ravel(explainer.expected_value)[0], len(Xt))
        grouped = _group(np.asarray(sv), list(Xt.columns), features)
        return shap.Explanation(values=grouped, base_values=np.asarray(base), data=X.to_numpy(),
                                feature_names=features)
    # model-agnostic path
    bg = background if background is not None else X
    codec = RawCodec(pd.concat([bg, X]), features)
    bg_enc = codec.encode(bg.sample(min(100, len(bg)), random_state=0))
    f = lambda A: model.predict(codec.decode(A))  # noqa: E731
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(bg_enc), max_evals=max_evals)
    ex = explainer(codec.encode(X))
    return shap.Explanation(values=ex.values, base_values=ex.base_values, data=X.to_numpy(),
                            feature_names=features)


def shap_importance(ex: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)


# --------------------------------------------------------------------------
def lime_explainer(model, X_train: pd.DataFrame, features: list[str], seed: int = 42):
    from lime.lime_tabular import LimeTabularExplainer
    codec = RawCodec(X_train, features)
    med = X_train[features].median(numeric_only=True)
    A = codec.encode(X_train[features].fillna(med))       # LIME cannot handle NaN
    # treat binary / constant numeric columns as categorical (avoids zero-variance bins)
    extra = [i for i, f in enumerate(features)
             if f not in codec.cats and set(np.unique(A[:, i])) <= {0.0, 1.0}]
    cat_idx = codec.cat_idx + extra
    names = {codec.features.index(c): lv for c, lv in codec.cats.items()}
    for i in extra:
        names[i] = ["0", "1"]
    explainer = LimeTabularExplainer(
        A, feature_names=features, mode="regression", categorical_features=cat_idx,
        categorical_names=names, discretize_continuous=True, random_state=seed)
    predict = lambda Z: model.predict(codec.decode(Z))  # noqa: E731
    return explainer, codec, predict, med


def lime_explain_row(model, X_train: pd.DataFrame, row: pd.DataFrame, features: list[str],
                     num_features: int = 10, num_samples: int = 3000, seed: int = 42, explainer_bundle=None):
    explainer, codec, predict, med = explainer_bundle or lime_explainer(model, X_train, features, seed)
    r = row[features].copy().fillna(med)
    return explainer.explain_instance(codec.encode(r)[0], predict,
                                      num_features=num_features, num_samples=num_samples)


def lime_table(exp) -> pd.DataFrame:
    return pd.DataFrame(exp.as_list(), columns=["condition", "weight_kg_ha"])


def compare_shap_lime(shap_row: shap.Explanation, exp, top: int = 8) -> pd.DataFrame:
    """Side-by-side ranking of the top features for one prediction."""
    s = pd.Series(np.abs(shap_row.values), index=shap_row.feature_names).sort_values(ascending=False)
    lime_feats = []
    for cond, w in exp.as_list():
        for f in sorted(shap_row.feature_names, key=len, reverse=True):
            if f in cond:
                lime_feats.append((f, w))
                break
    lime_s = pd.Series({f: abs(w) for f, w in lime_feats}).sort_values(ascending=False)
    n = max(top, 1)
    return pd.DataFrame({"SHAP rank": list(s.index[:n]) + [""] * (n - len(s.index[:n])),
                         "LIME rank": list(lime_s.index[:n]) + [""] * (n - len(lime_s.index[:n]))},
                        index=range(1, n + 1))
