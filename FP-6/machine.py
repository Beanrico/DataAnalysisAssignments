# machine.py
"""
FP-6: Machine Learning (student-mat.csv)

This module provides TWO options (both validated with a held-out test set):

1) Regression: predict final grade G3 (continuous)
2) Classification: predict grade_group (High/Low) derived from G3

Designed to work with your existing parse_data.py which defines p_d.df
with columns: ['studytime', 'absences', 'G1', 'G3'].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


FEATURE_COLS_DEFAULT = ["studytime", "absences", "G1"]


def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}. df.columns={list(df.columns)}")


def get_xy_regression(df: pd.DataFrame, feature_cols: list[str] | None = None):
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    _check_cols(df, feature_cols + ["G3"])
    X = df[feature_cols].copy()
    y = df["G3"].copy()
    return X, y


def get_xy_classification(
    df: pd.DataFrame,
    threshold: int = 10,
    feature_cols: list[str] | None = None
):
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    _check_cols(df, feature_cols + ["G3"])
    X = df[feature_cols].copy()
    y = (df["G3"] >= threshold).astype(int)
    return X, y


def fit_and_evaluate_regression(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 0,
    feature_cols: list[str] | None = None,
):
    X, y = get_xy_regression(df, feature_cols=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Baseline_mean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

    rows = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        })

    results_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]
    best_model = fitted[best_name]
    y_pred_best = best_model.predict(X_test)

    perm = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=30, random_state=random_state, scoring="r2"
    )
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    return results_df, best_name, best_model, (X_train, X_test, y_train, y_test), y_pred_best, importance


def cross_validate_regression_r2(
    df: pd.DataFrame,
    model=None,
    cv: int = 5,
    random_state: int = 0,
    feature_cols: list[str] | None = None,
):
    X, y = get_xy_regression(df, feature_cols=feature_cols)
    if model is None:
        model = GradientBoostingRegressor(random_state=random_state)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    return float(scores.mean()), float(scores.std()), scores


def plot_pred_vs_actual(y_true, y_pred, title="Regression: Predicted vs Actual (test)"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(importance: pd.Series, title="Feature importance (permutation)"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    imp = importance.sort_values(ascending=True)
    ax.barh(imp.index, imp.values)
    ax.set_title(title)
    ax.set_xlabel("Importance (higher = more influential)")
    plt.tight_layout()
    return fig, ax


def fit_and_evaluate_classification(
    df: pd.DataFrame,
    threshold: int = 10,
    test_size: float = 0.25,
    random_state: int = 0,
    feature_cols: list[str] | None = None,
):
    X, y = get_xy_classification(df, threshold=threshold, feature_cols=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
    ])

    models = {
        "Baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": logreg,
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    rows = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        })

    results_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]
    best_model = fitted[best_name]
    y_pred_best = best_model.predict(X_test)

    perm = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=30, random_state=random_state, scoring="f1"
    )
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    return results_df, best_name, best_model, (X_train, X_test, y_train, y_test), y_pred_best, importance


def cross_validate_classification_f1(
    df: pd.DataFrame,
    threshold: int = 10,
    model=None,
    cv: int = 5,
    random_state: int = 0,
    feature_cols: list[str] | None = None,
):
    X, y = get_xy_classification(df, threshold=threshold, feature_cols=feature_cols)
    if model is None:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
        ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
    return float(scores.mean()), float(scores.std()), scores


def plot_confusion(y_true, y_pred, title="Classification: Confusion matrix (test)"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Low", "High"],
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax