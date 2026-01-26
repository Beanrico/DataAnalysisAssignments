from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from IPython.display import display, Markdown


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )


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


RANDOM_STATE = 0


def _onehot_encoder(sparse: bool):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def build_preprocessor(X: pd.DataFrame, onehot_sparse: bool):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot_encoder(sparse=onehot_sparse)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def improved_regression_full(df_full: pd.DataFrame, drop_cols=None, svd_components=None):
    if drop_cols is None:
        drop_cols = []

    y = df_full["G3"].copy()
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X, onehot_sparse=False)

    steps = [("pre", pre)]
    if svd_components is not None:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)))
    steps.append(("model", GradientBoostingRegressor(random_state=RANDOM_STATE)))

    pipe = Pipeline(steps)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    ver = "full + preprocess only" if svd_components is None else f"full + SVD={svd_components}"
    return {
        "version": ver,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "dropped": drop_cols,
    }


def improved_classification_full(df_full: pd.DataFrame, drop_cols=None, svd_components=30, threshold=10):
    if drop_cols is None:
        drop_cols = []

    y = (df_full["G3"] >= threshold).astype(int)
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X, onehot_sparse=True)

    pipe = Pipeline([
        ("pre", pre),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return {
        "version": f"full + SVD={svd_components}",
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "threshold": threshold,
        "dropped": drop_cols,
    }


def _max_features_after_preprocess(df_full: pd.DataFrame, drop_cols=None, onehot_sparse=True):
    if drop_cols is None:
        drop_cols = []

    y = df_full["G3"].copy()
    X = df_full.drop(columns=["G3"] + drop_cols).copy()
    pre = build_preprocessor(X, onehot_sparse=onehot_sparse)
    Xt = pre.fit_transform(X, y)
    return Xt.shape[1]


def search_best_full(
    df_full: pd.DataFrame,
    drop_cols=None,
    reg_svd_list=(None, 10, 20, 30, 40, 50),
    clf_svd_list=(10, 20, 30, 40, 50),
    threshold=10
):

    if drop_cols is None:
        drop_cols = []

    max_k_clf = _max_features_after_preprocess(df_full, drop_cols=drop_cols, onehot_sparse=True)
    clf_svd_list = tuple(k for k in clf_svd_list if k <= max_k_clf)

    max_k_reg = _max_features_after_preprocess(df_full, drop_cols=drop_cols, onehot_sparse=False)
    reg_svd_list = tuple(k for k in reg_svd_list if (k is None) or (k <= max_k_reg))

    best_reg = None
    best_clf = None

    for k in reg_svd_list:
        reg = improved_regression_full(df_full, drop_cols=drop_cols, svd_components=k)
        if (best_reg is None) or (reg["R2"] > best_reg["R2"]):
            best_reg = reg

    for k in clf_svd_list:
        clf = improved_classification_full(df_full, drop_cols=drop_cols, svd_components=k, threshold=threshold)
        if (best_clf is None) or (clf["F1"] > best_clf["F1"]):
            best_clf = clf

    return best_reg, best_clf
    

def improved_regression_full_with_preds(df_full: pd.DataFrame, drop_cols=None, svd_components=None):
    if drop_cols is None:
        drop_cols = []
    y = df_full["G3"].copy()
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X, onehot_sparse=False)
    steps = [("pre", pre)]
    if svd_components is not None:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)))
    steps.append(("model", GradientBoostingRegressor(random_state=RANDOM_STATE)))
    pipe = Pipeline(steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    ver = "full + preprocess only" if svd_components is None else f"full + SVD={svd_components}"
    metrics = {
        "version": ver,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "dropped": drop_cols,
    }
    return metrics, y_test, pred


def improved_classification_full_with_preds(df_full: pd.DataFrame, drop_cols=None, svd_components=30, threshold=10):
    if drop_cols is None:
        drop_cols = []
    y = (df_full["G3"] >= threshold).astype(int)
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X, onehot_sparse=True)
    pipe = Pipeline([
        ("pre", pre),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "version": f"full + SVD={svd_components}",
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "threshold": threshold,
        "dropped": drop_cols,
    }
    return metrics, y_test, pred


def final_regression_result(df_small: pd.DataFrame, df_full: pd.DataFrame):
    display_title("Predicted vs actual G3 for the best regression model (test set)", pref="Figure", num=4, center=True)
    res_small, best_name_s, best_model_s, split_s, pred_s, _ = fit_and_evaluate_regression(df_small)
    Xtr, Xte, ytr, yte = split_s

    best_reg, _ = search_best_full(df_full)  # dict（R2/MAE/RMSE）  [oai_citation:7‡GitHub](https://raw.githubusercontent.com/Beanrico/DataAnalysisAssignments/refs/heads/main/FP-7/improved.py)
    k = None if "preprocess only" in best_reg["version"] else int(best_reg["version"].split("SVD=")[1])
    reg_m, yte_f, pred_f = improved_regression_full_with_preds(df_full, svd_components=k)

    best_is_full = reg_m["R2"] >= float(res_small.loc[0, "r2"])
    if best_is_full:
        best_label = f"GBR ({reg_m['version']})"
        best_metrics = {"r2": reg_m["R2"], "mae": reg_m["MAE"], "rmse": reg_m["RMSE"]}
        fig, ax = plot_pred_vs_actual(yte_f, pred_f, title="Regression: Best model (test)")
    else:
        best_label = best_name_s
        best_row = res_small[res_small["model"] == best_name_s].iloc[0]
        best_metrics = {"r2": float(best_row["r2"]), "mae": float(best_row["mae"]), "rmse": float(best_row["rmse"])}
        fig, ax = plot_pred_vs_actual(yte, pred_s, title="Regression: Best model (test)")

    base_row = res_small[res_small["model"] == "Baseline_mean"].iloc[0]
    table = pd.DataFrame([
        {"model": "Baseline_mean", "r2": float(base_row["r2"]), "mae": float(base_row["mae"]), "rmse": float(base_row["rmse"])},
        {"model": best_label, **best_metrics},
    ])
    return table, fig


def final_classification_result(df_small: pd.DataFrame, df_full: pd.DataFrame, threshold=10):
    display_title("Confusion matrix of the best classification model (test set).", pref="Figure", num=5, center=True)
    res_small, best_name_s, best_model_s, split_s, pred_s, _ = fit_and_evaluate_classification(df_small, threshold=threshold)
    Xtr, Xte, ytr, yte = split_s

    _, best_clf = search_best_full(df_full, threshold=threshold)  
    k = int(best_clf["version"].split("SVD=")[1])
    clf_m, yte_f, pred_f = improved_classification_full_with_preds(df_full, svd_components=k, threshold=threshold)

    best_f1_small = float(res_small.loc[0, "f1"])
    best_is_full = clf_m["F1"] >= best_f1_small

    if best_is_full:
        best_label = f"LogReg ({clf_m['version']})"
        best_metrics = {"accuracy": clf_m["Accuracy"], "f1": clf_m["F1"]}
        fig, ax = plot_confusion(yte_f, pred_f, title="Classification: Best model (test)")
    else:
        best_label = best_name_s
        best_row = res_small[res_small["model"] == best_name_s].iloc[0]
        best_metrics = {"accuracy": float(best_row["accuracy"]), "f1": float(best_row["f1"])}
        fig, ax = plot_confusion(yte, pred_s, title="Classification: Best model (test)")

    base_row = res_small[res_small["model"] == "Baseline_most_frequent"].iloc[0]
    table = pd.DataFrame([
        {"model": "Baseline_most_frequent", "accuracy": float(base_row["accuracy"]), "f1": float(base_row["f1"])},
        {"model": best_label, **best_metrics},
    ])
    return table, fig