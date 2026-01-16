import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score

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