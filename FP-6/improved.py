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


def _onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot_encoder()),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return pre


def improved_regression_full(df_full: pd.DataFrame, drop_cols=None, svd_components=60):

    if drop_cols is None:
        drop_cols = []

    y = df_full["G3"].copy()
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X)

    pipe = Pipeline([
        ("pre", pre),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return {
        "version": f"full + SVD={svd_components}",
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "dropped": drop_cols,
    }


def improved_classification_full(df_full: pd.DataFrame, drop_cols=None, svd_components=60, threshold=10):

    if drop_cols is None:
        drop_cols = []

    y = (df_full["G3"] >= threshold).astype(int)
    X = df_full.drop(columns=["G3"] + drop_cols).copy()

    pre = build_preprocessor(X)

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


def search_best_svd_full(df_full: pd.DataFrame, drop_cols=None, svd_list=(10, 20, 40, 60, 80), threshold=10):

    if drop_cols is None:
        drop_cols = []

    best_reg = None
    best_clf = None

    for k in svd_list:
        reg = improved_regression_full(df_full, drop_cols=drop_cols, svd_components=k)
        clf = improved_classification_full(df_full, drop_cols=drop_cols, svd_components=k, threshold=threshold)

        if (best_reg is None) or (reg["R2"] > best_reg["R2"]):
            best_reg = reg
        if (best_clf is None) or (clf["F1"] > best_clf["F1"]):
            best_clf = clf

    return best_reg, best_clf