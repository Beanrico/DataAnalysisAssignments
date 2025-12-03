import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

def add_grade_group(df, threshold=10):
    df = df.copy()
    df['grade_group'] = np.where(df['G3'] >= threshold, 'High', 'Low')
    return df


def fit_interaction_model(df, threshold=10):
    df2 = add_grade_group(df, threshold=threshold)

    model = smf.ols('G3 ~ absences * grade_group', data=df2).fit()
    return model


def summarize_interaction(model):

    params = model.params
    pvalues = model.pvalues

    interaction_name = [name for name in params.index if 'absences:grade_group' in name][0]

    beta_int = params[interaction_name]
    p_int = pvalues[interaction_name]

    print("Interaction term:", interaction_name)
    print(f"  beta = {beta_int:.3f}")
    print(f"  p    = {p_int:.4f}")

    if p_int < 0.05:
        print("  -> 欠席数と成績の関係は、High グループと Low グループで有意に異なります。")
    else:
        print("  -> 欠席数と成績の関係の違いは、統計的には有意とは言えません。")


def separate_correlations(df, threshold=10):
    """
    Low / High それぞれで absences と G3 の相関を計算する。
    Figure(d) の r を再現・確認する用途。
    """
    df2 = add_grade_group(df, threshold=threshold)

    corrs = {}
    for grp in ['Low', 'High']:
        sub = df2[df2['grade_group'] == grp]
        r, p = stats.pearsonr(sub['absences'], sub['G3'])
        corrs[grp] = {'r': r, 'p': p, 'n': len(sub)}

    return corrs

def one_way_anova_studytime(df, print_table=True):
    """
    一元配置 ANOVA:
        studytime の 4 水準 (1, 2, 3, 4) で G3 の平均を比較する。

    Parameters
    ----------
    df : pandas.DataFrame
        少なくとも 'studytime' と 'G3' 列を含むデータフレーム。
    print_table : bool
        True のとき、各群の記述統計と ANOVA 結果を表示する。

    Returns
    -------
    F : float
        ANOVA の F 値
    p : float
        ANOVA の p 値
    """

    # 必要な列だけ取り出し、欠損を除去
    sub = df[['studytime', 'G3']].dropna()

    # studytime ごとに G3 を集める
    groups = {}
    for level in sorted(sub['studytime'].unique()):
        vals = sub.loc[sub['studytime'] == level, 'G3'].values
        groups[int(level)] = vals

    # 2 群以上ないと ANOVA できない
    if len(groups) < 2:
        raise ValueError("少なくとも 2 つ以上の studytime 水準が必要です。")

    # 各群の記述統計を表示
    if print_table:
        print("studytime  n   mean(G3)   sd(G3)")
        for level, vals in groups.items():
            mean = np.mean(vals)
            sd   = np.std(vals, ddof=1)
            print(f"{level:9d} {len(vals):3d} {mean:9.3f} {sd:9.3f}")
        print()

    # 一元配置 ANOVA
    F, p = stats.f_oneway(*groups.values())

    if print_table:
        df_between = len(groups) - 1
        df_within  = len(sub) - len(groups)
        print(f"One-way ANOVA (studytime -> G3)")
        print(f"  F({df_between}, {df_within}) = {F:.3f},  p = {p:.4f}")
        if p < 0.05:
            print("  -> studytime の水準間で G3 の平均に有意な差があります。")
        else:
            print("  -> studytime の水準間で G3 の平均の差は有意とは言えません。")

    return F, p