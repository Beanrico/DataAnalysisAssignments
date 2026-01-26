import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from IPython.display import display, Markdown


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )


def run_h1(df: pd.DataFrame):
    display_title("Relationship between G1 and G3 (scatter with fitted regression line)", pref="Figure", num=2, center=True)
    """
    H1: Relationship between G1 and G3 (correlation + simple regression).
    Returns: stats_dict, fig
    """
    d = df[["G1", "G3"]].dropna().copy()
    n = len(d)

    r, p = stats.pearsonr(d["G1"], d["G3"])

    model = smf.ols("G3 ~ G1", data=d).fit()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(d["G1"], d["G3"], alpha=0.7)
    x = d["G1"].sort_values()
    yhat = model.predict(pd.DataFrame({"G1": x}))
    ax.plot(x, yhat, linewidth=2)
    ax.set_xlabel("G1 (first-period grade)")
    ax.set_ylabel("G3 (final grade)")
    fig.tight_layout()

    out = {
        "n": n,
        "r": float(r),
        "p": float(p),
        "coef_G1": float(model.params["G1"]),
        "intercept": float(model.params["Intercept"]),
        "r2": float(model.rsquared),
        "p_slope": float(model.pvalues["G1"]),
    }
    return out, fig


def run_h2(df: pd.DataFrame, threshold=10):
    display_title("Absences vs G3 by performance group, with interaction model fit (threshold = 10)", pref="Figure", num=3, center=True)
    """
    H2: Absences effect differs by performance group (interaction).
    Group: high if G3 >= threshold else low.
    Returns: stats_dict, fig
    """
    d = df[["absences", "G3"]].dropna().copy()
    d["group"] = (d["G3"] >= threshold).astype(int)

    high = d[d["group"] == 1]
    low  = d[d["group"] == 0]

    r_high, p_high = stats.pearsonr(high["absences"], high["G3"]) if len(high) >= 3 else (float("nan"), float("nan"))
    r_low,  p_low  = stats.pearsonr(low["absences"],  low["G3"])  if len(low) >= 3 else (float("nan"), float("nan"))

    model = smf.ols("G3 ~ absences * group", data=d).fit()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(low["absences"], low["G3"], alpha=0.7, label="Low performance")
    ax.scatter(high["absences"], high["G3"], alpha=0.7, label="High performance")

    xs = pd.Series(sorted(d["absences"].unique()))
    pred_low = model.predict(pd.DataFrame({"absences": xs, "group": 0}))
    pred_high = model.predict(pd.DataFrame({"absences": xs, "group": 1}))
    ax.plot(xs, pred_low, linewidth=2)
    ax.plot(xs, pred_high, linewidth=2)

    ax.set_xlabel("Absences (count)")
    ax.set_ylabel("G3")
    ax.legend()
    fig.tight_layout()

    out = {
        "threshold": threshold,
        "n": int(len(d)),
        "r_high": float(r_high),
        "p_high": float(p_high),
        "r_low": float(r_low),
        "p_low": float(p_low),
        "coef_absences": float(model.params.get("absences", float("nan"))),
        "p_absences": float(model.pvalues.get("absences", float("nan"))),
        "coef_interaction": float(model.params.get("absences:group", float("nan"))),
        "p_interaction": float(model.pvalues.get("absences:group", float("nan"))),
        "r2": float(model.rsquared),
    }
    return out, fig