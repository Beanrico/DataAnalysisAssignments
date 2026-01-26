import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    display_title("Summary statistics of studytime, absences, G1, and G3",pref="Table", num=1, center=False)
    cols = ["studytime", "absences", "G1", "G3"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    x = df[cols]
    out = pd.DataFrame({
        "mean": x.mean(numeric_only=True),
        "std": x.std(numeric_only=True),
        "min": x.min(numeric_only=True),
        "max": x.max(numeric_only=True),
        "median": x.median(numeric_only=True),
    })
    return out

def fig_g3_distribution(df: pd.DataFrame, bins=20):
    display_title("Distribution of final grade (G3)", pref="Figure", num=1, center=True)
    
    if "G3" not in df.columns:
        raise KeyError("Missing column 'G3' in df")

    s = df["G3"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=bins)

    mean = float(s.mean())
    median = float(s.median())
    ax.axvline(mean, linestyle="--", linewidth=2, label=f"Mean={mean:.2f}")
    ax.axvline(median, linestyle=":", linewidth=2, label=f"Median={median:.2f}")

    ax.set_xlabel("G3 (final grade)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig, ax