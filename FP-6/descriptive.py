import numpy as np
import parse_data as p_d
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
print(classical.my_function)
df = p_d.df

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )
    
def central(x, print_output=True):
    x0 = np.mean(x)
    x1 = np.median(x)
    x2 = stats.mode(x, keepdims=True).mode[0]
    return x0, x1, x2

def dispersion(x, print_output=True):
    y0 = np.std(x)
    y1 = np.min(x)
    y2 = np.max(x)
    y3 = y2 - y1
    y4 = np.percentile(x, 25)
    y5 = np.percentile(x, 75)
    y6 = y5 - y4
    return y0,y1,y2,y3,y4,y5,y6

def regression_line(x, y):
    a = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    b = np.mean(y) - a * np.mean(x)
    return a, b

def corrcoeff(x, y):
    return np.corrcoef(x, y)[0, 1]

def central_table(df):
    df_central = df.apply(lambda x: central(x), axis=0)
    df_central.index = ['Mean', 'Median', 'Mode']
    return df_central

def dispersion_table(df):
    df_disp = df.apply(lambda x: dispersion(x), axis=0)
    df_disp.index = ['Std', 'Min', 'Max', 'Range', '25%', '75%', 'IQR']
    return df_disp


def plot_regression_line(ax, x, y, **kwargs):
    a, b = regression_line(x, y)
    xline = np.linspace(min(x), max(x), 100)
    yline = a * xline + b

    ax.plot(xline, yline, **kwargs)

def plot_multiple_scatter(x_list, y, xlabels=None, colors=None, ylabel='G3'):
    n = len(x_list)
    if colors is None:
        colors = ['b', 'r', 'g'][:n]
    if xlabels is None:
        xlabels = [f'x{i+1}' for i in range(n)]

    fig, axs = plt.subplots(1, n, figsize=(10, 3), tight_layout=True)

    for ax, x, c, xlabel in zip(axs, x_list, colors, xlabels):
        ax.scatter(x, y, alpha=0.5, color=c)
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes,
                bbox=dict(color='0.8', alpha=0.7))
        ax.set_xlabel(xlabel)

    axs[0].set_ylabel(ylabel)
    for ax in axs[1:]:
        ax.set_yticklabels([])

    plt.show()

def plot_by_grade_split(x, y, split_value=10, xlabel='absences', ylabel='G3',
                        colors=('g', 'g'), titles=('Low grades', 'High grades')):
  
    i_low  = y <= split_value
    i_high = y > split_value
    masks  = [i_low, i_high]

    fig, axs = plt.subplots(1, 2, figsize=(8,3), tight_layout=True)

    for ax, mask, color, title in zip(axs, masks, colors, titles):
        ax.scatter(x[mask], y[mask], alpha=0.5, color=color)
        plot_regression_line(ax, x[mask], y[mask], color='k', ls='-', lw=2)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    
    axs[0].set_ylabel(ylabel)
    plt.show()



def plot_descriptive(G1, st, ab, G3):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)
    ivs = [G1, st, ab]
    colors = ['b', 'r', 'g']

    for ax, x, c in zip(axs.ravel(), ivs, colors):
        ax.scatter(x, G3, alpha=0.5, color=c)
        plot_regression_line(ax, x, G3, color='k', lw=2)
        r = corrcoeff(x, G3)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes)

    labels = ['G1', 'Study Time', 'Absences']
    for ax, lbl in zip(axs.ravel(), labels):
        ax.set_xlabel(lbl)

    for ax in axs[:, 0]:
        ax.set_ylabel("G3")

    ax = axs[1, 1]
    low = G3 <= 10
    high = G3 > 10

    groups = [(low, 'm', 'Low grades'),
              (high, 'c', 'High grades')]

    for mask, color, label in groups:
        ax.scatter(ab[mask], G3[mask], color=color, alpha=0.5, label=label)
        plot_regression_line(ax, ab[mask], G3[mask], color=color, lw=2)
        r = corrcoeff(ab[mask], G3[mask])
        ax.text(0.6, 0.2 if label=='Low grades' else 0.7,
                f'r = {r:.3f}', color=color, transform=ax.transAxes)

    ax.legend()
    ax.set_xlabel("Absences")

    for ax, lbl in zip(axs.ravel(), ['a', 'b', 'c', 'd']):
        ax.text(0.02, 0.92, f"({lbl})", transform=ax.transAxes)

    plt.show()

def figure(num):
    if num == 1:
        x = df['G1']
        y = df['G3']
        
        slope, intercept, r, p, _ = stats.linregress(x, y)
        print(f"G1–G3: r = {r:.3f}, p = {p:.4f}")
        
        xline = np.linspace(x.min(), x.max(), 100)
        yline = slope * xline + intercept
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, alpha=0.6, edgecolor='white', linewidth=0.5)
        ax.plot(xline, yline)
        
        ax.set_xlabel('G1 (first-period grade)')
        ax.set_ylabel('G3 (final grade)')
        ax.set_title('Figure 1. Relationship between G1 and G3')
        
        plt.tight_layout()
        plt.show()
    if num == 2:
        df2 = add_grade_group(df, threshold=10)
        fit_interaction_model(df, threshold=10)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {'Low': 'magenta', 'High': 'cyan'}
        
        for grp in ['Low', 'High']:
            sub = df2[df2['grade_group'] == grp]
            ax.scatter(sub['absences'], sub['G3'], alpha=0.6, label=f'{grp} grades',
                       edgecolor='white', linewidth=0.5, s=40, color=colors[grp])
            slope, intercept, r, p, _ = stats.linregress(sub['absences'], sub['G3'])
            print(f'{grp} group: r = {r:.3f}, p = {p:.4f}')
            xline = np.linspace(sub['absences'].min(), sub['absences'].max(), 50)
            yline = slope * xline + intercept
            ax.plot(xline, yline, color=colors[grp])
            xpos = sub['absences'].quantile(0.9)
            ypos = sub['G3'].quantile(0.1) if grp == 'Low' else sub['G3'].quantile(0.8)
            ax.text(xpos, ypos, f'{grp}: r = {r:.3f}', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('absences')
        ax.set_ylabel('G3')
        ax.set_title('Relationship between absences and final grades\nby performance group')
        ax.legend()
        plt.tight_layout()
        plt.show()
                