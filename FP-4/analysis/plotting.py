import numpy as np
import matplotlib.pyplot as plt
from .stats import regression_line, corrcoeff

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