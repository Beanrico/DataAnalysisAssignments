import numpy as np
from scipy import stats

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