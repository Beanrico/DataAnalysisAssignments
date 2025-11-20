import pandas as pd
from .stats import central, dispersion

def central_table(df):
    df_central = df.apply(lambda x: central(x), axis=0)
    df_central.index = ['Mean', 'Median', 'Mode']
    return df_central

def dispersion_table(df):
    df_disp = df.apply(lambda x: dispersion(x), axis=0)
    df_disp.index = ['Std', 'Min', 'Max', 'Range', '25%', '75%', 'IQR']
    return df_disp
