import pandas as pd

df_full = pd.read_csv('student-mat.csv', delimiter=';')
df = df_full[  ['studytime','absences','G1','G3']  ]

G1 = df['G1']
st = df['studytime']
ab = df['absences']
G3 = df['G3']