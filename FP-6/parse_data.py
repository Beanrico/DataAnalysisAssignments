import pandas as pd

df0 = pd.read_csv('student-mat.csv', delimiter=';')
df = df0[  ['studytime','absences','G1','G3']  ]

G1 = df['G1']
st = df['studytime']
ab = df['absences']
G3 = df['G3']