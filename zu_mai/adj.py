import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# Loading data
df = pd.read_csv(r"feat1.csv",index_col=0)
df.dropna(inplace=True)
df = df.sample(frac=0.125, random_state=42)

# Negative sampling
under = RandomUnderSampler(sampling_strategy=1)
steps = [('u', under)]
pipeline = Pipeline(steps=steps)
X=df
X_resampled, y_resampled = pipeline.fit_resample(X, X['Label'])
X_resampled.to_csv(r'data.csv', index=False)
label = X_resampled.iloc[:, 10]
df = X_resampled.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
df.columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

# Factor Analysis
faa = FactorAnalyzer(25, rotation=None)
faa.fit(df)
faa_six = FactorAnalyzer(7, rotation='varimax')  
faa_six.fit(df)
# 1. factor loading
df1 = pd.DataFrame(np.abs(faa_six.loadings_), index=df.columns)
df11 = np.array(df1)
# 2. factor score
df2 = pd.DataFrame(faa_six.transform(df))  
df22 = np.array(df2)

# adj
A = df22
A = np.dot(A,A.T)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
matrix = np.where(A >= np.percentile(A, 80),1, 0)
np.save("A.npy",A)
np.save("matrix.npy",matrix)