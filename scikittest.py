import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


data = load_wine()

print(type(data))
print(data.keys())

df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=["kind(target)"])

print(df_x)
print(df_y)

x_y = pd.concat([df_x, df_y], axis=1)
print(x_y)

n_clusters = 3
model = KMeans(n_clusters=n_clusters)

X = data.data[:, [0, 9]]
pred = model.fit_predict(X)
