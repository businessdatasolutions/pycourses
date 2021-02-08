# Supervised Learning



import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y =  make_blobs(n_samples=100, centers=4, random_state=500, cluster_std=1.25)
model = KMeans(n_clusters=4, random_state=0)

model.fit(x)
KMeans(n_clusters=4, random_state=0)

y_ = model.predict(x)
y_

sns.relplot(x=x[:, 0], y= x[:, 1], hue=y_);

