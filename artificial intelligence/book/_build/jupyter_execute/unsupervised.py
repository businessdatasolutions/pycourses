# Unsupervised Learning



Watch this video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/4b5d3muPQmA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<hr>

:::{admonition} Assignments:
:class: hint

1. *Individual* -- Find a real-life business application of unsupervised learning
2. *Team* -- Gather all applications and discuss what the shared characteristics are
3. *Team* -- Develop a concept for a new application that is based on unsupervised learning
:::

## An example

We will be using the k-means clustering algorithm to illustrate the functioning of unsupervised learning. The following code will generate some random clustered data. It has two dimension, $x$ and $y$.

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y =  make_blobs(n_samples=100, centers=4, random_state=500, cluster_std=1.25)

sns.relplot(x=x[:, 0], y= x[:, 1])

We will use this data to train a model that will try to detect four clusters. The colouring of the dots is based on what the algorithm has learned.

model = KMeans(n_clusters=4, random_state=0)

model.fit(x)

y_ = model.predict(x)
y_

sns.relplot(x=x[:, 0], y= x[:, 1], hue=y_)

:::{admonition} Assignments:
:class: hint

- *Individual* -- Try to change the parameters and understand the mechanics of the separate functions
:::