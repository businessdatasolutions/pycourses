# Neural Networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

$$
   f:\mathbb R \to \mathbb R, \ y = 2x^2 - \frac{1}{3}x^3
$$

def f(x):
    return 2 * x ** 2- x ** 3 / 3

x = np.linspace(-2, 4, 25)
x

y = f(x)
y


plt.plot(x, y, 'ro')

```{math}
    f:\mathbb R \to \mathbb R, \ y = \alpha + \beta * x
```

beta = np.cov(x, y, ddof=0)[0,1] / np.var(x)
beta

alpha = y.mean() - beta * x.mean()
alpha

y_ = alpha + beta * x
MSE = ((y - y_) ** 2).mean()
MSE

plt.plot(x, y, 'ro', label='sample data')
plt.plot(x, y_, lw=3.0, label='linear regression')
plt.legend()

