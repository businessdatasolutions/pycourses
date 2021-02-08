# Reinforcement Learning

Watch this video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/kopoLzvh5jY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<hr>

import numpy as np

ssp = [1, 1, 1, 1, 0]
asp = [1, 0]

def epoch():
    tr = 0
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
    return tr

rl = np.array([epoch() for _ in range(15)])
rl


round(rl.mean(), 2)

def epoch():
    tr = 0
    asp = [1, 0]
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
        asp.append(s)
    return tr

rl = np.array([epoch() for _ in range(15)])
rl

round(rl.mean(), 2)

