# Reinforcement Learning

Watch this video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/kopoLzvh5jY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<hr>

:::{admonition} Assignments:
:class: hint

1. *Individual* -- Find a real-life business application of reinforcement learning
2. *Team* -- Gather all applications and discuss what the shared characteristics are
3. *Team* -- Develop a concept for a new application that is based on reinforcement learning
:::

## An example

We will have the computer play a coin tossing game. The coin is biased and lands 80% of time on heads. First we'll let the computer bet randomly.

We will start by defining the state space which tells what the possible states of the coin can be (1 = heads, 0 = tails). We'll do the same for the action space which is the set of possible bets (1 = bet on heads, 0 = bet on tails).

import numpy as np
ssp = [1, 1, 1, 1, 0]
asp = [1, 0]




Next we'll define a function `epoch()` that plays the game a hundred times and let it run 15 times. As expected the average reward is around 50. 

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

Now we'll let the computer remember the states of the dice by adding them to the action set each time a game has been played.

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

:::{admonition} Assignments:
:class: hint

- *Individual* -- Try to change the parameters and understand the mechanics of the separate functions
:::