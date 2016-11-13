"""Softmax."""
"""First Python script"""

import numpy as np
from math import exp

#scores = [3.0, 1.0, 0.2]
#scores = [1.0, 2.0, 3.0]
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    softmaxScores = np.array([])
    
    expScores = np.array([])
    
    """Compute the exponential """
    expScores = np.array([np.exp(i) for i in scores])
    
    #print expScores[:,1]
    sumSoft = np.sum(expScores[:,z] for z in range(3))
    
    #print sumSoft
    
    softmaxScores = np.exp(x)/np.sum(np.exp(x), axis = 0)
    
    return softmaxScores
    #pass  # TODO: Compute and return softmax(x)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
