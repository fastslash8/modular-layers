from scipy import misc
import numpy as np
import sys
import math
import random as rand


array = np.ones((5,2,2))

newArray = np.pad(array, ((0,0), (0,0), (0,0)), 'constant', constant_values=0)

print(newArray)
