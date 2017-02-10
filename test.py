from scipy import misc
import numpy as np

arr = np.array([[1,2,3],[4,5,6]], [[11,22,33], [44,55,66]])

arr = np.reshape(np.ndarray.flatten(arr), (2,4,3))
print(arr)
