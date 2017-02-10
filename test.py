from scipy import misc
import numpy as np
import sys

class ReLULayer():

    def __init__(self):
        print("kek")
        #self.cache

    def forward(self, inputArr):
        self.cache = np.maximum(inputArr, 0)
        return self.cache

    def backward(self, gradient):
        return np.multiply(np.sign(self.cache), gradient)

arr = np.array([[1,2,3],[-1,-2,-3],[-1,1,-1]])

l = ReLULayer()

print(l.forward(arr))
