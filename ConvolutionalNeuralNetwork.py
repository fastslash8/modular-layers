import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys


class ConvolutionalLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    def __init__(self, width, height, depth, filters, fsize, stride, zero_padding):
        #width, height = dimensions of input
        #depth = number of inputs
        #filters = number of filters, fsize = side length of filter
        #stride = number of units moved during convolution by filter
        #zero_padding = number of zero "outlines" to surround input with during convolution
        self.width = width
        self.height = height
        self.depth = depth
        self.filters = filters
        self.fsize = fsize
        self.stride = stride
        self.zero_padding = zero_padding


        self.cache = np.zeros((rows,1))
        self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        self.mem_weights = np.zeros(self.weights.shape)

        #self.filters =
    def forward(self, inputArr):
        #filters = list of all filters
        #outputs = list(?) of outputs

        for layer in range(depth):
            for f in filters:
                output = np.multiply(, filters[]) #use the proper filter for each one





    def backward(self, gradient):
        self.weights -= np.outer(gradient, self.cache.T) * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)

        return np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1]
