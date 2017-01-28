import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys
#import mnist.py

from scipy import misc

class ConvolutionalLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    def __init__(self, width, height, depth, filter_num, fsize, stride, zero_padding):
        #width, height = dimensions of input
        #depth = number of inputs
        #filters = number of filters, fsize = side length of filter
        #stride = number of units moved during convolution by filter
        #zero_padding = number of zero "outlines" to surround input with during convolution
        self.width = width
        self.height = height
        self.depth = depth
        self.filter_num = filter_num
        self.fsize = fsize
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = [[np.random.uniform(-1, 1, (self.fsize,self.fsize)) for layer in range(self.depth)] for filter_col in range(self.filter_num)]
        self.bias = np.random.uniform(0, 1, self.filter_num)
        #self.cache = np.zeros((rows,1))
        #self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        #self.mem_weights = np.zeros(self.weights.shape)

        #self.filters =
    def forward(self, inputArr):
        #filters = list of all filters
        #outputs = list(?) of outputs
        o_width = int((self.width - self.fsize)/self.stride) + 1
        o_height = int((self.height - self.fsize)/self.stride) + 1


        output = np.zeros((self.filter_num, o_height, o_width))

        for f in range(self.filter_num):
            for layer in range(self.depth):
                for i in range(o_height):
                    for j in range(o_width):
                        #section = input section (x_ij)
                        section = np.zeros((self.fsize,self.fsize))
                        for m in range(self.fsize):
                            for n in range(self.fsize):
                                section[m][n] = inputArr[m + i*self.stride][n + j*self.stride][layer]

                        output[f][i][j] += np.sum(np.multiply(section, self.filters[f][layer])) #use the proper filter for each one
                    #print(i)
                    #sys.stdout.flush()

        return output





    def backward(self, gradient):
        self.weights -= np.outer(gradient, self.cache.T) * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)

        return np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1]

test_layer = ConvolutionalLayer(820,500,3,2,10,1,0);
img = misc.imread("boi.png")

new_imgs = test_layer.forward(img)

print(new_imgs)

plt.imshow(new_imgs[0], interpolation='nearest')
plt.show()
