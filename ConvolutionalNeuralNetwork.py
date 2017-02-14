import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys
import os
import random as rand

import mlayers as ml

#import mnist.py

#FIX THIS --- Filter back-propagation results in numbers too large; the np.exp in the softmax layer cannot be computed for such large numbers

from scipy import misc, ndimage

EPOCHS = 20000
LEARN_RATE = 0.00001
ml.LEARN_RATE = 0.001
GRADIENT_THRESHOLD = 1

debug_mode = False

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

        self.filters = [[np.random.uniform(0, math.sqrt(2/(self.height * self.width)), (self.fsize,self.fsize)) for layer in range(self.depth)] for filter_col in range(self.filter_num)]
        self.bias = np.random.uniform(0, 1, self.filter_num)
        #self.cache = np.zeros((rows,1))
        #self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        #self.mem_weights = np.zeros(self.weights.shape)

        #self.filters =
    def forward(self, inputArr):
        #filters = list of all filters
        #outputs = list(?) of outputs
        self.cache = inputArr

        self.o_width = int((self.width - self.fsize)/self.stride) + 1
        self.o_height = int((self.height - self.fsize)/self.stride) + 1


        output = np.zeros((self.filter_num, self.o_height, self.o_width))


        for f in range(self.filter_num):
            for layer in range(self.depth):
                if(debug_mode):
                    print("filter\n",self.filters[f][layer])
                    print("bias\n", self.bias[f])

                for i in range(self.o_height):
                    for j in range(self.o_width):
                        #section = input section (x_ij)
                        #section = np.zeros((self.fsize,self.fsize))
                        section = inputArr[layer, i*self.stride:i*self.stride + self.fsize:1, j*self.stride:j*self.stride + self.fsize:1]

                        """
                        for m in range(self.fsize):
                            for n in range(self.fsize):
                                section[m][n] = inputArr[m + i*self.stride][n + j*self.stride][layer]
                        """
                        #print(np.shape(inputArr), np.shape(section), np.shape(self.filters[f][layer]))
                        output[f][i][j] += np.sum(np.multiply(section, self.filters[f][layer])) + self.bias[f] #use the proper filter for each one
                    #print(i)
                    #sys.stdout.flush()
        return output




    def backward(self, gradient):
        dCdx = np.zeros((self.depth, self.height, self.width))

        """
        #Gradient Clipping
        if(np.abs(np.linalg.norm(gradient)) > GRADIENT_THRESHOLD):
            gradient = GRADIENT_THRESHOLD * gradient / np.linalg.norm(gradient)
        """

        for f in range(self.filter_num):
            for layer in range(self.depth):
                dCdf = np.zeros((self.fsize, self.fsize))
                #dzdx = np.zeros((self.o_height, self.o_width))

                for i in range(self.fsize):
                    for j in range(self.fsize):
                        #iteration TODO
                        for m in range(self.o_height):
                            for n in range(self.o_width):
                                dCdf[i][j] += self.cache[layer][i + m*self.stride][j + n*self.stride] * gradient[f][m][n]
                                self.bias[f] -= LEARN_RATE * gradient[f][m][n]

                                #Rotating filter for convolution
                                dCdx[layer][m*self.stride + i][n*self.stride + j] += self.filters[f][layer][-i][-j] * gradient[f][m][n]
                if(f == 0 and debug_mode):
                    #print("gradient\n", np.mean(gradient))
                    print("dCdf\n", dCdf)
                self.filters[f][layer] -= LEARN_RATE * dCdf




        return dCdx#np.dot(dCdx, gradient)


class MaxPoolingLayer():

    def __init__(self, chunk_width, chunk_height, averageValues=False):
        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.averageValues = averageValues

    def forward(self, inputArr):
        self.new_height = int(len(inputArr[0]) / self.chunk_height)
        self.new_width = int(len(inputArr[0][0]) / self.chunk_width)
        self.overhang_h = len(inputArr[0]) % self.chunk_height
        self.overhang_w = len(inputArr[0][0]) % self.chunk_width

        #print(self.new_height, self.new_width, self.overhang_h, self.overhang_w)

        self.depth = len(inputArr)

        pooled_arr = np.zeros((self.depth, self.new_height + np.sign(self.overhang_h), self.new_width + np.sign(self.overhang_w)))

        self.max_positions = [[[np.zeros(2) for x in range(self.new_width + np.sign(self.overhang_w))] for y in range(self.new_height + np.sign(self.overhang_h))] for layer in range(self.depth)]

        for layer in range(self.depth):
            for i in range(self.new_height + np.sign(self.overhang_h)):
                for j in range(self.new_width + np.sign(self.overhang_w)):
                    max_value = 0
                    max_x = 0
                    max_y = 0
                    for m in range(self.chunk_height if (i < self.new_height) else self.overhang_h):
                        for n in range(self.chunk_width if (j < self.new_width) else self.overhang_w):
                            #print("point\n", max_value, layer, i*self.chunk_height + m, j*self.chunk_width + n)
                            if(inputArr[layer][i*self.chunk_height + m][j*self.chunk_width + n] > max_value):
                                max_value = inputArr[layer][i*self.chunk_height + m][j*self.chunk_width + n]
                                max_x = j*self.chunk_width + n
                                max_y = i*self.chunk_height + m
                    pooled_arr[layer][i][j] = max_value
                    self.max_positions[layer][i][j] = np.array([max_x, max_y])
        return pooled_arr

    def backward(self, gradient):
        dCdP = np.zeros((self.depth, self.new_height * self.chunk_height + self.overhang_h, self.new_width * self.chunk_width + self.overhang_w))
        for layer in range(self.depth):
            for i in range(self.new_height + np.sign(self.overhang_h)):
                for j in range(self.new_width + np.sign(self.overhang_w)):
                    #Searching for max value position from input to distribute the error to
                    dCdP[layer][self.max_positions[layer][i][j][1]][self.max_positions[layer][i][j][0]] = gradient[layer][i][j]

        return dCdP


class ReLULayer():

    def __init__(self):
        print("kek")
        #self.cache

    def forward(self, inputArr):
        self.cache = np.maximum(inputArr, 0)
        return self.cache

    def backward(self, gradient):
        #print(np.multiply(np.sign(self.cache), gradient))
        return np.multiply(np.sign(self.cache), gradient)


class LeakyReLULayer():

    def __init__(self):
        print("kek")
        #self.cache

    def forward(self, inputArr):
        self.cache = np.maximum(inputArr, 0.1*inputArr)
        return self.cache

    def backward(self, gradient):
        #print(np.multiply(np.sign(self.cache), gradient))
        return np.multiply(np.sign(self.cache), gradient)



class FullyConnectedLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    def __init__(self, input_depth, input_height, input_width, new_dim):
        #rows = hidden layer size
        #cols = number of unique classifications - size of input vector

        self.old_height = input_height
        self.old_width = input_width

        self.cols = input_height * input_width * input_depth
        self.rows = new_dim
        self.depth = input_depth

        self.cache = np.zeros((self.rows,1))
        self.weights = np.random.uniform(-np.sqrt(1./self.cols), np.sqrt(1./self.cols), (self.rows, self.cols+1))

        self.mem_weights = np.zeros(self.weights.shape)
    def forward(self, inputArr):
        flatArr = np.ndarray.flatten(inputArr)

        self.cache = np.resize(np.append(flatArr, [1]), (len(flatArr) + 1, 1))
        self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2) #incrementing for adagrad

        return np.dot(self.weights, self.cache)

    def backward(self, gradient):

        self.weights -= np.outer(gradient, self.cache.T) * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)

        return np.reshape(np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1], (self.depth, self.old_height, self.old_width))



def subsample_layer(array, layer):
    newArray = np.zeros((1, len(array[0]), len(array[0][0])))
    for i in range(len(array)):
        for j in range(len(array[0])):
            newArray[0][i][j] = array[layer][i][j]

    return newArray

def seperate_layers(array):
    newArray = np.zeros((len(array[0][0]), len(array), len(array[0])))
    for i in range(len(array)):
        for j in range(len(array[0])):
            for k in range(len(array[0][0])):
                newArray[k][i][j] = array[i][j][k]

    return newArray

training_data = []

index = 0

for root, dirnames, filenames in os.walk("pixels"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = seperate_layers(ndimage.imread(filepath, mode="RGB"))
        training_data.append((index, image))
        index += 1

possible_classifications = len(training_data)
#layers = [ConvolutionalLayer(16,16,1,3,3,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), FullyConnectedLayer(3,7,7,30), LeakyReLULayer(), ml.InnerLayer(possible_classifications, 30), ml.SoftmaxLayer()]
#layers = [ConvolutionalLayer(64,64,3,3,7,2,0), ReLULayer(), ConvolutionalLayer(58,58,3,3,5,1,0), ReLULayer(), FullyConnectedLayer(2,7,7,10), ml.InnerLayer(possible_classifications, 10), ml.SoftmaxLayer()]
layers = [ConvolutionalLayer(32,32,1,5,5,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), ConvolutionalLayer(14,14,5,10,5,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), FullyConnectedLayer(10,5,5,20), LeakyReLULayer(), ml.InnerLayer(possible_classifications, 20), ml.SoftmaxLayer()]







error = np.zeros((0,2))


for i in range(EPOCHS):
    sample = rand.choice(training_data)
    #print(sample[1].shape)
    temp = np.divide(sample[1],255)
    temp = subsample_layer(temp, 0)

    expected = np.zeros((possible_classifications, 1))
    expected[sample[0]] = 1

    for layer in layers:
        temp = layer.forward(temp)
        if(debug_mode):
            print("forward pass", layer, np.mean(temp), temp.shape)

    #print("average value of weights", np.mean(layers[2].weights), np.mean(layers[3].weights))

    loss = np.subtract(temp, expected)

    #print(np.argmax(expected), np.argmax(temp))
    if(i%1 == 0):
        print(i, temp.T, expected.T)

    temp = expected

    layers.reverse()

    for layer in layers:
        temp = layer.backward(temp)
        if(debug_mode):
            print("backprop", layer, np.linalg.norm(temp), temp.shape)#, "\n", temp)

    layers.reverse()

    error = np.append(error, np.absolute(np.array([[i, np.sum(np.abs(loss))]])), axis=0)


plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.show()

for fil_layer in layers[0].filters:
    for fil in fil_layer:
        plt.imshow(fil)
