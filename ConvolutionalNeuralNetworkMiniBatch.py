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

EPOCHS = 10000
minibatch_size = 64

LEARN_RATE = 0.01/minibatch_size
ml.LEARN_RATE = 0.01/minibatch_size
GRADIENT_THRESHOLD = 1000
ml.GRADIENT_THRESHOLD = 1000



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

        self.filters = [[np.random.uniform(-10*math.sqrt(1/(self.height * self.width)), math.sqrt(10/(self.height * self.width)), (self.fsize,self.fsize)) for layer in range(self.depth)] for filter_col in range(self.filter_num)]
        self.bias = np.random.uniform(-math.sqrt(1/(self.height * self.width)), math.sqrt(1/(self.height * self.width)), self.filter_num)
        #self.cache = np.zeros((rows,1))
        #self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        #self.mem_weights = np.zeros(self.weights.shape)

        #self.filters =
    def forward(self, inputData):
        #filters = list of all filters
        #outputs = list(?) of outputs
        inputData = [np.pad(arr, ((0,0), (self.zero_padding,self.zero_padding), (self.zero_padding,self.zero_padding)), 'constant', constant_values=0) for arr in inputData]

        self.cache = inputData

        self.o_width = int((self.width - self.fsize)/self.stride) + 1
        self.o_height = int((self.height - self.fsize)/self.stride) + 1


        outputs = []#[np.zeros((self.filter_num, self.o_height, self.o_width)) for output in len(inputData)]

        for data in inputData:
            output = np.zeros((self.filter_num, self.o_height, self.o_width))

            for f in range(self.filter_num):
                for layer in range(self.depth):
                    #if(debug_mode):
                        #print("filter\n",self.filters[f][layer])
                        #print("bias\n", self.bias[f])

                    for i in range(self.o_height):
                        for j in range(self.o_width):
                            #section = input section (x_ij)
                            #section = np.zeros((self.fsize,self.fsize))
                            section = data[layer, i*self.stride:i*self.stride + self.fsize:1, j*self.stride:j*self.stride + self.fsize:1]

                            """
                            for m in range(self.fsize):
                                for n in range(self.fsize):
                                    section[m][n] = inputArr[m + i*self.stride][n + j*self.stride][layer]
                            """
                            #print(np.shape(inputArr), np.shape(section), np.shape(self.filters[f][layer]))
                            output[f][i][j] += np.sum(np.multiply(section, self.filters[f][layer])) + self.bias[f] #use the proper filter for each one
                        #print(i)
                        #sys.stdout.flush()
            outputs.append(output)

        return outputs




    def backward(self, gradients):
        dCdx_list = []
        dCdf = [[np.zeros((self.fsize, self.fsize)) for layer in range(self.depth)] for f in range(self.filter_num)]
        dCdb = np.zeros(self.filter_num)

        for grad in range(len(gradients)):
            dCdx = np.zeros((self.depth, self.height, self.width))
            gradient = gradients[grad]



        """dCdx_list = []
        dCdf = [[np.zeros((self.fsize, self.fsize)) for layer in range(self.depth)] for f in range(self.filter_num)]
        dCdb = np.zeros(self.filter_num)

        for grad in range(len(gradients)):
            dCdx = np.zeros((self.depth, self.height, self.width))
            gradient = gradients[grad]
            """
            """
            #Gradient Clipping
            if(np.abs(np.linalg.norm(gradient)) > GRADIENT_THRESHOLD):
                gradient = GRADIENT_THRESHOLD * gradient / np.linalg.norm(gradient)
            """
            """
            for f in range(self.filter_num):
                for layer in range(self.depth):
                    #dCdf = np.zeros((self.fsize, self.fsize))
                    #dzdx = np.zeros((self.o_height, self.o_width))

                    for i in range(self.fsize):
                        for j in range(self.fsize):
                            #iteration TODO
                            for m in range(self.o_height):
                                for n in range(self.o_width):
                                    dCdf[f][layer][i][j] += self.cache[grad][layer][i + m*self.stride][j + n*self.stride] * gradient[f][m][n]
                                    dCdb[f] += gradient[f][m][n]

                                    #Rotating filter for convolution
                                    dCdx[layer][m*self.stride + i][n*self.stride + j] += self.filters[f][layer][-i][-j] * gradient[f][m][n]

                    #if(f == 0 and debug_mode):
                        #print("gradient\n", np.mean(gradient))
                        #print("dCdf\n", dCdf[f][layer])


            dCdx_list.append(dCdx)#np.dot(dCdx, gradient)

        for f in range(self.filter_num):
            #if(np.abs(np.linalg.norm(dCdb[f])) > GRADIENT_THRESHOLD):
                #dCdb[f] = GRADIENT_THRESHOLD * dCdb[f] / np.linalg.norm(dCdb[f])

            self.bias[f] -= LEARN_RATE * dCdb[f]
            for layer in range(self.depth):
                if(np.abs(np.linalg.norm(dCdf[f][layer])) > GRADIENT_THRESHOLD):
                    print("gradient of", np.linalg.norm(dCdf[f][layer]), "was clipped")
                    #dCdf[f][layer] = GRADIENT_THRESHOLD * dCdf[f][layer] / np.linalg.norm(dCdf[f][layer])

                    #print("dCdf\n", dCdf[f][layer])

                self.filters[f][layer] -= LEARN_RATE * dCdf[f][layer]

        return dCdx_list"""





class MaxPoolingLayer():

    def __init__(self, chunk_width, chunk_height, averageValues=False):
        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.averageValues = averageValues

    def forward(self, inputData):
        old_height = len(inputData[0][0])
        old_width = len(inputData[0][0][0])

        self.new_height = int(old_height / self.chunk_height)
        self.new_width = int(old_width / self.chunk_width)
        self.overhang_h = old_height % self.chunk_height
        self.overhang_w = old_width % self.chunk_width

        #print(self.new_height, self.new_width, self.overhang_h, self.overhang_w)

        self.depth = len(inputData[0])

        outputs = []
        self.max_arrays = []

        for data in inputData:
            pooled_arr = np.zeros((self.depth, self.new_height + np.sign(self.overhang_h), self.new_width + np.sign(self.overhang_w)))

            max_positions = [[[np.zeros(2) for x in range(self.new_width + np.sign(self.overhang_w))] for y in range(self.new_height + np.sign(self.overhang_h))] for layer in range(self.depth)]

            for layer in range(self.depth):
                for i in range(self.new_height + np.sign(self.overhang_h)):
                    for j in range(self.new_width + np.sign(self.overhang_w)):
                        max_value = 0
                        max_x = 0
                        max_y = 0
                        for m in range(self.chunk_height if (i < self.new_height) else self.overhang_h):
                            for n in range(self.chunk_width if (j < self.new_width) else self.overhang_w):
                                #print("point\n", max_value, layer, i*self.chunk_height + m, j*self.chunk_width + n)
                                if(data[layer][i*self.chunk_height + m][j*self.chunk_width + n] > max_value):
                                    max_value = data[layer][i*self.chunk_height + m][j*self.chunk_width + n]
                                    max_x = j*self.chunk_width + n
                                    max_y = i*self.chunk_height + m
                        pooled_arr[layer][i][j] = max_value
                        max_positions[layer][i][j] = np.array([max_x, max_y])
            outputs.append(pooled_arr)
            self.max_arrays.append(max_positions)

        return outputs

    def backward(self, gradients):
        dCdP_list = []

        for grad in range(len(gradients)):
            gradient = gradients[grad]

            dCdP = np.zeros((self.depth, self.new_height * self.chunk_height + self.overhang_h, self.new_width * self.chunk_width + self.overhang_w))
            for layer in range(self.depth):
                for i in range(self.new_height + np.sign(self.overhang_h)):
                    for j in range(self.new_width + np.sign(self.overhang_w)):
                        #Searching for max value position from input to distribute the error to
                        dCdP[layer][self.max_arrays[grad][layer][i][j][1]][self.max_arrays[grad][layer][i][j][0]] = gradient[layer][i][j]

            dCdP_list.append(dCdP)

        return dCdP_list


class ReLULayer():

    def __init__(self):
        print("kek")
        #self.cache

    def forward(self, inputData):
        outputs = []

        for data in inputData:
            output = np.maximum(data, 0)
            outputs.add(output)

        self.cache = outputs
        return outputs

    def backward(self, gradients):
        #print(np.multiply(np.sign(self.cache), gradient))
        outputs = []

        for grad in range(len(gradients)):
            gradient = gradients[grad]

            outputs.append(np.multiply(np.sign(self.cache[grad]), gradient))

        return outputs


class LeakyReLULayer():

    def __init__(self):
        print("kek")
        #self.cache

    def forward(self, inputData):
        outputs = []

        for data in inputData:
            output = np.maximum(data, 0.1*data)
            outputs.append(output)

        self.cache = outputs
        return outputs

    def backward(self, gradients):
        #print(np.multiply(np.sign(self.cache), gradient))
        outputs = []

        for grad in range(len(gradients)):
            gradient = gradients[grad]

            outputs.append(np.multiply(np.sign(self.cache[grad]), gradient))

        return outputs



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
    def forward(self, inputData):

        self.cache = []
        outputs = []

        for data in inputData:
            flatArr = np.ndarray.flatten(data)

            augmented_data = np.resize(np.append(flatArr, [1]), (len(flatArr) + 1, 1))
            self.cache.append(augmented_data)
            self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2) #incrementing for adagrad

            outputs.append(np.dot(self.weights, augmented_data))

        return outputs


    def backward(self, gradients):
        dCdw = np.zeros(self.weights.shape)
        dCdz = []

        for grad in range(len(gradients)):
            gradient = gradients[grad]
            dCdw += np.outer(gradient, self.cache[grad].T) / np.sqrt(self.mem_weights + 1e-8)

            dCdz.append(np.reshape(np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1], (self.depth, self.old_height, self.old_width)))

        if(np.abs(np.linalg.norm(dCdw)) > GRADIENT_THRESHOLD):
            print("gradient of", np.linalg.norm(dCdw), "was clipped")
            dCdw = GRADIENT_THRESHOLD * dCdw / np.linalg.norm(dCdw)


        #print("dCdw\n", dCdw)
        self.weights -= ml.LEARN_RATE * dCdw

        return dCdz



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

for root, dirnames, filenames in os.walk("numbers"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = seperate_layers(ndimage.imread(filepath, mode="RGB"))
        training_data.append((index, image))
        index += 1

possible_classifications = len(training_data)

#layers = [ConvolutionalLayer(16,16,1,10,3,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), FullyConnectedLayer(10,7,7,30), LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 30), ml.SoftmaxLayer()]
#layers = [ConvolutionalLayer(32,32,1,32,5,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), FullyConnectedLayer(32,14,14,30), LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 30), ml.SoftmaxLayer()]
#layers = [ConvolutionalLayer(64,64,3,3,7,2,0), ReLULayer(), ConvolutionalLayer(58,58,3,3,5,1,0), ReLULayer(), FullyConnectedLayer(2,7,7,10), ml.InnerLayer(possible_classifications, 10), ml.SoftmaxLayer()]
layers = [ConvolutionalLayer(32,32,1,6,5,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), ConvolutionalLayer(14,14,6,16,5,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), ConvolutionalLayer(5,5,16,12,5,1,0), LeakyReLULayer(), FullyConnectedLayer(12,1,1,60), LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 60), ml.SoftmaxLayer()]







error = np.zeros((0,2))


for i in range(EPOCHS):
    samples = [rand.choice(training_data) for sample in range(minibatch_size)]
    #print(sample[1].shape)
    temp = [np.divide(samples[index][1],255) for index in range(minibatch_size)]
    temp = [subsample_layer(temp[index], 0) for index in range(minibatch_size)]

    expected = [np.zeros((possible_classifications, 1)) for classification in range(minibatch_size)]

    for c in range(minibatch_size):
        expected[c][samples[c][0]] = 1

    for layer in layers:
        temp = layer.forward(temp)
        if(debug_mode):
            print("forward pass", layer, np.mean(temp[0]), temp[0].shape)

    #print("average value of weights", np.mean(layers[2].weights), np.mean(layers[3].weights))

    loss = [np.subtract(temp[i], expected[i]) for i in range(minibatch_size)]

    #print(np.argmax(expected), np.argmax(temp))
    if(i%1 == 0):
        print(i, temp[0].T, expected[0].T)

    temp = expected

    layers.reverse()

    for layer in layers:
        temp = layer.backward(temp)
        if(debug_mode):
            print("backprop", layer, np.linalg.norm(temp[0]), temp[0].shape)#, "\n", temp)

    layers.reverse()

    for loss_index in range(len(loss)):
        error = np.append(error, np.absolute(np.array([[i*minibatch_size + loss_index, np.sum(np.abs(loss[loss_index]))]])), axis=0)


plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.show()

for fil_layer in layers[0].filters:
    for fil in fil_layer:
        plt.imshow(fil)
