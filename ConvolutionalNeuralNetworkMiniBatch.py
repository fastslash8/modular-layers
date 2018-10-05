import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys
import os
import time
import random as rand

import mlayers_minibatch as ml

import mnist

#FIX THIS --- Filter back-propagation results in numbers too large; the np.exp in the softmax layer cannot be computed for such large numbers

from scipy import misc, ndimage

if(len(sys.argv) < 5):
    print("Not enough arguments provided.  Try \"python ConvolutionalNeuralNetworkMiniBatch.py \'epochs\' \'minibatch size\' \'filter learn rate\' \'fully connected learn rate\'\"")
    exit()

EPOCHS = int(sys.argv[1])#80*32
ml.LEARN_RATE = float(sys.argv[4])#0.1
ml.LEARN_RATE_CONV = float(sys.argv[3])#0.001
ml.GRADIENT_THRESHOLD = 10000

ml.minibatch_size = int(sys.argv[2])#1

ml.debug_mode = False

np.set_printoptions(threshold=np.inf, precision=4, linewidth=300)



training_data = []

index = 0

for root, dirnames, filenames in os.walk("numbers"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = ml.seperate_layers(ndimage.imread(filepath, mode="RGB"))
        training_data.append((index, image))
        index += 1

possible_classifications = len(training_data)

possible_classifications = 10

#layers = [ml.ConvolutionalLayer(16,16,1,10,3,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(10,7,7,30), ml.LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 30), ml.SoftmaxLayer()]
#layers = [ml.ConvolutionalLayer(28,28,1,20,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(20,12,12,400), ml.LeakyReLULayer(), ml.InnerLayerRevised(80, 400), ml.LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 80), ml.SoftmaxLayer()]
#layers = [ConvolutionalLayer(64,64,3,3,7,2,0), ReLULayer(), ConvolutionalLayer(58,58,3,3,5,1,0), ReLULayer(), FullyConnectedLayer(2,7,7,10), ml.InnerLayer(possible_classifications, 10), ml.SoftmaxLayer()]
layers = [ml.ConvolutionalLayer(28,28,1,6,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.ConvolutionalLayer(12,12,6,16,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(16,4,4,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(40,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 40), ml.SoftmaxLayer()]

#layers = [ml.FullyConnectedLayer(1,28,28,100), ml.ReLULayer(), ml.InnerLayerRevised(30, 100), ml.ReLULayer(), ml.InnerLayerRevised(possible_classifications, 30), ml.SoftmaxLayer()]

training_data, classifications = mnist.load_mnist('training', path = r'C:\Users\fastslash8\OneDrive\Coding\Python\Machine Learning\modular-layers', asbytes=True)

testing_data, test_classifications = mnist.load_mnist('testing', path = r'C:\Users\fastslash8\OneDrive\Coding\Python\Machine Learning\modular-layers', asbytes=True)



error = np.zeros((0,2))


for i in range(EPOCHS):
    #samples = [rand.choice(training_data) for sample in range(ml.minibatch_size)]
    samples = [rand.randint(0,training_data.shape[0] - 1) for sample in range(ml.minibatch_size)]

    temp = [np.divide(training_data[samples[index]].reshape(1,training_data.shape[1],training_data.shape[2]),255/2) - 1 for index in range(ml.minibatch_size)]

    #print(temp[0])
    ml.LEARN_RATE_CONV = 0.001/(1 + (i/1000.0))
    #temp = [ml.subsample_layer(temp[index], 0) for index in range(ml.minibatch_size)]

    #print(temp[0])

    expected = [np.zeros((possible_classifications, 1)) for classification in range(ml.minibatch_size)]

    for c in range(ml.minibatch_size):
        expected[c][classifications[samples[c]] - 1] = 1

    for layer in layers:
        curr_time = time.time()
        temp = layer.forward(temp)
        if(ml.debug_mode):
            print("forward pass", layer, np.mean(temp[0]), temp[0].shape)
            print("Time elapsed during forward pass:", time.time() - curr_time)

    #print("average value of weights", np.mean(layers[2].weights), np.mean(layers[3].weights))

    loss = [np.subtract(temp[i], expected[i]) for i in range(ml.minibatch_size)]

    #print(np.argmax(expected), np.argmax(temp))
    if(i%10 == 0):
        print(i, temp[0].T, expected[0].T)

    temp = expected

    layers.reverse()

    for layer in layers:
        curr_time = time.time()
        temp = layer.backward(temp)
        if(ml.debug_mode):
            print("backprop", layer, np.linalg.norm(temp[0]), temp[0].shape)#, "\n", temp)
            print("Time elapsed during backward pass:", time.time() - curr_time)

    layers.reverse()

    for loss_index in range(len(loss)):
        if((i*ml.minibatch_size + loss_index)%(25*8) == 0):
            error = np.append(error, np.absolute(np.array([[i*ml.minibatch_size + loss_index, np.sum(np.abs(loss[loss_index]))]])), axis=0)


plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.show()







#_______________________________________________________________________________________________



test_num = 5000
successes = 0
error = np.zeros((0,2))


#samples = [rand.choice(training_data) for sample in range(ml.minibatch_size)]
samples = [rand.randint(0,testing_data.shape[0] - 1) for sample in range(test_num)]

temp = [np.divide(testing_data[samples[index]].reshape(1,testing_data.shape[1],testing_data.shape[2]),255/2) - 1 for index in range(test_num)]

#print(temp[0])
#ml.LEARN_RATE_CONV = 0.001/(1 + (i/1000.0))
#temp = [ml.subsample_layer(temp[index], 0) for index in range(ml.minibatch_size)]

#print(temp[0])

expected = [np.zeros((possible_classifications, 1)) for classification in range(test_num)]

for c in range(test_num):
    expected[c][test_classifications[samples[c]] - 1] = 1

for layer in layers:
    temp = layer.forward(temp)
    if(ml.debug_mode):
        print("forward pass", layer, np.mean(temp[0]), temp[0].shape)

#print("average value of weights", np.mean(layers[2].weights), np.mean(layers[3].weights))

loss = [np.subtract(temp[i], expected[i]) for i in range(test_num)]

for i in range(test_num):
    print(np.argmax(temp[i]), np.argmax(expected[i]))
    if(np.argmax(temp[i]) == np.argmax(expected[i])):
        successes += 1


#print(np.argmax(expected), np.argmax(temp))

for loss_index in range(len(loss)):
    error = np.append(error, np.absolute(np.array([[loss_index, np.sum(np.abs(loss[loss_index]))]])), axis=0)


plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

print(float(successes)/test_num * 100, "% Classification Accuracy")

plt.show()
