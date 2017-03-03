import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys
import os
import random as rand

import mlayers_minibatch as ml

#import mnist.py

#FIX THIS --- Filter back-propagation results in numbers too large; the np.exp in the softmax layer cannot be computed for such large numbers

from scipy import misc, ndimage

EPOCHS = 80*32
ml.LEARN_RATE = 0.1
ml.LEARN_RATE_CONV = 0.001
ml.GRADIENT_THRESHOLD = 100

ml.minibatch_size = 1

ml.debug_mode = False





training_data = []

index = 0

for root, dirnames, filenames in os.walk("overtraining"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = ml.seperate_layers(ndimage.imread(filepath, mode="RGB"))
        training_data.append((index, image))
        index += 1

possible_classifications = len(training_data)

#layers = [ConvolutionalLayer(16,16,1,10,3,1,0), LeakyReLULayer(), MaxPoolingLayer(2,2), FullyConnectedLayer(10,7,7,30), LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 30), ml.SoftmaxLayer()]
#layers = [ConvolutionalLayer(64,64,3,3,7,2,0), ReLULayer(), ConvolutionalLayer(58,58,3,3,5,1,0), ReLULayer(), FullyConnectedLayer(2,7,7,10), ml.InnerLayer(possible_classifications, 10), ml.SoftmaxLayer()]
layers = [ml.ConvolutionalLayer(32,32,1,6,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.ConvolutionalLayer(14,14,6,16,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(16,5,5,20), ml.LeakyReLULayer(), ml.InnerLayerRevised(possible_classifications, 20), ml.SoftmaxLayer()]







error = np.zeros((0,2))


for i in range(EPOCHS):
    samples = [rand.choice(training_data) for sample in range(ml.minibatch_size)]
    #print(sample[1].shape)
    temp = [np.divide(samples[index][1],255) for index in range(ml.minibatch_size)]
    temp = [ml.subsample_layer(temp[index], 0) for index in range(ml.minibatch_size)]

    expected = [np.zeros((possible_classifications, 1)) for classification in range(ml.minibatch_size)]

    for c in range(ml.minibatch_size):
        expected[c][samples[c][0]] = 1

    for layer in layers:
        temp = layer.forward(temp)
        if(ml.debug_mode):
            print("forward pass", layer, np.mean(temp[0]), temp[0].shape)

    #print("average value of weights", np.mean(layers[2].weights), np.mean(layers[3].weights))

    loss = [np.subtract(temp[i], expected[i]) for i in range(ml.minibatch_size)]

    #print(np.argmax(expected), np.argmax(temp))
    if(i%1 == 0):
        print(i, temp[0].T, expected[0].T)

    temp = expected

    layers.reverse()

    for layer in layers:
        temp = layer.backward(temp)
        if(ml.debug_mode):
            print("backprop", layer, np.linalg.norm(temp[0]), temp[0].shape)#, "\n", temp)

    layers.reverse()

    for loss_index in range(len(loss)):
        error = np.append(error, np.absolute(np.array([[i*ml.minibatch_size + loss_index, np.sum(np.abs(loss[loss_index]))]])), axis=0)


plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.show()

for fil_layer in layers[0].filters:
    for fil in fil_layer:
        plt.imshow(fil)
