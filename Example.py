import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys

import mlayers as ml

#-*- coding: utf-8 -*-

ml.LEARN_RATE = 0.001 #Learn rate used in back-propagation

EPOCHS = 20000  #Number of training data points

#Not recommended to use full array of
#characters = ' !"#$%&\'()*+,-—./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t�àáâäëéêíÀóúèîôç…ïý'

source = "Data.txt"

f = open(source, "r").read()
fl = open(source, "r")

lines = []

for line in fl:
    lines.append(line)
    print("line:", line)

characters = list(set(f))

print(characters)



layerNum = [len(characters),100,len(characters)]  #Number of neurons in each layer


layers = [ml.GRUInnerLayer(layerNum[1], layerNum[0]), ml.TanhLayer(), ml.InnerLayer(layerNum[2], layerNum[1]), ml.SoftmaxLayer()]


error = np.zeros((1,2))



for i in range(EPOCHS):
    #count = 0
    for line in lines:
        phrase = line[0]
        word = line
        loss = np.zeros((len(characters), 1))

        for j in range(len(word) - 1):
            #Create inital numpy array with input data and constant bias value, to begin the forward pass
            temp = np.zeros((1,len(characters)))
            temp[0][characters.index(word[j])] = 1
            temp = temp.T


            #Foward pass between layers

            for layer in layers:
                temp = layer.forward(temp)
                sys.stdout.flush()

            expected = np.zeros((len(characters), 1))
            expected[characters.index(word[j+1])][0] = 1;

            #Print output value and expected value
            phrase = phrase + characters[temp.argmax()]

            #Begin backward pass with expected value
            loss += np.subtract(temp, expected)
            temp = expected

            #Flip layer list for backward pass
            layers.reverse()

            #Backward pass between layers
            for layer in layers:
                temp = layer.backward(temp)

            #Reflip layer list for forward pass
            layers.reverse()

        error = np.append(error, np.absolute(np.array([[i, np.sum(loss)]])), axis=0)

        if(i % 500 == 0):  #Print gradual results every 500 iterations
            if(i != EPOCHS - 2):
                print(i, "- LINE:", phrase)
                sys.stdout.flush()
        #count += 1
        layers[0].clear_rcache()


"""
Uncomment to generate matplotlib graph of error over time after training is complete

plt.plot(error[:,0], error[:,1])
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.show()
"""

char = lines[0][0] #First character used in generating text
string = ""  #String to concatenate each generated character

print("characters", characters)

for i in range(1000):
    #if(i%10 == 0):
        #layers[0].clear_rcache()

    string += char

    temp = np.zeros((1,len(characters)))
    temp[0][characters.index(char)] = 1
    temp = temp.T

    #Foward pass between layers
    for layer in layers:
        temp = layer.forward(temp)

    char = characters[np.random.choice(range(len(characters)), p=temp.ravel())]

    if(i == 999):
        print(layers[0].weights)

print(string)
