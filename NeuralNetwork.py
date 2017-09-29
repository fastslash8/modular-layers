import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
import sys
import os
import time
import random as rand

import mlayers_minibatch as ml


class NeuralNetwork():

    def __init__(self, layers, epochs=10000, learn_rate=0.01, minibatch_size=1):
        self.layers = layers
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.minibatch_size = minibatch_size

        self.set_debug_options()

    def set_debug_options(self, show_passes=False, show_loss=False, time_passes=False, sample_frequency=10, graph_error=False):
        self.show_passes = show_passes
        self.show_loss = show_loss
        self.time_passes = time_passes
        self.graph_error = graph_error
        self.sample_frequency = sample_frequency

    def train_network(self, training_data, training_labels, possible_classifications):
        error = np.zeros((0,2))

        samples = [] #[rand.randint(0,training_data.shape[0] - 1) for sample in range(self.minibatch_size)]
        temp = [] #[training_data[samples[index]] for index in range(self.minibatch_size)]
        expected = [] #[np.zeros((possible_classifications, 1)) for classification in range(self.minibatch_size)]

        for i in range(self.epochs):
            for index in range(self.minibatch_size):
                samples.append(rand.randint(0,len(training_data) - 1))
                temp.append(training_data[samples[index]])
                exp_arr = np.zeros((possible_classifications, 1))
                exp_arr[training_labels[samples[index]]] = 1

                expected.append(exp_arr)

            for layer in self.layers:
                curr_time = time.time()
                temp = layer.forward(temp)

                if(i % self.sample_frequency == 0):
                    if(self.show_passes):
                        print("Forward pass", layer, np.mean(temp[0]), temp[0].shape)
                    if(self.time_passes):
                        print("Time elapsed during forward pass:", time.time() - curr_time)

            loss = [np.subtract(temp[l], expected[l]) for l in range(self.minibatch_size)]

            if(self.show_loss and i % self.sample_frequency == 0):
                print(i, temp[0].T, expected[0].T)

            temp = expected
            self.layers.reverse()

            for layer in self.layers:
                curr_time = time.time()
                temp = layer.backward(temp)

                if(i % self.sample_frequency == 0):
                    if(self.show_passes):
                        print("Backward pass", layer, np.mean(temp[0]), temp[0].shape)
                    if(self.time_passes):
                        print("Time elapsed during backward pass:", time.time() - curr_time)

            self.layers.reverse()

            if(self.graph_error and i % self.sample_frequency == 0):
                error = np.append(error, np.absolute(np.array([[i / self.sample_frequency, np.sum(np.abs(loss[0]))]])), axis=0)


    def test_network(self, testing_data, testing_labels, possible_classifications):
        successes = 0
        test_size = testing_labels.shape[0]

        temp = testing_data
        expected = [np.zeros((possible_classifications, 1)) for classification in range(test_size)]
        for c in range(test_size):
            expected[c][classifications[testing_labels[c]]] = 1

        for layer in self.layers:
            temp = layer.forward(temp)

        for i in range(test_num):
            print(np.argmax(temp[i]), np.argmax(expected[i]))
            if(np.argmax(temp[i]) == np.argmax(expected[i])):
                successes += 1

        print(float(successes)/test_num * 100, "% Classification Accuracy")
