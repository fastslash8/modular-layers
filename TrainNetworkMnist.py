import numpy as np
import mnist
import sys
import mlayers_minibatch as ml
import NeuralNetwork as nn

if(len(sys.argv) < 5):
    print("Not enough arguments provided.  Try \"python ConvolutionalNeuralNetworkMiniBatch.py \'epochs\' \'minibatch size\' \'filter learn rate\' \'fully connected learn rate\'\"")
    exit()

EPOCHS = int(sys.argv[1])#80*32
LEARN_RATE = float(sys.argv[4])#0.1
LEARN_RATE_CONV = float(sys.argv[3])#0.001
ml.GRADIENT_THRESHOLD = 10000
MINIBATCH_SIZE = int(sys.argv[2])#1

np.set_printoptions(threshold=np.inf, precision=4, linewidth=300)

layers = [ml.ConvolutionalLayer(28,28,1,6,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.ConvolutionalLayer(12,12,6,16,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(16,4,4,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(40,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(10, 40), ml.SoftmaxLayer()]


training_data, train_classifications = mnist.load_mnist('training', path = r'C:\Users\fastslash8\OneDrive\Coding\Python\Machine Learning\modular-layers', asbytes=True)

testing_data, test_classifications = mnist.load_mnist('testing', path = r'C:\Users\fastslash8\OneDrive\Coding\Python\Machine Learning\modular-layers', asbytes=True)


training_list = [np.divide(training_data[index].reshape(1,training_data.shape[1],training_data.shape[2]),255/2) - 1 for index in range(training_data.shape[0])]
testing_list = [np.divide(testing_data[index].reshape(1,testing_data.shape[1],testing_data.shape[2]),255/2) - 1 for index in range(testing_data.shape[0])]

train_classifications -= 1
test_classifications -= 1

network = nn.NeuralNetwork(layers, epochs=EPOCHS, learn_rate=LEARN_RATE, minibatch_size=MINIBATCH_SIZE)

network.set_debug_options(show_loss=True)

network.train_network(training_list, train_classifications, 10)
network.test_network(testing_list, test_classifications, 10)
