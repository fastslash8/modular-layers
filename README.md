# modular-layers
This project is an exercise in developing various deep learning algorithms such as fully connected networks, convolutional networks, and recurrent networks (LSTM).

### Architecture
Network architecture is defined using a list of 'layer' objects which are defined in mlayers_minibatch.py.

Example: ```python layers = [ml.ConvolutionalLayer(28,28,1,6,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.ConvolutionalLayer(12,12,6,16,5,1,0), ml.LeakyReLULayer(), ml.MaxPoolingLayer(2,2), ml.FullyConnectedLayer(16,4,4,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(40,100), ml.LeakyReLULayer(), ml.InnerLayerRevised(10, 40), ml.SoftmaxLayer()]```

Each layer handles one operation on the input data (for example, the ReLU layer calculates ReLU on each value it takes in).  To allow for both a forward pass and a backward pass, each layer has a ```forward()``` and ```backward()``` method defined, with the forward method passing the result of the function using the data passed in and the backward method passing the running gradient used to calculate the layer specific gradients for backpropagation.  For both passes, the data moves from layer to layer using these methods.  ```NeuralNetwork.py``` handles the training and testing of an arbitrary network, with various options for viewing debug data.

```TrainNetworkMnist.py``` is an example file of the required external code to train and test a CNN on the MNIST dataset.  Normalizing the data significantly boosts performance and is recommended.
