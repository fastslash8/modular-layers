import numpy as np


LEARN_RATE = 0.001
GRADIENT_THRESHOLD = 1000

class RecurrentInnerLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    rcache = np.array([0])  #Used to store and pass the recurrent values through time iterations
    rweights = np.array([0]) #Weights for rcache

    rprecache = np.array([0]) #Used to store the rcache values from the previous time iteration

    dLdW = np.array([0]) #Used in back-propagation of weights
    dLdU = np.array([0]) #Used in back-propagation of rweights
    rgradient = np.array([0])

    def __init__(self, rows, cols):
        #rows = hidden layer size
        #cols = number of unique classifications - size of input vector

        #Initalize all values

        self.cache = np.zeros((rows,1))
        self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        self.rcache = np.zeros((rows, 1))
        self.rprecache = np.zeros((rows, 1))
        self.rweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, rows+1))

        self.dLdW = np.zeros(self.rweights.shape)
        self.dLdU = np.zeros(self.weights.shape)
        self.rgradient = np.zeros((rows, 1))

        self.mem_weights = np.zeros(self.weights.shape) #weights memory for Adagrad
        self.mem_rweights = np.zeros(self.rweights.shape) #rweights memory for Adagrad

    def forward(self, inputArr):
        self.cache = np.resize(np.append(inputArr, [1]), (len(inputArr) + 1, 1))
        self.rcache = np.resize(np.append(self.rcache, [1]), (len(self.rcache) + 1, 1))

        output = np.dot(self.weights, self.cache) + np.dot(self.rweights, self.rcache)

        self.rcache = np.tanh(output)

        self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2) #incrementing for adagrad
        self.mem_rweights = 0.9*self.mem_rweights + 0.1*(self.rweights ** 2)

        return output #forward pass

    def backward(self, gradient):
        self.dLdW = np.outer(gradient, np.resize(np.append(self.rprecache, [1]), (len(self.rprecache) + 1, 1)).T)
        self.dLdU = np.outer(gradient, self.cache)#np.outer(gradient, np.reshape(np.append(self.cache, [1]), (len(self.cache) + 1, 1)))

        #Back Propagation

        self.rweights -= self.dLdW * LEARN_RATE / np.sqrt(self.mem_rweights + 1e-8)
        self.weights -= self.dLdU * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)

        self.rprecache = self.rcache

        return np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1]

    def clear_rcache(self):
        self.dLdW = np.zeros(self.dLdW.shape)
        self.dLdU = np.zeros(self.dLdU.shape)
        self.rgradient = np.zeros(self.rgradient.shape)
        self.rcache = np.zeros(self.rcache.shape)

        self.mem_weights = np.zeros(self.mem_weights.shape)
        self.mem_rweights = np.zeros(self.mem_rweights.shape)

class InnerLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    def __init__(self, rows, cols):
        #rows = hidden layer size
        #cols = number of unique classifications - size of input vector

        self.cache = np.zeros((rows,1))
        self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        self.mem_weights = np.zeros(self.weights.shape)
    def forward(self, inputArr):
        self.cache = np.resize(np.append(inputArr, [1]), (len(inputArr) + 1, 1))
        self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2) #incrementing for adagrad

        return np.dot(self.weights, self.cache)

    def backward(self, gradient):
        #Gradient Clipping
        if(np.abs(np.linalg.norm(gradient)) > GRADIENT_THRESHOLD):
            gradient = GRADIENT_THRESHOLD * gradient / np.linalg.norm(gradient)

        self.weights -= np.outer(gradient, self.cache.T) * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)

        return np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1]

class InnerLayerRevised():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    def __init__(self, rows, cols):
        #rows = hidden layer size
        #cols = number of unique classifications - size of input vector

        self.cache = np.zeros((rows,1))
        self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols+1))

        self.mem_weights = np.zeros(self.weights.shape)
    def forward(self, inputData):
        self.cache = []
        outputs = []

        for data in inputData:
            augmented_data = np.resize(np.append(data, [1]), (len(data) + 1, 1))
            self.cache.append(augmented_data)
            self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2) #incrementing for adagrad

            outputs.append(np.dot(self.weights, augmented_data))

        return outputs


    def backward(self, gradients):

        """
        #Gradient Clipping
        if(np.abs(np.linalg.norm(gradient)) > GRADIENT_THRESHOLD):
            gradient = GRADIENT_THRESHOLD * gradient / np.linalg.norm(gradient)
        """
        dCdw = np.zeros(self.weights.shape)
        dCdz = []

        for grad in range(len(gradients)):
            gradient = gradients[grad]
            dCdw += np.outer(gradient, self.cache[grad].T) / np.sqrt(self.mem_weights + 1e-8)

            dCdz.append(np.dot(self.weights.T, gradient)[:len(np.dot(self.weights.T, gradient)) - 1])

        if(np.abs(np.linalg.norm(dCdw)) > GRADIENT_THRESHOLD):
            print("gradient of", np.linalg.norm(dCdw), "was clipped")
            dCdw = GRADIENT_THRESHOLD * dCdw / np.linalg.norm(dCdw)


        #print("dCdw\n", dCdw)

        self.weights -= LEARN_RATE * dCdw

        return dCdz


class TanhLayer():
    cache = np.array([0])  #Used to store the values for back-propagation

    def forward(self, inputArr):
        self.cache = np.tanh(inputArr)

        return self.cache.copy()

    def backward(self, gradient):#nx1 * nx1

        return gradient * (1 - (self.cache ** 2))

    def sigmoid(self, inputArr):
        #Sigmoid Function
        return 1/(1 + np.exp(-1 * inputArr))

class SoftmaxLayer():
    def forward(self, inputData):
        #print(inputArr)
        #temp = inputArr - np.max(inputArr)
        outputs = []

        for data in inputData:
            data_adjusted = data - np.max(data)
            outputs.append(np.exp(data_adjusted)/np.sum(np.exp(data_adjusted)))

        self.cache = outputs[:]
        return outputs

    def backward(self, expectedValues):
        self.loss = []

        for e in range(len(expectedValues)):
            self.loss.append(np.subtract(self.cache[e], expectedValues[e]))

        return self.loss

    def clear_loss(self):
        self.loss = np.zeros(self.loss.shape)

class GRUInnerLayer():
    cache = np.array([0])  #Used to store the values for back-propagation
    weights = np.array([0])  #Weights for each connection between neurons represented as a matrix

    rcache = np.array([0])
    rweights = np.array([0])

    rprecache = np.array([0])

    dLdW = np.array([0])
    dLdU = np.array([0])
    rgradient = np.array([0])

    def __init__(self, rows, cols):
        #rows = hidden layer size - size of output
        #cols = number of unique classifications - size of input vector

        self.cache = np.zeros((rows,1))

        self.weights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols)) #reset calculation weights
        self.cweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols)) #candidate calculation weights
        self.zweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, cols)) #update calculation weights


        self.rcache = np.zeros((rows, 1))
        self.rprecache = np.zeros((rows, 1))

        self.rweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, rows)) #reset calculation memory weights
        self.crweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, rows)) #candidate calculation memory weights
        self.zrweights = np.random.uniform(-np.sqrt(1./cols), np.sqrt(1./cols), (rows, rows)) #update calculation memory weights



        self.dLdU = np.zeros(self.rweights.shape)
        self.dLdW = np.zeros(self.weights.shape)

        #self.rgradient = np.zeros((rows, 1))

        self.mem_weights = np.zeros(self.weights.shape) #weights memory for Adagrad
        self.mem_cweights = np.zeros(self.cweights.shape) #cweights memory for Adagrad
        self.mem_zweights = np.zeros(self.zweights.shape) #zweights memory for Adagrad

        self.mem_rweights = np.zeros(self.rweights.shape) #rweights memory for Adagrad
        self.mem_crweights = np.zeros(self.crweights.shape) #rweights memory for Adagrad
        self.mem_zrweights = np.zeros(self.zrweights.shape) #rweights memory for Adagrad

    def forward(self, inputArr):
        self.cache = inputArr

        self.reset = self.sigmoid(np.dot(self.weights, self.cache) + np.dot(self.rweights, self.rcache))

        np.multiply(self.reset, self.rcache)

        self.candidate = np.tanh(np.dot(self.cweights, self.cache) + np.dot(self.crweights, np.multiply(self.reset, self.rcache)))

        self.update = self.sigmoid(np.dot(self.zweights, self.cache) + np.dot(self.zrweights, self.rcache))

        self.rcache = np.multiply((1 - self.update), self.rcache) + np.multiply(self.update, self.candidate)

        #incrementing for adagrad

        self.mem_weights = 0.9*self.mem_weights + 0.1*(self.weights ** 2)
        self.mem_cweights = 0.9*self.mem_cweights + 0.1*(self.cweights ** 2)
        self.mem_zweights = 0.9*self.mem_zweights + 0.1*(self.zweights ** 2)

        self.mem_rweights = 0.9*self.mem_rweights + 0.1*(self.rweights ** 2)
        self.mem_crweights = 0.9*self.mem_crweights + 0.1*(self.crweights ** 2)
        self.mem_zrweights = 0.9*self.mem_zrweights + 0.1*(self.zrweights ** 2)

        return self.rcache

    def backward(self, gradient):
        dhdz = -self.rprecache + self.candidate #np.dot(self.weights, gradient) * self.rcache
        dhdhc = self.update #hc = h candidate

        dhcdr = np.multiply((1 - self.candidate ** 2), (np.dot(self.crweights, self.rprecache)))

        dLdWr = np.outer(gradient * self.update * dhcdr * self.reset * (1 - self.reset), self.cache)
        dLdUr = np.outer(gradient * self.update * dhcdr * self.reset * (1 - self.reset), self.rprecache)

        dLdW = np.outer(gradient * (1 - self.candidate ** 2), self.cache)
        dLdU = np.outer(gradient * (1 - self.candidate ** 2), np.multiply(self.reset, self.rprecache))

        dLdWz = np.outer(gradient * dhdz * self.update * (1 - self.update), self.cache)
        dLdUz = np.outer(gradient * dhdz * self.update * (1 - self.update), self.rprecache)

        self.weights -= dLdWr * LEARN_RATE / np.sqrt(self.mem_weights + 1e-8)
        self.cweights -= dLdW * LEARN_RATE / np.sqrt(self.mem_cweights + 1e-8)
        self.zweights -= dLdWz * LEARN_RATE / np.sqrt(self.mem_zweights + 1e-8)

        self.rweights -= dLdUr * LEARN_RATE / np.sqrt(self.mem_rweights + 1e-8)
        self.crweights -= dLdU * LEARN_RATE / np.sqrt(self.mem_crweights + 1e-8)
        self.zrweights -= dLdUz * LEARN_RATE / np.sqrt(self.mem_zrweights + 1e-8)

        self.rprecache = self.rcache

    def clear_rcache(self):
        self.dLdW = np.zeros(self.dLdW.shape)
        self.dLdU = np.zeros(self.dLdU.shape)
        self.rprecache = np.zeros(self.rprecache.shape)
        self.rcache = np.zeros(self.rcache.shape)

        self.mem_weights = np.zeros(self.mem_weights.shape)
        self.mem_cweights = np.zeros(self.mem_cweights.shape)
        self.mem_zweights = np.zeros(self.mem_zweights.shape)

        self.mem_rweights = np.zeros(self.mem_rweights.shape)
        self.mem_crweights = np.zeros(self.mem_crweights.shape)
        self.mem_zrweights = np.zeros(self.mem_zrweights.shape)

    def sigmoid(self, inputArr):
        return 1/(1 + np.exp(-inputArr))


class SigmoidLayer():
    #Save cache of output values, which are equal to sigmoid(input)
    cache = np.array([0])

    def sigmoid(self, inputArr):
        #Sigmoid Function
        return 1/(1 + np.exp(-1 * inputArr))

    def forward(self, inputArr):
        #Take sigmoid of input array and output it
        self.cache = self.sigmoid(inputArr.copy())
        return self.cache.copy()

    def backward(self, gradient):
        #Returning the sigmoid prime of the input from the forward pass (equal to sigmoid() * (1-sigmoid())) times the gradient

        gradient = (self.cache * (1 - self.cache)) * gradient
        return gradient

class CrossEntropyLossLayer():
    cache = np.array([0])

    def forward(self, inputArr):
        #Outputs the result of the forward pass
        self.cache = inputArr
        return self.cache;

    def backward(self, expectedValue):
        return np.subtract(self.cache, expectedValue)
