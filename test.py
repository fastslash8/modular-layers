from scipy import misc
import numpy as np
import sys
import math

debug_mode = True

LEARN_RATE = 1

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
                        print("coordinate", i, j, layer)
                        section = inputArr[i*self.stride:(i*self.stride + self.fsize):1, j*self.stride:(j*self.stride + self.fsize):1, layer]
                        """
                        for m in range(self.fsize):
                            for n in range(self.fsize):
                                section[m][n] = inputArr[m + i*self.stride][n + j*self.stride][layer]
                        """
                        #print(np.shape(inputArr), np.shape(section), np.shape(self.filters[f][layer]))
                        print(section)
                        output[f][i][j] += np.sum(np.multiply(section, self.filters[f][layer])) + self.bias[f] #use the proper filter for each one
                    #print(i)
                    #sys.stdout.flush()
        return output




    def backward(self, gradient):
        dCdx = np.zeros((self.o_height, self.o_width, self.depth))

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
                                dCdf[i][j] += self.cache[i + m*self.stride][j + n*self.stride][layer] * gradient[f][m][n]
                                self.bias[f] -= LEARN_RATE * gradient[f][m][n]

                                #Rotating filter for convolution
                                dCdx[m][n][layer] += self.filters[f][layer][-i][-j] * gradient[f][m][n]
                if(debug_mode):
                    print("gradient", np.mean(gradient))
                    print("dCdf\n", dCdf)
                self.filters[f][layer] -= LEARN_RATE * dCdf




        return dCdx#np.dot(dCdx, gradient)




test = ConvolutionalLayer(3,3,1,1,1,1,0)

arr = np.array([[[1],[1],[1]],[[1],[1],[1]],[[1],[1],[1]]])

print(arr)
print(test.backward(test.forward(arr)))
#print(test.backward(test.forward(arr)))
