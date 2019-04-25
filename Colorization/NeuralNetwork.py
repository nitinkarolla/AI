import numpy as np
import math

class NeuralNetwork():
    
    def __init__(self, X = None , y = None, hiddenLayers = 2, neuronsEachLayer = 2):
        self.weights = None
        self.activationHidden = self.sigmoid
        self.activationOut = self.sigmoid
        self.X = X
        self.Y = y
        self.hiddenLayers = hiddenLayers
        self.neuronsEachLayer = neuronsEachLayer

    def weightsInitialisation(self):

        #Initialising a numpy array of dim(hiddenlayers, neurons) to store weights
        self.weights = np.empty((self.hiddenLayers, self.neuronsEachLayer), dtype = object)
        for i in range(self.hiddenLayers):
            for j in range(self.neuronsEachLayer):

                #first hidden layer
                if i == 0:
                    self.weights[i,j] = np.random.normal(0, 0.5, size = 1 + self.X.shape[1])

                #rest hidden layers
                else:
                    self.weights[i,j] = np.random.normal(0, 0.5, size = 1 + self.neuronsEachLayer)

        #Weights for the final output layer
        self.outputLayerWeights =  np.random.normal(0,0.5, size = 1 + self.neuronsEachLayer)
    
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
    
    def squareErrorLoss(self,x,y):
        return (self.feedForward(x) - y)**2
    
    def feedForward(self, x):
        x.append(1.0)  
        for i in range(self.hiddenLayers + 1):
            outputFromCurrLayer = []

            #For first Layer
            if i == 0:
                for j in range(self.neuronsEachLayer):
                    outputFromCurrLayer.append(self.activationHidden(np.dot(self.weights[i,j],x)))
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()

            #output Layer
            elif i == self.hiddenLayers:
                return self.activationOut(np.dot(self.outputLayerWeights, outputFromPrevLayer))

            #Rest all Layers
            else:
                for j in range(self.neuronsEachLayer):
                    outputFromCurrLayer.append(self.activationHidden(np.dot(self.weights[i,j], outputFromPrevLayer)))
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()