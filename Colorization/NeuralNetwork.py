import numpy as np
import math
from statistics import mean 

class NeuralNetwork():
    
    def __init__(self, X = None , y = None, hiddenLayers = 2, neuronsEachLayer = 2, learning_rate = 0.01, epochs = 5, method = 'Linear'):
        self.weights = None
        self.activationHidden = self.sigmoid
        if method == 'Linear':
            self.activationOut = self.linear
            self.derivate_out = self.linear_der
        elif method == 'Logistic':
            self.activationOut = self.sigmoid
            self.derivate_out = self.sigmoid_der
        self.X = X
        self.Y = y
        self.hiddenLayers = hiddenLayers
        self.neuronsEachLayer = neuronsEachLayer
        self.learning_rate = learning_rate
        self.epochs = epochs

    def weightsInitialisation(self):
        #Initialising a numpy array of dim(hiddenlayers, neurons) to store weights
        self.weights = np.empty((self.hiddenLayers, self.neuronsEachLayer), dtype = object)
        for i in range(self.hiddenLayers):
            for j in range(self.neuronsEachLayer):
                #first hidden layer
                if i == 0:
                    self.weights[i,j] = np.random.normal(0,0.5, size = 1 + self.X.shape[1])
                #rest hidden layers
                else:
                    self.weights[i,j] = np.random.normal(0,0.5, size = 1 + self.neuronsEachLayer)
        #Weights for the final output layer
        self.outputLayerWeights =  np.random.normal(0,0.5, size = 1 + self.neuronsEachLayer)
    
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
    
    def linear(self,x):
        return x
    
    def sigmoid_der(self,x):  
        return self.sigmoid(x) *(1 - self.sigmoid (x))

    def linear_der(self, x):
        return 1.0
    
    def squareErrorLoss(self,x,y):
        return (self.feedForward(x) - y)**2
    
    def error(self, X, y):
        pred= []
        for i in X:
            pred.append(self.feedForward(i))
        return mean([(a_i - b_i)**2 for a_i, b_i in zip(pred, y)])
    
    def predict(self,X):
        pred = []
        for i in X:
            pred.append(self.feedForward(i))
        return pred

    def feedForward(self, x):
        self.x = np.append(x, 1.0)
        self.out = np.empty(shape = (self.hiddenLayers, self.neuronsEachLayer))
        for i in range(self.hiddenLayers + 1):
            outputFromCurrLayer = []
            #For first Layer
            if i == 0:
                for j in range(self.neuronsEachLayer):
                    z = self.activationHidden(np.dot(self.weights[i,j],self.x))
                    self.out[i,j] = z
                    outputFromCurrLayer.append(z)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()
            #Output Layer
            elif i == self.hiddenLayers:
                return self.activationOut(np.dot(self.outputLayerWeights,outputFromPrevLayer))
            #Rest all Layers
            else:
                for j in range(self.neuronsEachLayer):
                    z = self.activationHidden(np.dot(self.weights[i,j],outputFromPrevLayer))
                    self.out[i,j] = z
                    outputFromCurrLayer.append(z)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()

    def backProp(self, pred, n, actual):
        #Weight updation for Output Layer
        delta = []
        for i in range(len(self.outputLayerWeights)):
            if i == len(self.outputLayerWeights) - 1:
                self.outputLayerWeights[i] = self.outputLayerWeights[i] - self.learning_rate * (2.0 / n) * (pred- actual) *self.derivate_out(pred) * 1 
            else :
                d = (2.0 / n) * (pred- actual) * self.derivate_out(pred) * self.out[self.hiddenLayers -1, i]
                self.outputLayerWeights[i] = self.outputLayerWeights[i] - self.learning_rate * d
                delta.append(d)
        
        #For all other Layers
        #curr_weights = self.weights.copy()
        for l in reversed(range(self.hiddenLayers)):
            if np.array(delta).ndim == 1:
                delta_forward = delta.copy()
            else:
                delta_forward = np.array(delta).sum(axis = 0)
            delta = []
            if l == 0 :
                for j in range(self.neuronsEachLayer):
                    #weight = self.weights[l,j].copy()
                    for i in range(len(self.weights[l,j])):
                        if i == len(self.weights[l,j]) - 1:
                            self.weights[l,j][i] = self.weights[l,j][i] - self.learning_rate * (1.0 / n) * delta_forward[j] * self.sigmoid_der(self.out[l, j]) * 1.0
                        else :
                            d = (1.0 / n) * delta_forward[j] * self.sigmoid_der(self.out[l, j]) * self.x[i]
                            self.weights[l,j][i] = self.weights[l,j][i] - self.learning_rate * d
            else :
                for j in range(self.neuronsEachLayer):
                    #weight = self.weights[l,j].copy()
                    temp = []
                    for i in range(len(self.weights[l,j])):
                        if i == len(self.weights[l,j]) - 1:
                            self.weights[l,j][i] = self.weights[l,j][i] - self.learning_rate * (1.0 / n) * delta_forward[j] * self.sigmoid_der(self.out[l, j]) * 1.0
                        else :
                            d = (1.0 / n) * delta_forward[j] * self.sigmoid_der(self.out[l, j]) * self.out[l -1, i]
                            self.weights[l,j][i] = self.weights[l,j][i] - self.learning_rate * d
                            temp.append(d)
                    delta.append(temp)

    def fit(self,X,y):
        self.X = X
        self.y = y
        self.weightsInitialisation()
        i = 0
        while i < self.epochs:
            for j in range(len(X)):
                p = self.feedForward(X[j])
                self.backProp(p,1,y[j])
            print("Epoch : {} and MSE : {}".format(i, self.error(X,y)))
            i = i+1
            



### TESTING ####
X = np.random.normal(loc = 0, scale = 1, size = (1000,10))
y = np.random.randint(50, size=1000)
nn= NeuralNetwork(epochs= 1000, hiddenLayers= 4, neuronsEachLayer= 20, learning_rate= 0.1)
#nn.weightsInitialisation()
#p = nn.feedForward(x[0])
#print(p)
#print(nn.weights)
#print(nn.out)
#print(nn.x)
#nn.backProp( p, n = 1, actual= 1.0)
#print(nn.weights)
#print(nn.error(x,y))
#from sklearn.datasets import load_iris
#iris = load_iris()
#X = iris.data[:, (2, 3)] 
#y = (iris.target==0).astype(np.int8)
nn.fit(X,y)




