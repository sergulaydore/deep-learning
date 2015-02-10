# imports
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import os, subprocess

class Perceptron(object):
    """Perceptron for the last layer
    """
    def __init__(self, input, targets, n_features):
        """ Initialize parameters for Perceptron
        
        :type input:theano.tensor.TensorType
        :param input:symbolic variable that describes the 
                     input of the architecture
                     
        :type targets:theano.tensor.TensorType
        :param targets:symbolic variable that describes the 
                       targets of the architecture
                     
        :type n_features:int
        :param n_features:number of features 
                          (including "1" for bias term)   
                          
        """
        
        # initilialize with 0 the weights W as a matrix of shape 
        # n_features x n_targets
        
        self.w = theano.shared( value=np.zeros((n_features), dtype=theano.config.floatX),
                                name='w',
                                borrow=True
                                )  
                                
        self.b = theano.shared(0., 
                              name='b')    
                         
        self.params = [self.w, self.b]  
                                
        self.y_hat = T.nnet.sigmoid(T.dot(input,self.w)+self.b)  
        self.y_binary = self.y_hat>0.5
        self.binary_crossentropy = T.mean(T.nnet.binary_crossentropy(self.y_hat,targets))  
        self.error= T.mean(T.neq(self.y_binary, targets))      

# create training data
features = np.array([[0., 1.], [1.,0.], [ 1., 1.]])
targets = np.array([1., 1., 0.])
n_targets = features.shape[0]
n_features = features.shape[1]
                
# Symbolic variable initialization
X = T.matrix("X")
y = T.vector("y")   

my_classifier = Perceptron(input=X, targets=y,n_features=n_features)  
cost = my_classifier.binary_crossentropy 
error = my_classifier.error  
w_gradient = T.grad(cost=cost, wrt=my_classifier.w)
#b_gradient = T.grad(cost=cost, wrt=my_classifier.b)
#updates = [(my_classifier.w, my_classifier.w-w_gradient*0.05),
#            (my_classifier.b, my_classifier.b-b_gradient*0.05)]  
updates = [(my_classifier.w, my_classifier.w-w_gradient*0.05)] 
# compiling to a theano function
train = theano.function(inputs = [X,y], outputs=cost, updates=updates, allow_input_downcast=True)
#cost = train(features, targets)

# iterate through data
# Iterate through data
l = np.linspace(-1.1,1.1)
cost_list = []
for idx in range(500):
    cost = train(features, targets)
    if my_classifier.error==0:
        break
 


     
                     
      
                                                                                   
                                                                                                                                                              
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                               