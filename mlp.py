# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:17:45 2015

@author: sergulaydore
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
        
class HiddenLayer(object):
    
    def __init__( self, rng, input, n_in, n_out, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fullyconnected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        
        NOTE : The nonlinearity used here is tanh
        
        Hidden unit activation is given by: tanh(dot(input,W))
        
        :type rnf: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.sensor.dmatrix
        :param input: a symbolic tensor of shape (n_samples,n_in)
        
        :type n_in: int
        :param n_in: dimensionality of input
        
        :type n_out: int
        :param n_out: number of hidden units
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
 
        self.W = theano.shared(value=np.asarray(rng.uniform(low=-np.sqrt(6./(n_in+n_out)), high=np.sqrt(6./(n_in+n_out)), size=(n_in,n_out) ), dtype=theano.config.floatX ), 
                              name='W', borrow=True)
                             
        self.B = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX ),
                               name='B', borrow=True)    
        self.params = [self.W, self.B]                       
                              
        lin_output = T.dot(input, self.W) + self.B   
        self.output = ( lin_output if activation is None
                        else activation(lin_output))      
        self.output_binary =   self.output>0   
        
class MLP(object):
    """ Multilayer Perceptron Class
     A multilayer perceptron is a feedforward artificial neural network model
     that has one layer or more of hidden units and nonlinear activations.
    """
 
    def __init__(self, rng, input,targets, n_in, n_hidden, n_out):
        """ Initialize the parameters of the multilayer perceptron
    
    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights
    
    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the architecture
    
    :type n_in: int
    :paramm n_in: number of input units, the dimension of the space in 
                  which data points lie
                  
    :type n_hidden: int
    :param n_hidden: number of hidden units
    
    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
                  which the labels lie 
        """

        self.hiddenLayer = HiddenLayer(
              rng=rng,
              input=input,
              n_in=n_in,
              n_out=n_hidden,
              activation=T.tanh                        
        )    

        self.perceptron = Perceptron(
              input=self.hiddenLayer.output,
              targets=targets,
              n_features=n_hidden
        )
                
        self.cost = self.perceptron.binary_crossentropy
        self.errors = self.perceptron.error
        self.params = self.perceptron.params + self.hiddenLayer.params
        
        
# create training data
features = np.array([[ 0., 0],[ 0., 1.], [1.,0.], [1., 1.]])
targets = np.array([0., 1., 1., 0])
n_targets = features.shape[0]
n_features = features.shape[1]
learning_rate = 0.01

# Symbolic variable initialization
X = T.matrix("X")
y = T.vector("y")  

rng=np.random.RandomState(45656232)
my_classifier = MLP(rng=rng, input=X, targets=y, n_in=2, n_hidden=2, n_out=1)   
cost = my_classifier.cost # L1_reg*my_classifier.L1 + L2_reg*my_classifier.L2
gparams = [T.grad(cost, param) for param in my_classifier.params]
y_hat=my_classifier.perceptron.y_hat
hidden_output=my_classifier.hiddenLayer.output
w=my_classifier.perceptron.w
b=my_classifier.perceptron.b

updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(my_classifier.params, gparams)
    ]   
    
# compiling to a theano function
train = theano.function(inputs = [X,y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[hidden_output,y_hat,w,b],allow_input_downcast=True)

l = np.linspace(-1.1,1.1)
cost_list = []

cols = {0: 'r',1: 'b'}
for idx in range(8000):
    cost = train(features, targets)
    hidden_output,y_hat,w_new,b_new = predict(features)
    if idx%1000 == 0:
      plt.figure()
      for idx_samples in range(n_targets):
          x_1 = hidden_output[idx_samples][0]
          x_2 = hidden_output[idx_samples][1]
          plt.plot(x_1, x_2,cols[targets[idx_samples]]+'o', markersize=16)
          plt.title('Hidden Layer Output, blue = 1, red = 0')
          plt.xlabel('$x_1$', fontsize=20)
          plt.ylabel('$x_2$', fontsize=20)
          plt.xlim(-1.1, 1.1)
          plt.ylim(-1.1, 1.1)  
      a,b = -w_new[0]/w_new[1], -b_new/w_new[1]
      plt.plot(l, a*l+b, 'y--', linewidth=2)
      plt.title('Iteration %s\n' \
                          % (str(idx)))
      plt.savefig('it%s' % (str(idx)), \
          dpi=200, bbox_inches='tight')
    #if my_classifier.errors.eval({X:features, y:targets})==0:
    #   break
      
    
cost = train(features, targets)
hidden_output,y_hat,w_new,b_new = predict(features)  
plt.figure()
for idx_samples in range(n_targets):
          x_1 = hidden_output[idx_samples][0]
          x_2 = hidden_output[idx_samples][1]
          plt.plot(x_1, x_2,cols[targets[idx_samples]]+'o', markersize=16)
          plt.title('Hidden Layer Output, blue = 1, red = 0')
          plt.xlabel('$x_1$', fontsize=20)
          plt.ylabel('$x_2$', fontsize=20)
          plt.xlim(-1.1, 1.1)
          plt.ylim(-1.1, 1.1)  
a,b = -w_new[0]/w_new[1], -b_new/w_new[1]
plt.plot(l, a*l+b, 'k', linewidth=2)
plt.title('Iteration %s\n' \
                          % (str(idx)))
                          
plt.savefig('it%s' % (str(idx)), \
    dpi=200, bbox_inches='tight')

basedir = '/Users/sergulaydore/Dropbox/Deep Learning/codes/mlp/'
os.chdir(basedir)
pngs = [pl for pl in os.listdir(basedir) if pl.endswith('png')]
sortpngs = sorted(pngs, key=lambda a:int(a.split('it')[1][:-4]))
basepng = pngs[0][:-8]
[sortpngs.append(sortpngs[-1]) for i in range(4)]
#comm = ("convert -delay 50 %s" % ' '.join(sortpngs)).split()
comm = ("convert -delay 50 %s mlp.gif" % (' '.join(sortpngs))).split()

proc = subprocess.Popen(comm, stdin = subprocess.PIPE, stdout = subprocess.PIPE)
(out, err) = proc.communicate()                            
                            
