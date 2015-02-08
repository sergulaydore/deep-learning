# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:38:31 2015

@author: sergulaydore
"""

# imports
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import os, subprocess

# create training data
features = np.array([[1., 0., 0],[1., 0., 1.], [1.,1.,0.], [1., 1., 1.]])
targets = np.array([0., 1., 1., 1])
n_samples = features.shape[0]
n_features = features.shape[1]

#plot data
cols = {0: 'r',1: 'b'}
for idx in range(n_samples):
    x_1 = features[idx][1]
    x_2 = features[idx][2]
    plt.plot(x_1, x_2,cols[targets[idx]]+'o', markersize=16)
    plt.title('Training Data for OR, blue = 1, red = 0')
    plt.xlabel('$x_1$', fontsize=20)
    plt.ylabel('$x_2$', fontsize=20)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
 # Symbolic variable initialization
X = T.matrix("X")
y = T.vector("y")

# Our model
def model(X,w):
    sigmoid_value = T.nnet.sigmoid(T.dot(X,w)) 
#    prediction = sigmoid_value>0.5
    return sigmoid_value
    
# parameter initialization
w = theano.shared(np.array(np.random.random_sample((3,)), dtype=theano.config.floatX))

# estimated values
y_hat = model(X,w)

# cost function
cost = T.mean(T.nnet.binary_crossentropy(y_hat,y))
#cost = T.mean(T.sqr(y-(y_hat>0.5))) 
gradient = T.grad(cost=cost,wrt=w)
updates = [[w, w-gradient*0.05]]   

# compiling to a theano function
train = theano.function(inputs = [X,y], outputs=cost, updates=updates, allow_input_downcast=True)

# Iterate through data
l = np.linspace(-1.1,1.1)
cost_list = []
for idx in range(500):
    cost = train(features, targets)
    cost_list.append(cost)
#    print y_hat.eval({X:features})
    w_values = w.get_value(borrow=True)
#    print w_values
    if idx%50 == 0:
      a,b = -w_values[1]/w_values[2], -w_values[0]/w_values[2]
      plt.plot(l, a*l+b, 'y--', linewidth=2)
      plt.title('Iteration %s\n' \
                          % (str(idx)))
      plt.savefig('it%s' % (str(idx)), \
                            dpi=200, bbox_inches='tight')
    if np.sum(targets-(y_hat.eval({X: features})>0.5))==0:
        break
   
w_values = w.get_value(borrow=True)
a,b = -w_values[1]/w_values[2], -w_values[0]/w_values[2]
plt.plot(l, a*l+b, 'k', linewidth=2)       
plt.title('Iteration %s\n' \
                          % (str(idx)))
plt.savefig('it%s' % (str(idx)), \
                            dpi=200, bbox_inches='tight')   
                            
basedir = '/Users/sergulaydore/Dropbox/Deep Learning/codes/'
os.chdir(basedir)
pngs = [pl for pl in os.listdir(basedir) if pl.endswith('png')]
sortpngs = sorted(pngs, key=lambda a:int(a.split('it')[1][:-4]))
basepng = pngs[0][:-8]
[sortpngs.append(sortpngs[-1]) for i in range(4)]
#comm = ("convert -delay 50 %s" % ' '.join(sortpngs)).split()
comm = ("convert -delay 50 %s my.gif" % (' '.join(sortpngs))).split()

proc = subprocess.Popen(comm, stdin = subprocess.PIPE, stdout = subprocess.PIPE)
(out, err) = proc.communicate()                            
                            
plt.figure()
plt.plot(cost_list)       

