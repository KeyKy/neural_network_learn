# example of implementing neural network
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from dask.array.chunk import keepdims_wrapper

np.random.seed(0)
# Xshape=(200,2)
X, y = datasets.make_moons(200, noise=0.20)

num_examples = len(X) 
nn_input_dim = 2
nn_output_dim = 2

epsilon = 0.01 
reg_lambda = 0.01

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    # set keepdims=True for 2-dims of exp_scores
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    np.random.seed(0)
    # W1dim=(2,nn_hdim) W2dim=(nn_hdim,nn_output_dim)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    model = {}
    
    for i in xrange(0, num_passes):
        # forward propagation
        z1 = X.dot(W1) + b1 #z1dim=(200,nn_hdim)
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2 #z2dim=(200,nn_output_dim)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #probsdim=(200,nn_output_dim)

        # back propagation
        delta_output = probs
        delta_output[range(num_examples), y] -= 1 #dims=(200,nn_output_dims)

        dW2 = (a1.T).dot(delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)
        delta_fc2 = delta_output.dot(W2.T) * (1 - a1**2)
        dW1 = X.T.dot(delta_fc2)
        db1 = np.sum(delta_fc2, axis=0, keepdims=True)
        
        dW2 = reg_lambda * dW2
        dW1 = reg_lambda * dW1
        
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))
    return model

model = build_model(3, print_loss=True)
h = 0.02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy= np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()

    
    