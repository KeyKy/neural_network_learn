import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_cpu()

net = caffe.Net('conv.prototxt', caffe.TEST)

# net.blobs for input data and its propagation in the layers :
net.blobs['data'] # contains input data, an array of shape (1, 1, 100, 100)
net.blobs['conv'] # contains computed data in layer ‘conv’ (1, 3, 96, 96)
print [(k, v.data.shape) for k, v in net.blobs.items()]

# net.params a vector of blobs for weight and bias parameters
net.params['conv'][0] # contains the weight parameters, an array of shape (3, 1, 5, 5)
net.params['conv'][1] # contains the bias parameters, an array of shape (3,)
print [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]

solver = caffe.get_solver('models/bvlc_reference_caffenet/solver.prototxt')
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

# For the computation of the gradients
# computation of the net.blobs[k].diff and net.params[k][j].diff from the loss layer until input layer
solver.net.backward() 

# To launch one step of the gradient descent, that is a forward propagation, a backward propagation and the update of the net params given the gradients (update of the net.params[k][j].data) :
solver.step(1)

# To run the full gradient descent, that is the max_iter steps :
solver.solve()