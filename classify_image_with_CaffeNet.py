# Classification: Instant Recognition with Caffe
# http://localhost:8888/notebooks/examples/00-classification.ipynb
# In this example we'll classify an image with the bundled CaffeNet model 
# (which is based on the network architecture of Krizhevsky et al. for ImageNet).
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import sys, os
caffe_root = '/Users/zj-db0655/Downloads/caffe-master'
print sys.path
sys.path.insert(0, caffe_root + os.sep + 'python') # add path temporally
#print sys.path

import caffe
if os.path.isfile(caffe_root + os.sep + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system(' '.join(['cd', caffe_root, '&&', '/Users/zj-db0655/anaconda/bin/python ./scripts/download_model_binary.py ./models/bvlc_reference_caffenet']))
    
caffe.set_mode_cpu()
model_def = caffe_root + os.sep + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weight = caffe_root + os.sep + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weight, caffe.TEST)
print net.blobs['data'].data.shape
mu = np.load(caffe_root + os.sep + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(50, # batch size
                          3,  # 3-channel (BGR) images
                          227,227) # image size is 227x227
image = caffe.io.load_image(caffe_root + os.sep + 'examples/images/cat.jpg')
transformered_image = transformer.preprocess('data', image)
print transformered_image.shape

net.blobs['data'].data[...] = transformered_image

output = net.forward()
output_prob = output['prob'][0]

print 'predicted class is:', output_prob.argmax()

labels_file = caffe_root + os.sep + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system(' '.join(['cd', caffe_root, '&&', '/Users/zj-db0655/anaconda/bin/python ./data/ilsvrc12/get_ilsvrc_aux.sh']))
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]



