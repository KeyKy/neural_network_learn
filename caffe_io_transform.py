transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_transpose('data', (2, 0, 1))
# python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，
# 所以需要一个转换
transformer.set_raw_scale('data', 255)
# caffe中图片是BGR格式，而原始格式是RGB，所以要转化
transformer.set_channel_swap('data', (2, 1, 0))
# 将输入图片格式转化为合适格式（与deploy文件相同）
net.blobs['data'].reshape(batch_size, 3, crop_dims[0], crop_dims[1])

import caffe
import lmdb

lmdb_env = lmdb.open('directory_containing_mdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    for l, d in zip(label, data):
            print l, d

import caffe
import lmdb
from PIL import Image

in_db = lmdb.open('image-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()