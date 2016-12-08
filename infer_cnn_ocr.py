# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import pudb
import mxnet as mx
import numpy as np
import cv2, random
from cnn_ocr import gen_sample, OCRIter
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

def get_ocrnet():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 256)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc25 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25], dim = 0)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25], dim = 0)
    return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")


def inference_init():
    batch_size = 1
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/new_chkpt', 20)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 25, 75))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

def inference_process(mod, path):
    print path
    img = gen_sample(path, 150/2, 50/2)
    img = np.expand_dims(img, axis=0)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()
    predicted = ''.join([str(np.argmax(p.asnumpy())) for p in prob])
    print 'predicted: ' + predicted
    print predicted


if __name__ == '__main__':
    batch_size = 1
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/new_chkpt', 20)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 25, 75))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    path = '/Users/smart/data/dianrong/train/22483.jpg'
    img = gen_sample(path, 150/2, 50/2)
    img = np.expand_dims(img, axis=0)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()
    predicted = ''.join([str(np.argmax(p.asnumpy())) for p in prob])
    print 'predicted: ' + predicted
