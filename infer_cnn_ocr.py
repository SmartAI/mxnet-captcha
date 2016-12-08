# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import mxnet as mx
import numpy as np
from cnn_ocr import gen_sample
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


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
    return predicted


if __name__ == '__main__':
    mod = inference_init()
    predicted = inference_process(mod, '22483.jpg')
    print predicted
