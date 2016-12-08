# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random


def gen_sample(fpath, width, height):
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return img

class OCRIter(mx.io.DataIter):
    def __init__(self, flist, width=None, height=None, batch_size=1):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.img_list = []
        with open(flist, "r") as fin:
            self.img_list = fin.read().splitlines()
        self.count = len(self.img_list)
        self.cur_index = 0
        if width == None:
            width, heigh, _ = cv2.imread(self.img_list[0].split()[0])
        self.width, self.height = width, height
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax1_label', (self.batch_size,)),
                              ('softmax2_label', (self.batch_size,)),
                              ('softmax3_label', (self.batch_size,)),
                              ('softmax4_label', (self.batch_size,)),
                              ('softmax5_label', (self.batch_size,)),
        ]


    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = [[], [], [], [], []]
            for i in range(self.batch_size):
                fpath, num = self.img_list[self.cur_index].split()
                num = [int(x) for x in num]
                self.cur_index += 1
                img = gen_sample(fpath, self.width, self.height)
                data.append(img)
                for i in range(5):
                    label[i].append(num[i])

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(l) for l in label]
            yield mx.io.DataBatch(data=data_all, label=label_all)

    def reset(self):
        print 'reseting...'
        self.cur_index = 0
        random.shuffle(self.img_list)


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
    sm1 = mx.symbol.SoftmaxOutput(data=fc21, name='softmax1')
    sm2 = mx.symbol.SoftmaxOutput(data=fc22, name='softmax2')
    sm3 = mx.symbol.SoftmaxOutput(data=fc23, name='softmax3')
    sm4 = mx.symbol.SoftmaxOutput(data=fc24, name='softmax4')
    sm5 = mx.symbol.SoftmaxOutput(data=fc25, name='softmax5')
    softmax = mx.symbol.Group([sm1, sm2, sm3, sm4, sm5])
    return softmax


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 5):
        ok = True
        for j in range(5):
            k = i * 5 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

if __name__ == '__main__':
    network = get_ocrnet()
    devs = mx.gpu()
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 20,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)

    batch_size = 8
    data_train = OCRIter('./flist.txt', 150/2, 50/2, batch_size)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X = data_train, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=mx.callback.do_checkpoint('models/new_chkpt'))

    model.save("cnn-ocr")
