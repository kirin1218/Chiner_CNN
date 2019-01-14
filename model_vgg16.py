import chainer
import chainer.links as L
import chainer.functions as F

import config as cf

class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc1 = L.Linear(None, 128)
            self.fc_out = L.Linear(None, cf.Class_num)


    def __call__(self, x):
        #print(self.base.available_layers)
        #h = self.base(x, layers=['fpool5'])['pool5']
        h = self.base.extract(x, layers=['pool5'], size=(cf.Height, cf.Width))['pool5']
        fc1 = self.fc1(h)
        fc1_a = F.relu(fc1)
        fc1_d = F.dropout(fc1_a, ratio=0.5)
        fc_out = self.fc_out(fc1_a)
        return fc_out


    ## For visualize inter layer
    def get_inter_layer(self, x):
        conv1 = self.conv1(x)
        conv1_a = F.relu(conv1)
        pool1 = F.max_pooling_2d(conv1, ksize=2, stride=2)
        conv2 = self.conv2(pool1)
        conv2_a = F.relu(conv2)
        pool2 = F.max_pooling_2d(conv2, ksize=2, stride=2)
        fc1 = self.fc1(pool2)
        fc1_a = F.relu(fc1)
        fc1_d = F.dropout(fc1_a, ratio=0.5)
        fc_out = self.fc_out(fc1_a)
        return conv2_a