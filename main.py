import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

import config as cf
from data_loader import DataLoader

# For GPU
GPU = True 
GPU_ID = 0

class Mynet(chainer.Chain):

    def __init__(self):
        super(Mynet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=3, pad=1, nobias=False)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=False)
            self.fc1 = L.Linear(None, 128, nobias=False)
            self.fc_out = L.Linear(None, cf.Class_num, nobias=False)

    def __call__(self, x):
        conv1 = self.conv1(x)
        conv1 = F.relu(conv1)
        pool1 = F.max_pooling_2d(conv1, ksize=2, stride=2)
        conv2 = self.conv2(pool1)
        conv2 = F.relu(conv2)
        pool2 = F.max_pooling_2d(conv2, ksize=2, stride=2)
        fc1 = self.fc1(pool2)
        fc1_a = F.relu(fc1)
        fc1_d = F.dropout(fc1_a, ratio=0.5)
        fc_out = self.fc_out(fc1_d)
        return fc_out

model = Mynet()

if GPU:
    chainer.cuda.get_device(GPU_ID).use()
    model.to_gpu()

optimizer = chainer.optimizers.MomentumSGD(0.001)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

dl = DataLoader(phase='Train', shuffle=True)

dl_test = DataLoader(phase='Test', shuffle=True)

test_imgs, test_gts = dl_test.get_minibatch(shuffle=True)

if GPU:
    test_imgs = chainer.cuda.to_gpu(test_imgs)
    test_gts = chainer.cuda.to_gpu(test_gts)

train_losses = []
train_accuracies = []

for epoch in range(100):
    epoch += 1
    imgs, gts = dl.get_minibatch(shuffle=True)

    if GPU:
        x = chainer.cuda.to_gpu(imgs)
        t = chainer.cuda.to_gpu(gts)
    else:
        x = imgs
        t = gts

    y = model(x)

    loss = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y, t)
    model.cleargrads()
    loss.backward()
    optimizer.update()

    if GPU:
        train_losses.append(chainer.cuda.to_cpu(loss.data))
        train_accuracies.append(chainer.cuda.to_cpu(accuracy.data))
    else:
        train_losses.append(loss.data)
        train_accuracies.append(accuracy.data)

    if epoch % 10 == 0:
        y_test = model(test_imgs)
        loss_test = F.softmax_cross_entropy(y_test, test_gts)
        accu_test = F.accuracy(y_test, test_gts)

        if GPU:
            loss_test = chainer.cuda.to_cpu(loss_test.data)
            accu_test = chainer.cuda.to_cpu(accu_test.data)
        else:
            loss_test = loss_test.data
            accu_test = accu_test.data

        print('Epoch: {}, Loss {:.3f}, Accu: {:.3f}, Loss-test {:.3f}, Accu-test {:.3f}'.format(
            epoch, np.mean(train_losses), np.mean(train_accuracies), loss_test, accu_test))

chainer.serializers.save_npz("out.npz", model)