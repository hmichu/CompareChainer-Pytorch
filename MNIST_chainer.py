# python 3.6.10
# numpy 1.18.1
# chainer 7.1.0
# cupy-cuda101 7.1.1

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import Chain
from chainer import datasets

from matplotlib import pyplot as plt
import pandas as pd

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 6, 5)
            self.conv2 = L.Convolution2D(6, 16, 5)
            self.fc1 = L.Linear(256, 120)
            self.fc2 = L.Linear(120, 64)
            self.fc3 = L.Linear(64, 10)

    def forward(self, x):
        x = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        x = F.max_pooling_2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the MNIST dataset
def DataSet(train_batch_size, test_batch_size):
    TrainDataset, TestDataset = chainer.datasets.get_mnist(ndim=3)
    split_at = int(len(TrainDataset) * 0.8)
    train, valid = datasets.split_dataset_random(TrainDataset, split_at)
    train_data_size = len(train)
    valid_data_size = len(valid)
    
    train_iter = chainer.iterators.SerialIterator(train, batch_size=train_batch_size)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size=train_batch_size)
    test_iter = chainer.iterators.SerialIterator(TestDataset, batch_size=test_batch_size, repeat=False, shuffle=False)

    return train_data_size, valid_data_size, train_iter, valid_iter, test_iter

def TrainBatch(train_data_size, batchsize, train_iter, device, model, opt):
    train_loss = 0
    train_acc = 0
    cnt = 0
    for i in range(0, train_data_size, batchsize):
        train_batch = train_iter.next()
        x, t = chainer.dataset.concat_examples(train_batch, device)
        model.cleargrads()
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        train_loss += loss.array
        acc = F.accuracy(y, t)
        train_acc += acc.array
        loss.backward()
        opt.update()
        cnt += 1
    avg_train_loss = train_loss / train_data_size
    avg_train_acc = train_acc / cnt
    return chainer.cuda.to_cpu(avg_train_loss), chainer.cuda.to_cpu(avg_train_acc)

def ValidBatch(valid_data_size, batchsize, valid_iter, device, model):
    val_loss = 0
    val_acc = 0
    cnt = 0
    for i in range(0, valid_data_size, batchsize):
        valid_batch = valid_iter.next()
        x, t = chainer.dataset.concat_examples(valid_batch, device)
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        val_loss += loss.array
        acc = F.accuracy(y, t)
        val_acc += acc.array
        cnt += 1
    avg_val_loss = val_loss / valid_data_size
    avg_val_acc = val_acc / cnt
    return chainer.cuda.to_cpu(avg_val_loss), chainer.cuda.to_cpu(avg_val_acc)


def ViewGraph(epoch_num, train_loss_log, train_acc_log, val_loss_log, val_acc_log):
    plt.figure()
    plt.plot(range(epoch_num), train_loss_log, color="blue", linestyle="-", label="train_loss")
    plt.plot(range(epoch_num), val_loss_log, color="green", linestyle="--", label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation loss")
    plt.grid()
    
    plt.figure()
    plt.plot(range(epoch_num), train_acc_log, color="blue", linestyle="-", label="train_acc")
    plt.plot(range(epoch_num), val_acc_log, color="green", linestyle="--", label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Training and Validation accuracy")
    plt.grid()

    plt.show()


def main():
    #データの読み込み
    train_batch_size = 128
    test_batch_size = 1000
    train_data_size, valid_data_size, train_loader, val_loader, test_loader = DataSet(train_batch_size=train_batch_size, test_batch_size=test_batch_size)
    print("train data size:",train_data_size, " valid data size:",valid_data_size)

    model = CNN()
    if chainer.cuda.available:
        device = chainer.get_device(0)
        print("use gpu")
    else:
        device = -1
        print("use cpu")
    model.to_device(device)
    #device.use
    
    opt = optimizers.Adam()
    opt.setup(model)

    epoch_num = 10
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []
    
    print("epoch:",epoch_num,"batch size:",train_batch_size)
    print("start train.")
    import time
    for epoch in range(epoch_num):
        start_time = time.perf_counter()
        avg_train_loss, avg_train_acc = TrainBatch(train_data_size, train_batch_size, train_loader, device, model, opt)
        end_time = time.perf_counter()

        s_val_time = time.perf_counter()
        avg_val_loss, avg_val_acc = ValidBatch(valid_data_size, train_batch_size, val_loader, device, model)
        e_val_time = time.perf_counter()

        proc_time = end_time - start_time
        val_time = e_val_time - s_val_time
        print("Epoch[{}/{}], train loss: {loss:.4f}, valid loss: {val_loss:.4f}, valid acc: {val_acc:.4f}, "\
        "train time: {proc_time:.4f}sec, valid time: {val_time:.4f}sec"\
            .format(epoch+1, epoch_num, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc, 
            proc_time=proc_time, val_time=val_time))
        
        train_loss_log.append(avg_train_loss)
        train_acc_log.append(avg_train_acc)
        val_loss_log.append(avg_val_loss)
        val_acc_log.append(avg_val_acc)

    ViewGraph(epoch_num, train_loss_log, train_acc_log, val_loss_log, val_acc_log)


if __name__=='__main__':
    main()
