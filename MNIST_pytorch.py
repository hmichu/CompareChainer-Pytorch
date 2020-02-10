# python 3.6.10
# numpy 1.18.1
# pytorch 1.4.0
# cudatoolkit 10.1.243
# torchvision 0.5.0

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
import pandas as pd

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        #x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# data
def DataSet(train_batch_size, test_batch_size):
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans = transforms.ToTensor()

    Train_DataSet = datasets.MNIST('~/dataset/MINST/', train=True, download=True, transform=trans)
    Test_DataSet = datasets.MNIST('~/dataset/MINST/', train=False, download=True, transform=trans)
    #print(Train_DataSet)
    #print(Test_DataSet)

    # データの分割
    n_samples = len(Train_DataSet) # 全データ数
    train_size = int(n_samples * 0.8) # 学習データは全データの8割
    val_size = n_samples - train_size # 評価データは残り
    train_dataset, valid_dataset = torch.utils.data.random_split(Train_DataSet, [train_size, val_size])
    #print(train_dataset)
    #print(valid_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Test_DataSet, batch_size=test_batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def TrainBatch(train_data_size, train_loader, device, model, loss_func, optimizer):
    train_loss = 0
    train_acc = 0
    cnt = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, target)
        train_loss += loss.item()
        #_, pre = torch.max(out, 1)
        pre = out.max(1)[1]
        train_acc += (pre == target).sum().item()
        cnt += target.size(0)
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / train_data_size
    avg_train_acc = train_acc / cnt
    return avg_train_loss, avg_train_acc

def ValBatch(val_data_size, val_loader, device, model, loss_func):
    val_loss = 0
    val_acc = 0
    cnt = 0
    model.eval()
    with torch.no_grad():  # 必要のない計算を停止
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_func(out, target)
            val_loss += loss.item()
            _, pre = torch.max(out, 1)
            pre = out.max(1)[1]
            val_acc += (pre == target).sum().item()
            cnt += target.size(0)
    avg_val_loss = val_loss / val_data_size
    avg_val_acc = val_acc / cnt
    return avg_val_loss, avg_val_acc


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

def PlotConfusionMatrix(cm, labels):
    import seaborn as sns
    sns.set()

    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels

    f, ax = plt.subplots()
    #f, ax = plt.figure()
    sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_ylim(len(labels), 0)
    plt.show()

def Test():
    from sklearn.metrics import confusion_matrix, classification_report
    #fig = plt.figure()
    all_labels = np.array([])
    all_preds = np.array([])

    
    #データの読み込み
    _, _, test_loader = DataSet(train_batch_size=128, test_batch_size=1000)

    
    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    mymodel = CNN().to(device)
    mymodel.load_state_dict(torch.load("mymodel.ckpt"))

    mymodel.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = mymodel(data)
            #_, preds = torch.max(outputs.data, 1)
            preds = outputs.max(1)[1]
            test_acc += (preds == target).sum().item()
            total += target.size(0)

            all_labels = np.append(all_labels, target.cpu().data.numpy())
            all_preds = np.append(all_preds, preds.cpu().numpy())

        print("正解率: {}%".format(100*test_acc/total))
        report = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True))
        report.to_csv("classification_report.csv")
        print(report)
        cm = confusion_matrix(all_labels, all_preds)
        PlotConfusionMatrix(cm, np.array(["0","1","2","3","4","5","6","7","8","9"]))

def main():
    #データの読み込み
    train_batch_size = 128
    test_batch_size = 1000
    train_loader, val_loader, test_loader = DataSet(train_batch_size=train_batch_size, test_batch_size=test_batch_size)
    train_data_size = len(train_loader.dataset)
    val_data_size = len(val_loader.dataset)
    print("train data size:",train_data_size," valid data size:", val_data_size)
    
    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("use "+device)

    
    model = CNN().to(device)
    #print(model)

    # optimizing
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    
    epoch_num = 10

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    print("epoch: ",epoch_num," batch size:", train_batch_size)
    print("start train.")
    import time
    for epoch in range(epoch_num):
        start_time = time.perf_counter()
        avg_train_loss, avg_train_acc = TrainBatch(train_data_size, train_loader, device, model, loss_func, optimizer)
        end_time = time.perf_counter()

        s_val_time = time.perf_counter()
        avg_val_loss, avg_val_acc = ValBatch(val_data_size, val_loader, device, model, loss_func)
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

    # モデルの保存
    torch.save(model.state_dict(), "mymodel.ckpt")
    print("save model.")

    ViewGraph(epoch_num, train_loss_log, train_acc_log, val_loss_log, val_acc_log)


if __name__ == '__main__':
    main()
    #Test()
