"""
Python 3.8
torch 1.11
torchvision 0.12
device: cpu
"""
import datetime
import random
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()  #
        self.Flatten = torch.nn.Flatten()
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784, 512)  # 隐含层
        self.fc2 = torch.nn.Linear(512, 128)  # 隐含层
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        # 前向传播，输入值：x，返回值out
        x = self.Flatten(x)  # 将一个多行的Tensor,拼接成一行
        out = F.relu(self.fc1(x))  # 使用 relu 激活函数
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)  # 输出层使用softmax
        # 784×1的张量最后输出为10×1的张量，
        # 每个值为0-9类别的概率分布，最后选取概率最大的作为预测值输出
        return out


def main(args):

    # device = args.device
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model
    model = MLP()
    model.to('cpu')

    # Loss
    lossfunc = torch.nn.CrossEntropyLoss()

    # Set up optimizers
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

    # Dataset
    dataset_train = datasets.MNIST(root='./dataset/mnist', train=True,
                                   download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(root='./data/mnist', train=False,
                                  download=True, transform=transforms.ToTensor())
    data_loader_train = DataLoader(dataset_train, args.batch_size)
    data_loader_test = DataLoader(dataset_test, args.batch_size)

    # epoch
    print("Start training\n")
    start_time = time.time()
    train_loss_pic = []
    accuracy_pic = []

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}")
        train_loss = 0.0
        best_metric = 0.0

        print("train_one_epoch begin")

        # train
        for data, target in data_loader_train:
            optimizer.zero_grad()  # 清空上一步的残余，更新参数值
            output = model(data)  # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(data_loader_train.dataset)
        train_loss_pic.append(train_loss)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

        # val
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader_test:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_pic.append(accuracy)
        print('Accuracy on the test images: {:.2%}'.format(correct / total))
        if args.output_dir and accuracy > best_metric:
            best_metric = accuracy
            checkpoint_path = output_dir / "BEST_MLP.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "accuracy": best_metric,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch+1,
                    "args": args
                },
                checkpoint_path,
            )
        print("train_one_epoch ends\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    x = np.arange(args.epochs)
    y1 = train_loss_pic
    y2 = accuracy_pic
    fig = plt.figure(2, figsize=(16, 8), dpi=50)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')
    ax1.plot(x, y1, 'r', label='train_loss')
    ax2.plot(x, y2, 'g--', label='accuracy')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--output-dir', default="models")
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--optimizer', default="adam", type=str)
    args = parser.parse_args()

    print(args)

    main(args)


