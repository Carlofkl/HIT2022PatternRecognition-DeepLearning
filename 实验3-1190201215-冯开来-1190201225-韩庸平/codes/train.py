import glob
import time

import cv2 as cv
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
import json

cuda = torch.cuda.is_available()


class my_dataset(Dataset):

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.len = len(labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return self.len


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=3),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride = (1, 1), padding=2),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.l1 = nn.Linear(6*6*128*2, 4096)
        self.l2 = nn.Linear(4096, 4096)
        self.l3 = nn.Linear(4096, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = x.view(x.shape[0], -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


def load_data():

    with open('label_dic.json', 'r') as f:
        label_dic = json.load(f)
    f.close()

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform = transforms.ToTensor()
    # transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    img_paths = glob.glob('../dataset/caltech-101/101_ObjectCategories/*/*.jpg')
    data = []
    labels = []
    for img_path in tqdm(img_paths):

        label = img_path.split(os.path.sep)[-2]
        if label == 'BACKGROUND_Google':
            continue

        img = cv.imread(img_path)
        img = cv.resize(img, (224, 224))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = transform(img)

        data.append(img)
        labels.append(label_dic[label])

    return data, labels


def train(epoch):
    model.train()
    loss_sum = 0.0
    print('Beginning of epoch %d, lr = %f' % (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))
    tic = time.time()
    for idx, data in enumerate(train_dataloader, 1):
        input, target = data
        if cuda:
            input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        if idx % 20 == 0:
            print('[epoch %d, batch %d] loss: %.3f' % (epoch + 1, idx, loss_sum / 20))
            loss_sum = 0.0
    toc = time.time()
    print('Epoch %d finished, took %.2fs' % (epoch + 1, toc - tic))


def validation():
    model.eval()
    correct_cnt = 0  # 预测正确的数量
    total_cnt = 0  # 样本总数
    with torch.no_grad():
        for data in val_dataloader:  # 每次取出一个mini-batch
            input, labels = data
            if cuda:
                input, labels = input.cuda(), labels.cuda()
            output = model(input)
            _, pred = torch.max(output.data, dim=1)  # 选取数值最大的一维对应的标签作为预测标签
            total_cnt += labels.size(0)
            correct_cnt += (pred == labels).sum().item()
    print('Accuracy on validation set: %.2f%%\n' % (100.0 * correct_cnt / total_cnt))


if __name__ == '__main__':
    batch_size = 64
    epochs = 20
    lr = 0.001
    data, labels = load_data()
    train_val_dataset = my_dataset(data, labels)
    train_dataset, val_dataset = random_split(train_val_dataset,
                                              [8000, 677],
                                              generator=torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.3)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    if cuda:
        model.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        scheduler.step()
        train(epoch)
        validation()

    torch.save(model, '../models/model.pth')

