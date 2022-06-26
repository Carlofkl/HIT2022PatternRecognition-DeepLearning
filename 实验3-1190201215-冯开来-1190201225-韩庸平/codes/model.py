import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def train(net, train_loader, device, epoch, opt, model_path):
    print('training.....')
    net.train()
    Swriter = SummaryWriter(log_dir='./vision')
    loss_func = nn.CrossEntropyLoss()        # 交叉熵损失函数
    if opt.optim == 'Adam':
        print("optimizer: Adam")
        optimizer = optim.Adam(net.parameters(), lr=0.0001)  # adam优化器
    else:
        print("optimizer: SGD")
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # SGD优化器
    for i in range(epoch):
        train_loss = 0
        tran_data = tqdm(train_loader)
        for j, (image, label) in enumerate(tran_data):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = net(image)
            loss = loss_func(out, label).to(device)
            train_loss += loss.data
            loss.backward()
            optimizer.step()
        print('epoch {}/{}, Loss: {:.6f}'.format(i + 1, epoch, train_loss / len(train_loader)))
        Swriter.add_scalar('loss_' + opt.net + '_' + opt.optim + '_' + opt.aug,
                           train_loss / len(train_loader), global_step=i+1)
    torch.save(net, model_path)
    print('training completed')
