import VGG
import model
import torch
import writer
import ResNet
import argparse
import SE_ResNet
import data_path as dp
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import dataloader


epoch = 40
batch_size = 32
train_data, labels_index = dp.get_data()  # 获取所有数据的路径，labels_index存放所有类别的label

data_augmentation = {
    "yes": transforms.Compose([transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(45),
                               transforms.RandomAffine(degrees=0, translate=(0, 0.2)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "no": transforms.Compose([transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='GPU')  # 选择GPU还是CPU
    parser.add_argument('--net', type=str, default=None)    # 选择使用的网络
    parser.add_argument('--aug', type=str, default='yes')   # 选择是否进行数据增强
    parser.add_argument('--optim', type=str, default='Adam')  # 选择使用的优化器
    opt = parser.parse_args()
    # 运算设备,默认选择GPU
    if opt.device == 'GPU':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("设备: " + torch.cuda.get_device_name(device))
    # 选择是否进行数据增强，默认选择yes
    if opt.aug == 'yes':
        augment = '_AUGMENTATION'
        print("data augmentation...")
        train_set = datasets.ImageFolder(root='train', transform=data_augmentation["yes"])
    else:
        augment = '_NO_AUGMENTATION'
        train_set = datasets.ImageFolder(root='train', transform=data_augmentation["no"])
    train_loader = dataloader.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # 选择想要使用的网络
    net = None
    if opt.net == 'VGG':
        net = VGG.VGG().to(device)
        print("VGG loading...")
    elif opt.net == 'ResNet':
        net = ResNet.ResNet().to(device)
        print("ResNet loading...")
    elif opt.net == 'SEResNet':
        net = SE_ResNet.SEResNet().to(device)
        print("SE_ResNet loading...")
    modelPath = 'models/' + opt.net + '_' + opt.optim + augment + '.pkl'
    csvPath = 'result/' + opt.net + '_' + opt.optim + augment + '.csv'
    model.train(net, train_loader, device, epoch, opt, modelPath)
    writer.write_to_csv(net, data_augmentation["no"], device, labels_index, csvPath)
