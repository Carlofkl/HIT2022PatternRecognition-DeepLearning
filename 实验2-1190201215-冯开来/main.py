import argparse
import os
import random
import torch.nn as nn
import torch.optim
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            # 输入 (3, 109, 109)
            # 输出 (8, 55, 55)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5), stride=(2, 2), padding=2),  # (109-5+4)/2+1=55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3+0)/2+1=27
            # 输入 (8, 27, 27)
            # 输出 (16, 27, 27)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (27-3+2)/1+1=27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3+0)/2+1=13
            # 输入 (16, 13, 13)
            # 输出 (32, 13, 13)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (13-3+2)/1+1=13
            nn.ReLU(),
            # 输入 (32, 13, 13)
            # 输出 (64, 13, 13)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (13-3+2)/1+1=13
            nn.ReLU(),
            # 输入 (64, 13, 13)
            # 输出 (128, 6, 6)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (13-3+0)/2+1=6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 101)
        )

        self.init_weights()  # initialize bias

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[8].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)


class MyDataset(Dataset):

    def __init__(self, args):
        self.imgs = []
        self.labels = []
        self.transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ])

        img_dir = os.path.join(args.data_path, '101_ObjectCategories/')
        label = -1
        for path in os.listdir(img_dir):
            if path == 'BACKGROUND_Google':
                continue
            path_name = os.path.join(img_dir, path)
            label += 1
            for name in os.listdir(path_name):
                file_name = os.path.join(path_name, name)
                self.labels.append(label)
                img = self.transforms(Image.open(file_name).convert('RGB'))
                self.imgs.append(img)

    def __getitem__(self, idx):
        return {"img": self.imgs[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


def main(args):

    device = args.device

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataset
    my_dataset = MyDataset(args)
    full_size = my_dataset.__len__()
    train_size = int(0.8 * full_size)
    val_size = int(0.1 * full_size)
    test_size = full_size - train_size - val_size
    train_dataset, rest_dataset = random_split(my_dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = random_split(rest_dataset, [val_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # model
    alexnet = AlexNet()
    writer = SummaryWriter()
    input_img = torch.randn(30, 3, 109, 109)
    writer.add_graph(alexnet, input_img)
    alexnet.to(device)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(alexnet.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, [20, 25], 0.1)
    lossfunc = nn.CrossEntropyLoss()

    # Only run on test
    if args.test:

        print("*****Starting testing*****\n")

        test_num = 0
        test_accur = 0.0
        load_dir = Path(args.output_path) / "AlexNet.pth"
        checkpoint = torch.load(load_dir)
        alexnet.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        alexnet.eval()
        with torch.no_grad():
            for data in test_loader:
                img, label = data['img'].to(device), data['label'].to(device)
                outputs = alexnet(img)
                _, pred = torch.max(outputs.data, dim=1)
                test_num += label.size(0)
                test_accur += (pred == label).sum().item()
        test_accur /= test_num
        print('Test Accuracy: {:.2%}'.format(test_accur))

        print("\n*****Testing ends*****")
        return

    # start train and val
    epochs = args.epochs
    train_loss_all = []
    val_accur_all = []
    best_accur = 0.0

    print("*****Starting training*****\n")

    for epoch in range(epochs):
        print("Epoch [{}/{}]".format(epoch+1, epochs))
        train_loss = 0.0
        train_num = 0.0
        val_accur = 0.0
        val_num = 0.0

        # train
        alexnet.train()
        for idx, data in enumerate(train_loader):
            img, label = data['img'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = alexnet(img)
            loss = lossfunc(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            train_num += img.size(0)
        train_loss /= train_num
        train_loss_all.append(train_loss)
        scheduler.step()

        # val
        alexnet.eval()
        with torch.no_grad():
            for data in val_loader:
                img, label = data['img'].to(device), data['label'].to(device)
                outputs = alexnet(img)
                _, pred = torch.max(outputs.data, dim=1)

                val_num += label.size(0)
                val_accur += (pred == label).sum().item()
        val_accur = val_accur / val_num
        val_accur_all.append(val_accur)

        if val_accur > best_accur and args.output_path:
            best_accur = val_accur
            output_dir = Path(args.output_path) / "AlexNet.pth"
            state = {
                'net': alexnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1
            }
            torch.save(state, output_dir)

        print('Epoch:  {} \tTraining Loss: {:.6f} \tValidation Accuracy: {:.2%}'
              .format(epoch + 1, train_loss, val_accur))

    for x in range(epochs):
        writer.add_scalar('train_loss', train_loss_all[x], x)
        writer.add_scalar('val_accuracy', val_accur_all[x], x)
    writer.close()

    print("\n*****Training ends*****")


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AlexNet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=30)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--seed", default=2)
    parser.add_argument("--learning-rate", default=0.001)
    parser.add_argument("--output-path", default="./model")  
    parser.add_argument("--data-path", default="./caltech-101/")
    parser.add_argument("--img-size", default=109)
    parser.add_argument("--test", action="store_true", help="Only run test")

    args = parser.parse_args()
    print(args)

    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
