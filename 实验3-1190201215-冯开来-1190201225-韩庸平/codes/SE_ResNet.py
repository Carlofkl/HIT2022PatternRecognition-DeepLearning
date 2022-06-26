from torch import nn
import blocks as Block


class SEResNet(nn.Module):
    def __init__(self):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            Block.SECommonBlock(64, 64),
            Block.SECommonBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            Block.SESpecialBlock(64, 128, 2),         # stride != 1 and in_channel != out_channel，需要下采样
            Block.SECommonBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            Block.SESpecialBlock(128, 256, 2),
            Block.SECommonBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            Block.SESpecialBlock(256, 512, 2),
            Block.SECommonBlock(512, 512)
        )
        self.dense = nn.Sequential(                # 最后用于分类的全连接层，根据需要灵活变化
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),       # 自适应平均池化
            nn.Flatten(),
            nn.Linear(512, 12)
        )

    def forward(self, image):
        img = self.conv1(image)
        img = self.conv2(img)          # 四个卷积单元
        img = self.conv3(img)
        img = self.conv4(img)
        img = self.conv5(img)
        img = self.dense(img)            # 全连接
        return img
