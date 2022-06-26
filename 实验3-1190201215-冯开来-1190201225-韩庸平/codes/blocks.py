from torch import nn
from torch.nn import functional as F


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, image):
        identity = image                                            # shortcut
        out = self.conv1(image)
        out = self.conv2(out)
        out += identity                                           # 两路相加
        return F.relu(out)


class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次下采样
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.down_sample = nn.Sequential(                    # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, 1, stride),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, image):
        identity = self.down_sample(image)                       # 下采样，为后面相加做准备
        out = self.conv1(image)
        out = self.conv2(out)                           # 完成残差部分的卷积
        out += identity
        return F.relu(out)            # 输出卷积单元


class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))       # 全局池化
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )

    def forward(self, image):
        b, c, _, _ = image.size()
        y = self.global_pooling(image).view(b, c)   # 得到B*C*1*1,然后转成B*C，才能送入到FC层中
        y = self.fc(y).view(b, c, 1, 1)     # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return image * y.expand_as(image)   # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。


class SECommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel):        # 普通Block简单完成两次卷积操作
        super(SECommonBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.se = SEBlock(out_channel)

    def forward(self, image):
        identity = image                                    # shortcut
        out = self.conv1(image)
        out = self.conv2(out)
        out = self.se(out)                              # SE模块
        out += identity                            # 两路相加
        return F.relu(out)


class SESpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次下采样
    def __init__(self, in_channel, out_channel, stride):
        super(SESpecialBlock, self).__init__()
        self.down_sample = nn.Sequential(                    # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, 1, stride),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.se = SEBlock(out_channel)

    def forward(self, image):
        identity = self.down_sample(image)                       # 下采样，为后面相加做准备
        out = self.conv1(image)
        out = self.conv2(out)                           # 完成残差部分的卷积
        out = self.se(out)                  # SE模块
        out += identity
        return F.relu(out)            # 输出卷积单元
