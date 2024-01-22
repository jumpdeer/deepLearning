import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)   # 卷积层  输入通道数为3，输出的四维张量通道数为16，卷积核为5*5，步长默认为1，
        self.pool1 = nn.MaxPool2d(2, 2)    # 最大池化层  池化窗口 2*2，移动步长为2
        self.conv2 = nn.Conv2d(16, 32, 5)  # 卷积层  输入通道数为16(接受上个卷积层的数据)，输出的四维张量通道数为32，卷积核为5*5
        self.pool2 = nn.MaxPool2d(2, 2)    # 最大池化层  池化窗口 2*2，移动步长为2
        self.fc1 = nn.Linear(32*5*5, 120)  # 全连接层 输入通道为 32*5*5 ，输出通道为120
        self.fc2 = nn.Linear(120, 84)      # 全连接层 输入通道为 120 输出通道为84
        self.fc3 = nn.Linear(84, 10)       # 输入通道为 84   输出通道为10

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


