import torch.nn as nn
import torch.nn.functional as F


# 输入图像尺寸为(227,227,3)
class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,stride=4)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2,stride=1)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1,stride=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.dense1 = nn.Linear(6*6*256,4096)
        self.dense2 = nn.Linear(4096,4096)
        self.dense3 = nn.Linear(4096,num_classes)
        self.dropout = nn.Dropout(p=0.5)

        if init_weights:
            self._initialize_weights()



    def forward(self,x):
        x = F.relu(self.conv1(x))      # 第一层卷积运算并进行relu   (55,55,96)
        x = self.pool(x)        # 最大池化    (27,27,96)
        x = F.relu(self.conv2(x))      # 第二层卷积运算并relu    (27,27,256)
        x = self.pool(x)        # 最大池化    (13,13,256)
        x = F.relu(self.conv3(x))      # 第三层卷积运算并进行relu  (13,13,384)
        x = F.relu(self.conv4(x))      # 第四层卷积运算并进行relu  (13,13,384)
        x = F.relu(self.conv5(x))      # 第五层卷积运算并进行relu  (13,13,256)
        x = self.pool(x)        # 最大池化    (6,6,256)

        x = x.view(x.size(0),-1)

        x = F.relu(self.dense1(x))   # 全连接并relu
        x = self.dropout(x)          # dropout
        x = F.relu(self.dense2(x))   # 全连接并relu
        x = self.dropout(x)
        x = self.dense3(x)

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

