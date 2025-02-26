import torch.nn as nn
import torch.nn.functional as F

class Bottleneck_18_34(nn.Module):
    def __init__(self,in_channels,out_channels,downsample):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self,x):
        skip_var = x  # 跳跃连接的特征图
        if self.downsample is not None:
            skip_var = self.downsample(skip_var)

        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)

        x +=skip_var
        x = F.relu(x)

        return x

class Bottleneck_50_101_152(nn.Module):
    def __init__(self,in_channels,out_channels,downsample):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels,out_channels,1,1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.layer3 = nn.Conv2d(out_channels,out_channels*4,1,1)
        self.bn3 = nn.BatchNorm2d(out_channels*4)

        self.downsample = downsample

    def forward(self,x):
        skip_var = x  # 跳跃连接的特征图
        if self.downsample is not None:
            skip_var = self.downsample(skip_var)

        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)

        x +=skip_var
        x = F.relu(x)

        return x


class ResNet18(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(ResNet18,self).__init__()
        self.block = [64,128,256,512]
        self.conv_num = [2,2,2,2]
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.modellist = self._construct_layer()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(7*7*512,num_classes)


    def _construct_layer(self):
        layers = nn.ModuleList()
        downsample = None
        for index in range(4):
            if index != 0:
                downsample = nn.Conv2d(self.block[index-1],self.block[index],1)
            flag = True
            for num in range(self.conv_num[index]):
                if flag:
                    if index!=0:
                        layers.append(Bottleneck_18_34(self.block[index-1],self.block[index],downsample))
                    else:
                        layers.append(Bottleneck_18_34(64,self.block[index],downsample))
                    flag=False
                else:
                    layers.append(Bottleneck_18_34(self.block[index],self.block[index],None))

        return layers

    def forward(self,x):

        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.modellist:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dense(x)

        return x

class ResNet34(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(ResNet34,self).__init__()
        self.block = [64,128,256,512]
        self.conv_num = [3,4,6,3]
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.modellist = self._construct_layer()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(7*7*512,num_classes)


    def _construct_layer(self):
        layers = nn.ModuleList()
        downsample = None
        for index in range(4):
            if index != 0:
                downsample = nn.Conv2d(self.block[index-1],self.block[index],1)
            flag = True
            for num in range(self.conv_num[index]):
                if flag:
                    if index!=0:
                        layers.append(Bottleneck_18_34(self.block[index-1],self.block[index],downsample))
                    else:
                        layers.append(Bottleneck_18_34(64,self.block[index],downsample))
                    flag=False
                else:
                    layers.append(Bottleneck_18_34(self.block[index],self.block[index],None))

        return layers

    def forward(self,x):

        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.modellist:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dense(x)

        return x

class ResNet50(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(ResNet50,self).__init__()
        self.block = [64,128,256,512]
        self.conv_num = [3,4,6,3]
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.modellist = self._construct_layer()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(7*7*512,num_classes)

    def _construct_layer(self):
        layers = nn.ModuleList()
        downsample = None
        for index in range(4):
            if index != 0:
                downsample = nn.Conv2d(self.block[index-1]*4,self.block[index]*4,1)
            flag = True
            for num in range(self.conv_num[index]):
                if flag:
                    if index!=0:
                        layers.append(Bottleneck_50_101_152(self.block[index-1]*4,self.block[index],downsample))
                    else:
                        layers.append(Bottleneck_50_101_152(64,self.block[index],downsample))
                    flag=False
                else:
                    layers.append(Bottleneck_50_101_152(self.block[index]*4,self.block[index],None))

        return layers

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.modellist:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dense(x)

        return x

class ResNet101(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(ResNet101,self).__init__()
        self.block = [64,128,256,512]
        self.conv_num = [3,4,23,3]
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.modellist = self._construct_layer()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(7*7*512,num_classes)

    def _construct_layer(self):
        layers = nn.ModuleList()
        downsample = None
        for index in range(4):
            if index != 0:
                downsample = nn.Conv2d(self.block[index-1]*4,self.block[index]*4,1)
            flag = True
            for num in range(self.conv_num[index]):
                if flag:
                    if index!=0:
                        layers.append(Bottleneck_50_101_152(self.block[index-1]*4,self.block[index],downsample))
                    else:
                        layers.append(Bottleneck_50_101_152(64,self.block[index],downsample))
                    flag=False
                else:
                    layers.append(Bottleneck_50_101_152(self.block[index]*4,self.block[index],None))

        return layers

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.modellist:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dense(x)

        return x

class ResNet152(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(ResNet152,self).__init__()
        self.block = [64,128,256,512]
        self.conv_num = [3,8,36,3]
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.modellist = self._construct_layer()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(7*7*512,num_classes)

    def _construct_layer(self):
        layers = nn.ModuleList()
        downsample = None
        for index in range(4):
            if index != 0:
                downsample = nn.Conv2d(self.block[index-1]*4,self.block[index]*4,1)
            flag = True
            for num in range(self.conv_num[index]):
                if flag:
                    if index!=0:
                        layers.append(Bottleneck_50_101_152(self.block[index-1]*4,self.block[index],downsample))
                    else:
                        layers.append(Bottleneck_50_101_152(64,self.block[index],downsample))
                    flag=False
                else:
                    layers.append(Bottleneck_50_101_152(self.block[index]*4,self.block[index],None))

        return layers

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.modellist:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dense(x)

        return x