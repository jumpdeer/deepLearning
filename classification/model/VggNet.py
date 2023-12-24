import torch.nn as nn
import torch.nn.functional as F



class vgg11(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg11,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.conv4 = nn.Conv2d(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(256,512,3,1,1)
        self.conv6 = nn.Conv2d(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv7 = nn.Conv2d(512,512,3,1,1)
        self.conv8 = nn.Conv2d(512,512,3,1,1)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        self.dropout = nn.Dropout(p=0.5)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool4(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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


class vgg11_LRN(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg11_LRN,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.LRN = nn.LocalResponseNorm(5)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.conv4 = nn.Conv2d(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(256,512,3,1,1)
        self.conv6 = nn.Conv2d(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv7 = nn.Conv2d(512,512,3,1,1)
        self.conv8 = nn.Conv2d(512,512,3,1,1)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.LRN(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool4(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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


class vgg13(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg13,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv7 = nn.Conv2d(256,512,3,1,1)
        self.conv8 = nn.Conv2d(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,3,1,1)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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


class vgg16(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg16,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.conv7 = nn.Conv2d(256,256,1,1,0)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv8 = nn.Conv2d(256,512,3,1,1)
        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,1,1,0)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        self.conv13 = nn.Conv2d(512,512,1,1,0)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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


class vgg16_second(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg16_second,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.conv7 = nn.Conv2d(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv8 = nn.Conv2d(256,512,3,1,1)
        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        self.conv13 = nn.Conv2d(512,512,3,1,1)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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


class vgg19(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(vgg19,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.conv7 = nn.Conv2d(256,256,3,1,1)
        self.conv8 = nn.Conv2d(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv9 = nn.Conv2d(256,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,3,1,1)
        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv13 = nn.Conv2d(512,512,3,1,1)
        self.conv14 = nn.Conv2d(512,512,3,1,1)
        self.conv15 = nn.Conv2d(512,512,3,1,1)
        self.conv16 = nn.Conv2d(512,512,3,1,1)
        self.pool5 = nn.MaxPool2d(2,2)


        self.Linear1 = nn.Linear(7*7*512,4096)
        self.Linear2 = nn.Linear(4096,4096)
        self.Linear3 = nn.Linear(4096,num_classes)

        if init_weight == True:
            self._initialize_weights()


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool4(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

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