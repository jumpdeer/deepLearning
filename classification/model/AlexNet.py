import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,11,padding=2,stride=4)    # in:(224,224,3)     out:(55,55,64)
        self.pool1 = nn.MaxPool2d(3,2)      # in:(55,55,64)       out:(27,27,64)
        self.conv2 = nn.Conv2d(64,192,5,padding=2)            # in:(27,27,64)     out:(27,27,192)
        self.pool2 = nn.MaxPool2d(3,2)      # in:(27,27,192)        #(13,13,192)
        self.conv3 = nn.Conv2d(192,384,3,padding=1)     # in:(13,13,192)          out:(13,13,384)
        self.conv4 = nn.Conv2d(384,256,3,padding=1)     # in:(13,13,384)          out:(13,13,256)
        self.conv5 = nn.Conv2d(256,256,3,padding=1)     # in:(13,13,256)          out:(13,13,256)
        self.pool3 = nn.MaxPool2d(3,2)                  # in:(13,13,256)          out:(6,6,256)
        self.dense1 = nn.Linear(256*6*6,4096)
        self.dense2 = nn.Linear(4096,1024)
        self.dense3 = nn.Linear(1024,75)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(x.size(0),-1)

        x = F.dropout(x,p=0.6,inplace=False)
        x = F.relu(self.dense1(x))
        x = F.dropout(x,p=0.6,inplace=False)
        x = F.relu(self.dense2(x))
        x = F.dropout(x,p=0.6,inplace=False)
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