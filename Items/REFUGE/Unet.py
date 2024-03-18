import torch.nn as nn
import torchvision.transforms.functional as F
import torch

# class DownSample(nn.Module):
#     def __init__(self,in_channels:int,out_channels:int):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,3)
#         self.conv2 = nn.Conv2d(out_channels,out_channels,3)
#         self.pool = nn.MaxPool2d(2,2)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         PFMap = F.relu(x)
#
#         downMap = self.pool(PFMap)
#
#         return PFMap,downMap
#
#
# class UpSample(nn.Module):
#     def __init__(self,in_channels:int,out_channels:int):
#         super().__init__()
#         self.transConv1 = nn.ConvTranspose2d(in_channels,out_channels,2,)
#         self.conv1 = nn.Conv2d(2*out_channels,out_channels,3)
#         self.conv2 = nn.Conv2d(out_channels,out_channels,3)
#
#     def forward(self, x, PFMap:torch.Tensor):
#         x = self.transConv1(x)
#
#         PFMap = TransF.center_crop(PFMap,[x.shape[2],x.shape[3]])
#         x = torch.cat((PFMap,x),1)
#
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#
#         return x
#
#
# class Unet(nn.Module):
#     def __init__(self,num_classes):
#         super().__init__()
#         self.down1 = DownSample(3,64)
#         self.down2 = DownSample(64,128)
#         self.down3 = DownSample(128,256)
#         self.down4 = DownSample(256,512)
#
#         self.conv1 = nn.Conv2d(512,1024,3)
#         self.conv2 = nn.Conv2d(1024,1024,3)
#
#         self.up1 = UpSample(1024,512)
#         self.up2 = UpSample(512,256)
#         self.up3 = UpSample(256,128)
#         self.up4 = UpSample(128,64)
#
#         self.conv3 = nn.Conv2d(64,num_classes,1)
#
#     def forward(self, x):
#
#         PFMap1,x = self.down1(x)
#         PFMap2,x = self.down2(x)
#         PFMap3,x = self.down3(x)
#         PFMap4,x = self.down4(x)
#
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#
#         x = self.up1(x,PFMap4)
#         x = self.up2(x,PFMap3)
#         x = self.up3(x,PFMap2)
#         x = self.up4(x,PFMap1)
#
#         x = self.conv3(x)
#
#         return x

class DoubleConvolution(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.first = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.act2 = nn.ReLU()

    def forward(self,x:torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x:torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)

    def forward(self,x:torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    def forward(self,x:torch.Tensor,contracting_x:torch.Tensor):
        contracting_x = F.center_crop(contracting_x,[x.shape[2],x.shape[3]])
        x = torch.cat([x,contracting_x],dim=1)

        return x


class Unet(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConvolution(i,o) for i,o in [(in_channels,32),(32,64),(64,128),(128,256)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        self.middle_conv = DoubleConvolution(256,512)

        self.up_sample = nn.ModuleList([UpSample(i,o) for i,o in [(512,256),(256,128),(128,64),(64,32)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i,o) for i,o in [(512,256),(256,128),(128,64),(64,32)]])

        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv2d(32,out_channels,kernel_size=1)


    def forward(self,x:torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x,pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)

        return x


