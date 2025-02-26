import torch.nn as nn
import torchvision.transforms.functional as F
import torch

class ConvBatchRelu(nn.Module):
    def __init__(self,ins,outs):
        super(ConvBatchRelu,self).__init__()
        self.conv = nn.Conv2d(ins,outs,3,2,1)
        self.bn = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self,ins:int,outs:int):
        super(AttentionRefinementModule,self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(ins,outs,1)
        self.bn = nn.BatchNorm2d(ins)
        self.sg = nn.Sigmoid()
        self.upsam = nn.ConvTranspose2d(ins,outs,4,2,1)

    def forward(self, x):
        a = self.pool(x)
        a = self.conv(a)
        a = self.bn(a)
        a = self.sg(a)
        a = self.upsam(a)

        x = torch.mul(x,a)

        return x

class FeatureFusionModule(nn.Module):
    def __init__(self,ins,outs):
        super(FeatureFusionModule,self).__init__()
        self.convBnRelu = ConvBatchRelu(ins,outs)
        self.pool = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(ins,outs,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ins,outs,1)
        self.sg = nn.Sigmoid()
        self.upsam = nn.ConvTranspose2d(ins,outs,4,2,1)

    def forward(self,input1,input2):
        feamap = torch.concat((input1,input2),dim=1)
        feamap = self.convBnRelu(feamap)

        tmap = self.pool(feamap)
        tmap = self.conv1(tmap)
        tmap = self.relu(tmap)
        tmap = self.conv2(tmap)
        tmap = self.sg(tmap)
        tmap = self.upsam(tmap)

        feamap2 = torch.mul(feamap,tmap)

        featureMap = torch.add(feamap,feamap2)

        featureMap = self.upsam(featureMap)

        return featureMap



class ContextPath(nn.Module):
    def __init__(self,ins,outs):
        super(ContextPath,self).__init__()
        self.conv1 = nn.Conv2d(ins,64,5,2,2)
        self.conv2 = nn.Conv2d(64,128,3,2,1)
        self.conv3 = nn.Conv2d(128,outs,3,2,1)
        self.ARM1 = AttentionRefinementModule(outs,outs)
        self.conv4 = nn.Conv2d(outs,outs,3,2,1)
        self.ARM2 = AttentionRefinementModule(outs,outs)
        self.upconv1 = nn.ConvTranspose2d(outs,outs,4,2,1)



    def forward(self,x):
        x = self.conv1(x)

        x = self.conv2(x)

        down_16 = self.conv3(x)
        x1 = self.ARM1(down_16)

        down_32 = self.conv4(down_16)
        x2 = self.ARM2(down_32)
        x2 = torch.add(down_32,x2)
        x2 = self.upconv1(x2)

        res_16 = torch.add(x1,x2)


        return res_16


class SpatialPath(nn.Module):
    def __init__(self,ins,outs):
        super(SpatialPath,self).__init__()
        self.convBnRe1 = ConvBatchRelu(ins,64)
        self.convBnRe2 = ConvBatchRelu(64,128)
        self.convBnRe3 = ConvBatchRelu(128,outs)

    def forward(self,x):
        x = self.convBnRe1(x)
        x = self.convBnRe2(x)
        x = self.convBnRe3(x)

        return x


class BiSeNet(nn.Module):
    def __init__(self):
        super(BiSeNet,self).__init__()
        self.sp = SpatialPath(3,256)
        self.cp = ContextPath(3,256)
        self.ffm = FeatureFusionModule(512,512)
        self.up_sample1 = nn.ConvTranspose2d(512,19,4,2,1)
        self.up_sample2 = nn.ConvTranspose2d(19,19,4,2,1)
        self.up_sample3 = nn.ConvTranspose2d(19,19,4,2,1)

    def forward(self,input):
        out1 = self.sp(input)
        out2 = self.cp(input)

        result = self.ffm(out1,out2)

        result = self.up_sample1(result)
        result = self.up_sample2(result)
        result = self.up_sample3(result)

        return result


