# from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from utils.helpers import get_upsampling_weight
import torch
from itertools import chain
import numpy as np


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2  # 获取卷积核的一半，不整向上取
    if kernel_size % 2 == 1:         # 如果卷积核为奇数
            center = factor - 1
    else:
            center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size] # 返回两个ndarray数组，分别是 kernerl_size * 1 和 1 * kernel_size
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

# class FCN8(nn.Module):
#     def __init__(self, num_classes, pretrained=True, freeze_bn=False, **_):
#         super(FCN8, self).__init__()
#         vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)       # 使用预训练好的vgg16模型
#         features = list(vgg.features.children())    # 获取features层
#         classifier = list(vgg.classifier.children())  # 获取classifier层
#
#         # Pad the input to enable small inputs and allow matching feature maps
#         # features[0].padding = (100, 100)   # features中的第一个卷积层进行像素填充(100,100)
#
#         # Enbale ceil in max pool, to avoid different sizes when upsampling
#         # 池化层自动补足
#         for layer in features:
#             if 'MaxPool' in layer.__class__.__name__:
#                 layer.ceil_mode = True
#
#         # Extract pool3, pool4 and pool5 from the VGG net
#         self.pool3 = nn.Sequential(*features[:17])   # 定义提取features层的前17层
#         '''
#         (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (1): ReLU(inplace=True)
#         (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (3): ReLU(inplace=True)
#         (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (6): ReLU(inplace=True)
#         (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (8): ReLU(inplace=True)
#         (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (11): ReLU(inplace=True)
#         (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (13): ReLU(inplace=True)
#         (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (15): ReLU(inplace=True)
#         (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         '''
#         # 输出 28x28x256       八分之一图像输入
#
#         self.pool4 = nn.Sequential(*features[17:24])  # 提取从17~23
#         '''
#         (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (18): ReLU(inplace=True)
#         (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (20): ReLU(inplace=True)
#         (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (22): ReLU(inplace=True)
#         (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         '''
#         # 输出 14x14x512       十六分之一图像输入
#
#         self.pool5 = nn.Sequential(*features[24:]) # 提取24~30
#         '''
#         (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (25): ReLU(inplace=True)
#         (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (27): ReLU(inplace=True)
#         (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (29): ReLU(inplace=True)
#         (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         '''
#         # 输出 7x7x512        三十二分之一图像输入
#
#         # Adjust the depth of pool3 and pool4 to num_classes
#         self.adj_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)  # 创建 输入：256，输出：num_classes,卷积核为1x1 的卷积层，作用为调节通道数
#         self.adj_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)  # 创建 输入：512，输出：num_classes,卷积核为1x1 的卷积层，作用为调节通道数
#
#         # Replace the FC layer of VGG with conv layers
#         conv6 = nn.Conv2d(512, 4096, kernel_size=7)  # 创建 输入：512，输出：4096，卷积核为7x7 的卷积层
#         conv7 = nn.Conv2d(4096, 4096, kernel_size=1)  # 创建 输入：4096，输出：4096，卷积核为1x1 的卷积层
#         output = nn.Conv2d(4096, num_classes, kernel_size=1)  # 创建 输入：4096，输出：num_classes，卷积核为1x1 的卷积层
#
#         # Copy the weights from VGG's FC pretrained layers
#         conv6.weight.data.copy_(classifier[0].weight.data.view(
#             conv6.weight.data.size()))              # conv6从classifier层的第0层复制参数
#         '''
#         (0): Linear(in_features=25088, out_features=4096, bias=True)
#         '''
#
#         conv6.bias.data.copy_(classifier[0].bias.data)  # conv6复制偏置值从classifier层的第0层
#
#         conv7.weight.data.copy_(classifier[3].weight.data.view(
#             conv7.weight.data.size()))              # conv7从classifier层的第3层复制参数
#         conv7.bias.data.copy_(classifier[3].bias.data)
#         '''
#         (3): Linear(in_features=4096, out_features=4096, bias=True)
#         '''
#
#         # Get the outputs
#         self.output = nn.Sequential(conv6, nn.ReLU(inplace=True), nn.Dropout(),
#                                     conv7, nn.ReLU(inplace=True), nn.Dropout(),
#                                     output)  # 创建一个输出层，里面包含上面的conv6,relu层,dropout层，conv7,relu层,dropout层,output层
#
#
#         # We'll need three upsampling layers, upsampling (x2 +2) the ouputs
#         # upsampling (x2 +2) addition of pool4 and upsampled output
#         # upsampling (x8 +8) the final value (pool3 + added output and pool4)
#         self.up_output = nn.ConvTranspose2d(num_classes, num_classes,
#                                             kernel_size=4, stride=2, bias=False)   # 创建一个输入：num_classes，输出：num_classes，卷积核为4x4，步长为2的转置卷积
#         self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes,
#                                                kernel_size=4, stride=2, bias=False)  # 创建一个输入：num_classes，输出：num_classes，卷积核为4x4，步长为2的转置卷积
#         self.up_final = nn.ConvTranspose2d(num_classes, num_classes,
#                                            kernel_size=16, stride=8, bias=False)  # 创建一个输入：num_classes，输出：num_classes，卷积核为16*16，步长为2的转置卷积
#
#         # We'll use guassian kernels for the upsampling weights
#         self.up_output.weight.data.copy_(
#             get_upsampling_weight(num_classes, num_classes, 4))   #对up_output的参数进行初始化
#         self.up_pool4_out.weight.data.copy_(
#             get_upsampling_weight(num_classes, num_classes, 4))   #对up_output的参数进行初始化
#         self.up_final.weight.data.copy_(
#             get_upsampling_weight(num_classes, num_classes, 16))  #对up_output的参数进行初始化
#
#         # We'll freeze the wights, this is a fixed upsampling and not deconv
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 m.weight.requires_grad = False       # 将转置卷积的权重设置为不参与梯度计算
#         if freeze_bn: self.freeze_bn()
#         # if freeze_backbone:
#         #     set_trainable([self.pool3, self.pool4, self.pool5], False)
#
#     def forward(self, x):
#         imh_H, img_W = x.size()[2], x.size()[3]  # 获取输入张量的高和宽
#
#         # Forward the image
#         pool3 = self.pool3(x)             # 开始前向传播
#         pool4 = self.pool4(pool3)
#         pool5 = self.pool5(pool4)
#
#         # Get the outputs and upsmaple them
#         output = self.output(pool5)       # output层计算
#         up_output = self.up_output(output)    # 进行up_output层计算(转置卷积)
#
#         # Adjust pool4 and add the uped-outputs to pool4
#         adjstd_pool4 = self.adj_pool4(0.01 * pool4)    # 使用self.adj_pool4对通道数进行调节，将原特征图的十六分之一乘0.01然后输入
#         add_out_pool4 = self.up_pool4_out(adjstd_pool4[:, :, 5: (5 + up_output.size()[2]),
#                                           5: (5 + up_output.size()[3])]
#                                           + up_output)       # 根据论文中FCN，进行特征图的相加，然后进行上采样
#
#         # Adjust pool3 and add it to the uped last addition
#         adjstd_pool3 = self.adj_pool3(0.0001 * pool3)   # 使用self.adj_pool3对通道数调节，将原特征图的八分之一乘0.0001然后输入
#         final_value = self.up_final(
#             adjstd_pool3[:, :, 9: (9 + add_out_pool4.size()[2]), 9: (9 + add_out_pool4.size()[3])]
#             + add_out_pool4)   # 上采样，特征图相加，上采样
#
#         # Remove the corresponding padded regions to the input img size
#         final_value = final_value[:, :, 31: (31 + imh_H), 31: (31 + img_W)].contiguous()  # 最后输出
#         return final_value

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = models.vgg19(weights=models.VGG19_Weights).features  # 去除全连接层

        self.ConvTrans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        self.ConvTrans2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.ConvTrans3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)  # 1x1卷积， 在像素级别进行分类
        # 将对应的池化层存入字典，方便到时候提取该层的特征进行求和：
        self.layers = {'18': 'maxpool_3', '27': 'maxpool_4', '36': 'maxpool_5', }

    def forward(self, x):
        output = {}  # 用来保存中间层的特征
        # 首先利用预训练的VGG19提取特征：
        for name, layer in self.base_model._modules.items():
            x = layer(x)

            # 如果当前层的特征需要被保存：
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output['maxpool_5']  # 原图的H/32, W/32
        x4 = output['maxpool_4']  # 原图的H/16, W/16
        x3 = output['maxpool_3']  # 原图的H/ 8, W/ 8

        # 对特征进行相关转置卷积操作，逐渐恢复到原图大小:
        score = self.ConvTrans1(x5)  # 提取maxpool_5的特征，转置卷积进行上采样，激活函数输出
        score = self.ConvTrans2(score + x4)  # 上采样后的特征再与maxpool_4的特征相加，并进行归一化操作
        score = self.ConvTrans3(score + x3)  # score
        score = self.classifier(score)

        return score

    def get_backbone_params(self):
        return chain(self.pool3.parameters(), self.pool4.parameters(), self.pool5.parameters(),
                     self.output.parameters())

    def get_decoder_params(self):
        return chain(self.up_output.parameters(), self.adj_pool4.parameters(), self.up_pool4_out.parameters(),
                     self.adj_pool3.parameters(), self.up_final.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()    