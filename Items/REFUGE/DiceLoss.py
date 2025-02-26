import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,predict:torch.Tensor,mask:torch.Tensor):

        smooth = 1e-7

        batch_size = predict.shape[0]


        # for i in range(batch_size):
        #     pre_bin = predict[i,:,:].view(predict[i,:,:].size(0),-1)
        #     mask_bin = mask[i,:,:].view(mask[i,:,:].size(0),-1)
        #
        #     print(f'predict[i,:,:].size(0)的值是{predict[i,:,:].size(0)}')
        #     print(f'pre_bin是:{pre_bin}，其shape为:{pre_bin.shape}')
        #     print(f'mask_bin是:{mask_bin}，其shape为:{mask_bin.shape}')
        #
        #     interaction = (pre_bin*mask_bin).sum()
        #     dice = torch.add(2. * interaction,smooth)/torch.add(pre_bin.sum(),mask_bin.sum(),smooth)
        #
        #     allLoss += 1.-dice

        mask_bin = F.one_hot(mask).permute(0,3,1,2).reshape(batch_size,-1)
        predict_bin = F.softmax(predict,dim=1).reshape(batch_size,-1)

        interaction = torch.sum(mask_bin*predict_bin,dim=1)
        cardinality = torch.sum(mask_bin,dim=1)+torch.sum(predict_bin,dim=1)

        dice = 2*(interaction+smooth)/(cardinality+smooth)

        # print(f'分子的值为{torch.add(2.*interaction,smooth)}')
        # print(f'分母的值为{torch.add(torch.add(pre_bin[i].sum(),mask_bin[i].sum()),smooth)}')
        # print(f'dice系数为{torch.div(torch.add(2.*interaction,smooth),torch.add(torch.add(pre_bin[i].sum(),mask_bin[i].sum()),smooth))}')



        return 1-dice.mean()