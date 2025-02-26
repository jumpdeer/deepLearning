import torch
import torch.nn as nn
import torch.nn.functional as F


# 目前对batchsize为1合适
# class FocalLoss(nn.Module):
#     def __init__(self,gamma=0.,alpha=None,device='cpu'):
#         super().__init__()
#         self.gamma =gamma
#         self.alpha = alpha
#         self.device = device
#
#     # 默认背景标签为0
#     def forward(self,predict:torch.Tensor,mask:torch.Tensor):
#
#         batch_size = predict.shape[0]
#
#         mask_t = torch.zeros((predict.shape[1],mask.shape[1],mask.shape[2]))
#
#         for b in range(batch_size):
#             for i in range(mask.shape[1]):
#                 for j in range(mask.shape[2]):
#                     mask_t[mask[b,i,j],i,j]=1
#
#         CE = F.log_softmax(predict,dim=0)*mask_t
#
#         pt = torch.exp(-CE)
#
#         # 计算Focal Loss
#         loss = (1-pt)**self.gamma*CE   # 三维的判断特征图，第一个维度为每个分类（0，1，2），第二第三分别是长和宽
#
#         print(loss)
#         print(loss.shape)
#
#         # 如果有alpha参数，则进行权重调整
#         if self.alpha is not None:
#             alpha = torch.tensor(self.alpha,dtype=torch.float).to(self.device)
#
#             alphat = torch.tensor((loss.shape[0],loss.shape[1],loss.shape[2]))
#             for i in range(loss.shape[1]):
#                 alphat[i,:,:]=alpha[i]
#
#
#             loss = torch.sum(alphat*loss,dim=0)
#
#             print(loss)
#
#         return torch.mean(loss)

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean',device='cpu'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.device = device
        self.alpha = torch.tensor(alpha,dtype=torch.float).to(self.device)
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            # alpha: [C]
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


