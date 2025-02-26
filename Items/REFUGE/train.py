import torch.utils.data
import numpy as np
from torch import nn,optim
from DiceLoss import DiceLoss
import torch.nn.functional as F
from lossFunction.FocalLoss import CrossEntropyFocalLoss
from Unet import Unet
from REFUGE_DataSet import REFUGE_Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional



def Iou(target_all, pred_all,n_class):
    """
        target是真实标签，shape为(h,w)，像素值为0，1，2...
        pred是预测结果，shape为(h,w)，像素值为0，1，2...
        n_class:为预测类别数量
        """
    pred_all = pred_all.to('cpu')
    target_all = target_all.to('cpu')
    iou = []
    for i in range(target_all.shape[0]):
        pred = pred_all[i]
        target = target_all[i]

        h, w = target.shape
        # 转为one-hot，shape变为(h,w,n_class)
        target_one_hot = np.eye(n_class)[target]
        pred_one_hot = np.eye(n_class)[pred]

        target_one_hot[target_one_hot != 0] = 1
        pred_one_hot[pred_one_hot != 0] = 1
        join_result = target_one_hot * pred_one_hot

        join_sum = np.sum(np.where(join_result == 1))  # 计算相交的像素数量
        pred_sum = np.sum(np.where(pred_one_hot == 1))  # 计算预测结果非0得像素数
        target_sum = np.sum(np.where(target_one_hot == 1))  # 计算真实标签的非0得像素数

        iou.append(join_sum / (pred_sum + target_sum - join_sum + 1e-6))

    return np.mean(iou)

def cropOutputImage(image,ori_size):
    new_image = torchvision.transforms.functional.center_crop(image,ori_size)
    return new_image

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    print("using {} device.".format(device))

    train_set = REFUGE_Dataset('./data/Train400/AugmentData-Training400/','./data/Train400/AugmentMask-Training400/', 3,transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)



    net = Unet(in_channels=3,out_channels=3)
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = DiceLoss()
    # loss_function = CrossEntropyFocalLoss([2,0.5,0.5],0.5,'mean',device)
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    net.train()

    for epoch in range(200):

        running_loss = 0.0
        for step,data in enumerate(train_loader,start=0):
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            outputs = F.log_softmax(outputs, dim=1) #结果输出值尺寸为(1,3,2048,2048)  标签尺寸为(1,2048,2048)
            #
            pre_lab = torch.argmax(outputs, 1)
            # print(pre_lab.shape)

            loss = loss_function(outputs,labels)

            # print(loss)

            loss.requires_grad_(True)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            PA = torch.sum(pre_lab == labels.data)/(2048*2048)

            print('epoch:{} | step:{} | val loss:{:.5f} | PA:{:.5f} | MIOU:{:.5f}'.format(epoch, step, loss.item(),PA, Iou(pre_lab,labels,3)))



    save_path = './Unet.pth'
    torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    main()