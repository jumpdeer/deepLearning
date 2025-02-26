import os

import cv2
from albumentations import HorizontalFlip,VerticalFlip,RandomRotate90

for root,dirs,files in os.walk('./data/Train400/Data-Training400'):
    for item in files:
        oldImage = cv2.imread('./data/Train400/Data-Training400/'+item,cv2.COLOR_BGR2RGB)
        oldMask = cv2.imread('./data/Train400/Mask-Training400/'+item.split('.')[0]+'.png',cv2.COLOR_BGR2GRAY)

        Horizontalresult = HorizontalFlip(always_apply=True,p=1.0)(image=oldImage,mask=oldMask)
        Verticalresult = VerticalFlip(p=1.0)(image=oldImage,mask=oldMask)
        Randomrotateresult1 = RandomRotate90(always_apply=False,p=0.5)(image=oldImage,mask=oldMask)
        Randomrotateresult2 = RandomRotate90(always_apply=False,p=0.5)(image=oldImage,mask=oldMask)

        # 保存水平翻转数据
        cv2.imwrite('./data/Train400/AugmentData-Training400/'+item.split('.')[0]+'Horizon.jpg',Horizontalresult['image'])
        cv2.imwrite('./data/Train400/AugmentMask-Training400/'+item.split('.')[0]+'Horizon.png',Horizontalresult['mask'])

        # 保存竖直翻转数据
        cv2.imwrite('./data/Train400/AugmentData-Training400/'+item.split('.')[0]+'Vertical.jpg',Verticalresult['image'])
        cv2.imwrite('./data/Train400/AugmentMask-Training400/' + item.split('.')[0] + 'Vertical.png',Verticalresult['mask'])

        # 保存随机旋转1数据
        cv2.imwrite('./data/Train400/AugmentData-Training400/' + item.split('.')[0] + 'Random1.jpg',Randomrotateresult1['image'])
        cv2.imwrite('./data/Train400/AugmentMask-Training400/' + item.split('.')[0] + 'Random1.png',Randomrotateresult1['mask'])

        # 保存随机旋转2数据
        cv2.imwrite('./data/Train400/AugmentData-Training400/' + item.split('.')[0] + 'Random2.jpg',Randomrotateresult2['image'])
        cv2.imwrite('./data/Train400/AugmentMask-Training400/' + item.split('.')[0] + 'Random2.png',Randomrotateresult2['mask'])

