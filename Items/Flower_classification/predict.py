import torch
import torchvision.transforms as transforms
from PIL import Image
import csv
from model import AlexNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ])

    classes = ('daisy','dandelion','rose','sunflower','tulip')


    net = AlexNet(num_classes=5)
    net.load_state_dict(torch.load('Alexnet.pth'))

    sum=0
    right = 0
    with torch.no_grad():
        with open('Dataset/test_label.csv', 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                sum+=1   # 总数加1
                filename = row['imageName']
                label = row['label']

                im = Image.open('./Dataset/test_image/'+filename)
                im = transform(im)    # [C, H, W]
                im = torch.unsqueeze(im,dim=0)  # [N, C, H, W]

                outputs = net(im)
                predict = torch.max(outputs, dim=1)[1].data.numpy()

                if int(label) == int(predict):
                    right+=1

    print("accuracy of model is :"+str(right/sum))


if __name__ == '__main__':
    main()
