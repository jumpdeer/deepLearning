import torchvision.transforms as transforms
import torch
from classification.model.VggNet import vgg19
from PIL import Image

def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    classes = ('正常','坑洼')

    net = vgg19(num_classes=2)
    net.load_state_dict(torch.load('vgg19Net.pth'))

    im = Image.open('1.jpg')
    im = transform(im)
    im = torch.unsqueeze(im,dim=0)

    with torch.no_grad():
        outputs = net(im)

    print(outputs)

if  __name__ == '__main__':
    main()