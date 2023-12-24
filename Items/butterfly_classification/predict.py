import torch
import torchvision.transforms as transforms
from PIL import Image

from Items.butterfly_classification.model import AlexNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ])

    classes = (
    'ADONIS', 'BROWN SIPROETA', 'MONARCH', 'GREEN CELLED CATTLEHEART', 'CAIRNS BIRDWING', 'EASTERN DAPPLE WHITE',
    'RED POSTMAN', 'MANGROVE SKIPPER',
    'BLACK HAIRSTREAK', 'CABBAGE WHITE', 'RED ADMIRAL', 'PAINTED LADY', 'PAPER KITE', 'SOOTYWING', 'PINE WHITE',
    'PEACOCK', 'CHECQUERED SKIPPER', 'JULIA',
    'COMMON WOOD-NYMPH', 'BLUE MORPHO', 'CLOUDED SULPHUR', 'STRAITED QUEEN', 'ORANGE OAKLEAF', 'PURPLISH COPPER',
    'ATALA', 'IPHICLUS SISTER', 'DANAID EGGFLY',
    'LARGE MARBLE', 'PIPEVINE SWALLOW', 'BLUE SPOTTED CROW', 'RED CRACKER', 'QUESTION MARK', 'CRIMSON PATCH',
    'BANDED PEACOCK', 'SCARCE SWALLOW', 'COPPER TAIL',
    'GREAT JAY', 'INDRA SWALLOW', 'VICEROY', 'MALACHITE', 'APPOLLO', 'TWO BARRED FLASHER', 'MOURNING CLOAK',
    'TROPICAL LEAFWING', 'POPINJAY', 'ORANGE TIP',
    'GOLD BANDED', 'BECKERS WHITE', 'RED SPOTTED PURPLE', 'MILBERTS TORTOISESHELL', 'SILVER SPOT SKIPPER',
    'AMERICAN SNOOT', 'AN 88', 'ULYSES', 'COMMON BANDED AWL',
    'CRECENT', 'SOUTHERN DOGFACE', 'METALMARK', 'SLEEPY ORANGE', 'PURPLE HAIRSTREAK', 'ELBOWED PIERROT', 'GREAT EGGFLY',
    'ORCHARD SWALLOW', 'ZEBRA LONG WING',
    'WOOD SATYR', 'MESTRA', 'EASTERN PINE ELFIN', 'EASTERN COMA', 'YELLOW SWALLOW TAIL', 'CLEOPATRA', 'GREY HAIRSTREAK',
    'BANDED ORANGE HELICONIAN', 'AFRICAN GIANT SWALLOWTAIL',
    'CHESTNUT', 'CLODIUS PARNASSIAN')


    net = AlexNet()
    net.load_state_dict(torch.load('Alexnet.pth'))

    im = Image.open('archive/test/Image_1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
