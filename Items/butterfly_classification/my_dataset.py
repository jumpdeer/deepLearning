import csv

from torch.utils.data import Dataset
from PIL import Image
import os




class butterfly_set(Dataset):
    def __init__(self,data_dir,csv_file,transform):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.label_map = self._label_to_int()
        self.data = self._load_data()


    def _load_data(self):    # 数据加载类
        data = []
        with open(self.csv_file,'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                img_filename = row['filename']      # 图像位置
                label = row['label']                              # 标签

                label = self.label_map[label]

                data.append([img_filename,label])

            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_filename,label = self.data[index]

        # 加载图像
        img = Image.open(os.path.join(self.data_dir,img_filename))

        # 可选的数据预处理
        if self.transform:
            img = self.transform(img)

        return img,label

    def _label_to_int(self):
        classes_map = {
            'ADONIS':0, 'BROWN SIPROETA':1, 'MONARCH':2, 'GREEN CELLED CATTLEHEART':3, 'CAIRNS BIRDWING':4,
            'EASTERN DAPPLE WHITE':5,
            'RED POSTMAN':6, 'MANGROVE SKIPPER':7,
            'BLACK HAIRSTREAK':8, 'CABBAGE WHITE':9, 'RED ADMIRAL':10, 'PAINTED LADY':11, 'PAPER KITE':12, 'SOOTYWING':13, 'PINE WHITE':14,
            'PEACOCK':15, 'CHECQUERED SKIPPER':16, 'JULIA':17,
            'COMMON WOOD-NYMPH':18, 'BLUE MORPHO':19, 'CLOUDED SULPHUR':20, 'STRAITED QUEEN':21, 'ORANGE OAKLEAF':22,
            'PURPLISH COPPER':23,
            'ATALA':24, 'IPHICLUS SISTER':25, 'DANAID EGGFLY':26,
            'LARGE MARBLE':27, 'PIPEVINE SWALLOW':28, 'BLUE SPOTTED CROW':29, 'RED CRACKER':30, 'QUESTION MARK':31, 'CRIMSON PATCH':32,
            'BANDED PEACOCK':33, 'SCARCE SWALLOW':34, 'COPPER TAIL':35,
            'GREAT JAY':36, 'INDRA SWALLOW':37, 'VICEROY':38, 'MALACHITE':39, 'APPOLLO':40, 'TWO BARRED FLASHER':41, 'MOURNING CLOAK':42,
            'TROPICAL LEAFWING':43, 'POPINJAY':44, 'ORANGE TIP':45,
            'GOLD BANDED':46, 'BECKERS WHITE':47, 'RED SPOTTED PURPLE':48, 'MILBERTS TORTOISESHELL':49, 'SILVER SPOT SKIPPER':50,
            'AMERICAN SNOOT':51, 'AN 88':52, 'ULYSES':53, 'COMMON BANDED AWL':54,
            'CRECENT':55, 'SOUTHERN DOGFACE':56, 'METALMARK':57, 'SLEEPY ORANGE':58, 'PURPLE HAIRSTREAK':59, 'ELBOWED PIERROT':60,
            'GREAT EGGFLY':61,
            'ORCHARD SWALLOW':62, 'ZEBRA LONG WING':63,
            'WOOD SATYR':64, 'MESTRA':65, 'EASTERN PINE ELFIN':66, 'EASTERN COMA':67, 'YELLOW SWALLOW TAIL':68, 'CLEOPATRA':69,
            'GREY HAIRSTREAK':70,
            'BANDED ORANGE HELICONIAN':71, 'AFRICAN GIANT SWALLOWTAIL':72,
            'CHESTNUT':73, 'CLODIUS PARNASSIAN':74}

        return classes_map
