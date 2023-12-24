import xml.etree.ElementTree as ET
import csv
import os
import cv2 as cv

image_path = 'archive/dataset-master/dataset-master/JPEGImages/'
annotation_path = 'archive/dataset-master/dataset-master/Annotations/'
LabelImage_path = 'archive/dataset-master/dataset-master/Label/'


class Connect:


    def __init__(self):
        self.image_files = []
        self.annotation_files = []
        for root,dirs,files in os.walk(image_path):
            self.image_files = files

        for root,dirs,files in os.walk(annotation_path):
            self.annotation_files = files


    def imgAndAnno(self):
        connectList = []
        for i in range(len(self.image_files)):
            str = self.image_files[i].split('.')[0]+'.xml'
            for j in range(len(self.annotation_files)):
                if str == self.annotation_files[j]:
                    connectList.append((self.image_files[i],self.annotation_files[j]))
                    break

        return connectList

def xmltolabel():
    conn = Connect()
    for con in conn.imgAndAnno():
        label_name = LabelImage_path + con[0].split('.')[0]+'.png'
        img = cv.imread(image_path+con[0])
        root = ET.parse(annotation_path+con[1]).getroot()
        objects = root.findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text.strip()))
            ymin = int(float(bbox.find('ymin').text.strip()))
            xmax = int(float(bbox.find('xmax').text.strip()))
            ymax = int(float(bbox.find('ymax').text.strip()))
            label_image = cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv.imwrite(label_name,label_image)

if __name__ == '__main__':
    xmltolabel()

