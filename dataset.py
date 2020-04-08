#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------

import os
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from utils import resize_image
import cv2
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, root_path="data/VOCdevkit", year="2012", mode="train", image_size=448):
        if mode in ["train", "test", "val"] and year in ["2012"]:
            self.data_path = os.path.join(root_path, "VOC{}".format(year))
        
        id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
        self.ids = [id.strip() for id in open(id_list_path)]

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        anno = ET.parse(image_xml_path)

        objects = []
        for obj in anno.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in ["xmin", "xmax", "ymin", "ymax"]]
            box_w = xmax - xmin
            box_h = ymax - ymin
            center_x = xmin + box_w / 2
            center_y = ymin + box_h / 2
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([center_x, center_y, box_w, box_h, label])

        # resize image
        image, objects = resize_image(image, objects, self.image_size)

        array_objects = np.array(objects, dtype=np.float32)
        t = np.zeros([50, 5])
        t[:array_objects.shape[0]] = array_objects

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), t


if __name__ == "__main__":
    training_set = VOCDataset("D:\\dataset\\VOC\\VOCdevkit", "2012", "train")
    dataloader = DataLoader(training_set, batch_size=16)

    print("Training len: ", len(dataloader))
    for iter, batch in enumerate(dataloader):
        image, label = batch
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("label: ", label)
        break
