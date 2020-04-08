#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------
import torch
import args
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import VOCDataset
from yolov2 import YOLOv2

args_ = args.arg_parse()

# define and load YOLOv2
net = YOLOv2(args_)
net.load_weight("./darknet19_448.conv.23")

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0001)

training_set = VOCDataset("D:\\dataset\\VOC\\VOCdevkit", "2012", "train", image_size=416)
dataloader = DataLoader(training_set, batch_size=4)

def train():
    net.train()

    for batch_idx, (images, labels) in enumerate(dataloader):
        print("===============> step: {} <=================".format(batch_idx))
        optimizer.zero_grad()
        output = net.forward(images)
        loss = net.loss(output, labels)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()

    