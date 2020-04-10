#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------
import torch
import args
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import VOCDataset
from yolov2 import YOLOv2
from utils import draw_boxes, get_detection_result

args_ = args.arg_parse()

# define and load YOLOv2
net = YOLOv2(args_)
net.load_weight("./darknet19_448.conv.23")
net.train()

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)

training_set = VOCDataset("D:\\dataset\\VOC\\VOCdevkit", "2012", "train", image_size=416)
dataloader = DataLoader(training_set, shuffle= True, batch_size=2)

def train():

    writer = SummaryWriter()
    writer.add_graph(net, torch.rand(4, 3, 416, 416))

    for batch_idx, (images, labels) in enumerate(dataloader):
        print("===============> step: {} <=================".format(batch_idx))
        images, labels = Variable(images, requires_grad = True), labels

        optimizer.zero_grad()
        output = net.forward(images)
        print("output shape: ", output.shape)
        loss, loss_coord, loss_conf, loss_cls = net.loss(output, labels)
        loss.backward()
        optimizer.step()
        

        writer.add_scalar('Train/Total_loss', loss, batch_idx)
        writer.add_scalar('Train/Coordination_loss', loss_coord, batch_idx)
        writer.add_scalar('Train/Confidence_loss', loss_conf, batch_idx)
        writer.add_scalar('Train/Class_loss', loss_cls, batch_idx)

        if batch_idx % 10 == 0:
            boxes = get_detection_result(output, conf_thres=.8, nms_thres=0.4)

            # draw detected boxes and save sample
            im = images[0].data.numpy().astype('uint8')
            im = im.transpose(1,2,0)
            im = im.copy()

            color_red = (0, 0, 255)
            color_green = (0, 255, 0)
            im = draw_boxes(im, labels[0], color=color_green)
            im = draw_boxes(im, boxes[0], color = color_red)
            cv2.imwrite("result/result_iter_{}.jpg".format(batch_idx), im)

    writer.close()

if __name__ == "__main__":
    train()

    