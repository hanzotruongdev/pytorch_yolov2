#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------
import torch
from args import arg_parse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import cv2
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import VOCDataset
from yolov2 import YOLOv2
from utils import draw_boxes, get_detection_result

parser = arg_parse()
args = parser.parse_args()

# define and load YOLOv2
net = YOLOv2(args)
net.load_weight()

def train():

    net.train()

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    # create data batch generator
    training_set = VOCDataset("D:/dataset/VOC/VOCdevkit/", "2012", "train", image_size=net.IMAGE_W)
    dataloader = DataLoader(training_set, shuffle= True, batch_size=net.BATCH_SIZE)
    
    N_ITERS_PER_EPOCH = len(dataloader) // net.BATCH_SIZE

    writer = SummaryWriter()
    writer.add_graph(net, torch.rand(4, 3, 416, 416))

    for epoch in range(args.epoch):
        
        for batch_idx, (images, labels) in enumerate(dataloader):

            print("")
            print("========== Epoch: {}, step: {} ==========".format(epoch, batch_idx))

            if torch.cuda.is_available():
                image = Variable(images.cuda(), requires_grad=True)
            else:
                image = Variable(images, requires_grad=True)

            optimizer.zero_grad()
            output = net.forward(images)

            loss, loss_coord, loss_conf, loss_cls = net.loss(output, labels)
            loss.backward()
            optimizer.step()

            loss, loss_coord, loss_conf, loss_cls = [l.item() for l in [loss, loss_coord, loss_conf, loss_cls]]

            ### logs to tensorboard
            writer.add_scalar('Train/Total_loss', loss, epoch * N_ITERS_PER_EPOCH + batch_idx)
            writer.add_scalar('Train/Coordination_loss', loss_coord, epoch * N_ITERS_PER_EPOCH + batch_idx)
            writer.add_scalar('Train/Confidence_loss', loss_conf, epoch * N_ITERS_PER_EPOCH + batch_idx)
            writer.add_scalar('Train/Class_loss', loss_cls, epoch * N_ITERS_PER_EPOCH + batch_idx)

            ### log to console
            print('- Train/Total_loss: ', loss)
            print('- Train/Coordination_loss: ', loss_coord)
            print('- Train/Confidence_loss: ', loss_conf)
            print('- Train/Class_loss: ', loss_cls)

            if batch_idx % 10 == 0:
                boxes = get_detection_result(output, net.ANCHORS, net.CLASS, conf_thres=.8, nms_thres=0.4)

                # draw detected boxes and save sample
                im = images[0].data.numpy().astype('uint8')
                im = im.transpose(1,2,0)
                im = im.copy()
                color_red = (0, 0, 255)
                color_green = (0, 255, 0)
                im = draw_boxes(im, labels[0], net.LABELS, color=color_green)
                im = draw_boxes(im, boxes[0], net.LABELS, color = color_red)

                file_path = os.path.join(args.output, "result_epoch_{}_iter_{}.jpg".format(epoch, batch_idx))
                cv2.imwrite(file_path, im)

    writer.close()

if __name__ == "__main__":
    train()

    