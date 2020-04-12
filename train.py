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
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import VOCDataset
from yolov2 import YOLOv2
from utils import draw_boxes, get_detection_result

parser = arg_parse()
args = parser.parse_args()

# create some output folder if not exist
if not os.path.isdir(args.output):
    os.mkdir(args.output)
if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)

# define and load YOLOv2
net = YOLOv2(args)
if torch.cuda.is_available():
    net.cuda()

if not args.load_model:
    net.load_weight()
else:
    model_path = os.path.join(args.model_dir, args.model_name)
    net.load_state_dict(torch.load(model_path))
    print("Load full model : ", model_path)

def train():

    net.train()

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    # create data batch generator
    training_set = VOCDataset("D:/dataset/VOC/VOCdevkit/", "2012", "train", image_size=net.IMAGE_W)
    dataloader = DataLoader(training_set, shuffle= True, batch_size=net.BATCH_SIZE)
    
    N_ITERS_PER_EPOCH = len(dataloader)

    writer = SummaryWriter()

    if torch.cuda.is_available():
        writer.add_graph(net.cpu(), torch.rand(4, 3, 416, 416))
    else:
        writer.add_graph(net, torch.rand(4, 3, 416, 416))

    for epoch in range(args.epoch):
        for step, (images, labels) in enumerate(dataloader):

            if images.shape[0] != net.BATCH_SIZE:
                continue

            print("")
            print("========== Epoch: {}, step: {}/{} ==========".format(epoch, step, N_ITERS_PER_EPOCH))

            time_start = time.time()

            if torch.cuda.is_available():
                image = Variable(images.cuda(), requires_grad=True)
            else:
                image = Variable(images, requires_grad=True)

            optimizer.zero_grad()
            output = net.forward(images)

            loss_xy, loss_wh, loss_conf, loss_cls = net.loss(output, labels)
            loss_coord = loss_xy + loss_wh
            total_loss = loss_coord + loss_conf + loss_cls

            total_loss.backward()
            optimizer.step()

            

            total_loss, loss_xy, loss_wh, loss_conf, loss_cls = [l.item() for l in [total_loss, loss_xy, loss_wh, loss_conf, loss_cls]]

            

            ### logs to tensorboard
            writer.add_scalar('Train/Total_loss', total_loss, epoch * N_ITERS_PER_EPOCH + step)
            writer.add_scalar('Train/Coordination_xy_loss', loss_xy, epoch * N_ITERS_PER_EPOCH + step)
            writer.add_scalar('Train/Coordination_wh_loss', loss_wh, epoch * N_ITERS_PER_EPOCH + step)
            writer.add_scalar('Train/Confidence_loss', loss_conf, epoch * N_ITERS_PER_EPOCH + step)
            writer.add_scalar('Train/Class_loss', loss_cls, epoch * N_ITERS_PER_EPOCH + step)

            ### log to console
            print('- Train step time: {} seconds'.format(time.time() - time_start))
            print('- Train/Coordination_xy_loss: ', loss_xy)
            print('- Train/Coordination_wh_loss: ', loss_wh)
            print('- Train/Confidence_loss: ', loss_conf)
            print('- Train/Class_loss: ', loss_cls)
            print('- Train/Total_loss: ', total_loss)

            if step % 10 == 0:
                boxes = get_detection_result(output, net.ANCHORS, net.CLASS, conf_thres=0.5, nms_thres=0.4)

                # draw detected boxes and save sample
                im = images[0].data.numpy().astype('uint8')
                im = im.transpose(1,2,0)
                im = im.copy()
                color_red = (0, 0, 255)
                color_green = (0, 255, 0)
                im = draw_boxes(im, labels[0], net.LABELS, color=color_green)
                im = draw_boxes(im, boxes[0], net.LABELS, color = color_red)

                file_path = os.path.join(args.output, "result_epoch_{}_iter_{}.jpg".format(epoch, step))
                cv2.imwrite(file_path, im)

        ### save model
        model_path = os.path.join(args.model_dir, "yolov2_epoch_{}.weights".format(epoch))
        torch.save(net.state_dict(), model_path)
        print("Saved model: ", model_path)

    writer.close()

if __name__ == "__main__":

    train()

    