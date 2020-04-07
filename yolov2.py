#--------------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# Written by Noi Truong email: noitq.hust@gmail.com
#--------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import args
from darknet import Darknet19, conv_bn_leaky

LABELS = []

IMAGE_W, IMAGE_H    = 320, 320
GRID_W, GRID_H      = 10, 10
BOX = 5
CLASS = len(LABELS)
OBJ_THRESHOLD       = 0.6
NMS_THRESHOLD       = 0.4
ANCHORS             = [[123, 324],[123, 324],[123, 324],[123, 324],[123, 324]]

# hyper parameters for loss function
LAMBDA_OBJECT       = 0.5
LAMBDA_NO_OBJECT    = 1.0
LAMBDA_COORD        = 1.0
LAMBDA_CLASS        = 1.0

BATCH_SIZE          = 8
TRUE_BOX_BUFFER     = 50

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x

class YOLOv2(nn.Module):
    def __init__(self, args):
        super(YOLOv2, self).__init__()
        self.num_classes = CLASS
        self.num_anchors = len(ANCHORS)
        darknet19 = Darknet19()

        # define elements used
        # darknet backbone
        self.conv1 = nn.Sequential(darknet19.layer0, darknet19.layer1,
                                   darknet19.layer2, darknet19.layer3, darknet19.layer4)

        self.conv2 = darknet19.layer5

        # detection layers
        self.conv3 = nn.Sequential(conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
                                   conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.conv4 = nn.Sequential(conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

        # reorg 
        self.reorg = ReorgLayer()

        return

    def forward(self, x):
        """
        Only output an feature map that didn't transform.
        """
        x = self.conv1(x)
        shortcut = self.reorg(self.downsampler(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([shortcut, x], dim=1)
        out = self.conv4(x)

        return out

    def loss(self, y_pred, y_true):
        """
        return YOLOv2 loss
        input:
            - pred: YOLOv2 prediction, the output feature map of forward() function
                    shape of [N, S*S*B, 5 + n_class]
            - label: ground truth in shape of [N, S*S*B, 5 + n_class]
        output:
            YOLOv2 loss includes coordinate loss, confidence score loss, and class loss.
        """

        # loss_xywh
        # loss_class
        # loss_class = F.nll_loss(pred_class_prob, true_class_id)
        # If i calculate this way, it will include the noobj to loss ... 
        # loss_confidence

    def load_weight(self, weight_file):
        pass

if __name__ == "__main__":
    args_ = args.arg_parse()
    net = YOLOv2(args_)
    # net.load_weight("./darknet19_448.conv.23")


    input = torch.randn(4, 3, IMAGE_H, IMAGE_W)
    output = net.forward(input)
    print("Input shape: ", input.shape)
    print("Output shape: ", output.shape)