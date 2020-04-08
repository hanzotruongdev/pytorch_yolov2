#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import args
from darknet import Darknet19, conv_bn_leaky
from utils import bbox_ious, BestAnchorFinder

# VOC 2012 dataset
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

CLASS = len(LABELS)

IMAGE_W, IMAGE_H    = 320, 320
GRID_W, GRID_H      = 10, 10

OBJ_THRESHOLD       = 0.6
NMS_THRESHOLD       = 0.4

ANCHORS             = [[123, 324],[123, 324],[123, 324],[123, 324],[123, 324]]
BOX                 = len(ANCHORS)

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
        self.best_anchor_finder = BestAnchorFinder(ANCHORS)
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

    def loss(self, y_pred, true_boxes):
        """
        return YOLOv2 loss
        input:
            - Y_pred: YOLOv2 predicted feature map, the output feature map of forward() function
              shape of [N, B*(5+C), Grid, Grid], t_x, t_y, t_w, t_h, t_c, and (class1_score, class2_score, ...)
            - true_boxes: all ground truth boxes, maximum 50 objects
        output:
            YOLOv2 loss includes coordinate loss, confidence score loss, and class loss.
        """

        shift_x, shift_y = torch.meshgrid(torch.arange(0, GRID_W), torch.arange(0, GRID_H))
        c_x = shift_x.t().contiguous.float()
        c_y = shift_y.t().contiguous.float()
        grid_xy = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], -1)  # [S*S, 2]
        grid_xy = grid_xy.repead(BOX, 1)                             # [S*S*B, 2]

        y_true      = self.build_target(true_boxes) #  shape of [N, S*S*B, 5 + n_class]
        coord_mask  = y_true.new_zero([BATCH_SIZE, GRID_H * GRID_W * B, 1])
        conf_mask   = y_true.new_zero([BATCH_SIZE, GRID_H * GRID_W * B, 1])
        class_mask  = y_true.new_zero([BATCH_SIZE, GRID_H * GRID_W * B, 1])

        '''
        Adjust prediction
        '''

        # adjust output to the shape of [N, S*S*B, 5 + n_class]
        y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(BATCH_SIZE, GRID_H * GRID_W * BOX, 5 + CLASS)

        # adjust xy, wh
        pred_box_xy = F.sigmoid(y_pred[..., 0:2]) + grid_xy         # [N, S*S*B, 2] + [S*S*B, 2] 
        pred_box_wh = torch.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, B, 2])
        pred_box_xywh = torch.cat([pred_box_xy, pred_box_wh], -1)

        # adjust confidence score
        pred_box_conf = F.sigmoid(y_pred[..., 4])

        # adjust class propabilities
        pred_box_class = y_pred[...,5:]

        '''
        Adjust ground truth
        '''
        # adjust true xy and wh
        true_box_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]

        # adjust true confidence score
        iou_scores = bbox_ious(pred_box_xywh, y_true[:4])

        true_box_conf = iou_scores * y_true[..., 4]
        
        # adjust class probabilities
        true_box_class = torch.argmax(y_true[...,5:], -1)
        
        '''
        Determine the mask
        '''
        ### coordinate mask, simply is all predictors
        coord_mask = y_true[..., 4] * LAMBDA_COORD          # [N, S*S*B, 1]

        ### confidence mask: penalize predictors and boxes with low IoU
        # first, penalize boxes, which has IoU with any ground truth box < 0.6
        iou_scores = bbox_ious(pred_box_xywh.unsqueeze(2), true_boxes[..., :4]) #[N, S*S*B, , 1]       

        best_iou = torch.max(iou_scores, dim=2, keepdim=False)     #[N, S*S*B, 1]
        conf_mask = conf_mask + (best_ious < 0.6).float() * (1 - y_true[..., 4]) * LAMBDA_NO_OBJECT

        # second, penalized predictors
        conf_mask = conf_mask + y_true[..., 4] * LAMBDA_OBJECT

        ### class mask: simply the positions containing true boxes
        class_mask = y_true[..., 4] * LAMBDA_CLASS

        '''
        Finalize the loss
        '''
        nb_coord_box = (coord_mask > 0.0).float().sum()
        nb_conf_box = (conf_mask > 0.0).float().sum()
        nb_class_box = (class_mask > 0.0).float().sum()

        # loss_xywh
        loss_xy = F.mse_loss(pred_box_xy * coord_mask, true_box_xy * coord_mask, reduction='sum') / (nb_conf_box + 1e-6)
        loss_wh = F.mse_loss(pred_box_wh * coord_mask, true_box_wh * coord_mask, reduction='sum')/ (nb_coord_box + 1e-6)

        # loss_class
        class_mask = class_mask.view(-1, 1)
        t_pred_box_class = pred_box_class.view(-1, CLASS).masked_select(class_mask).view(-1, CLASS)
        t_true_box_idx = true_box_class.view(-1, 1).masked_select(class_mask).view(-1, 1)
        loss_class = F.nll_loss(t_pred_box_class, t_true_box_idx, reduction='sum') / (nb_class_box + 1e-6)

        # loss_confidence
        loss_conf = F.mse_loss(pred_box_conf * conf_mask, true_box_conf * conf_mask, reduction='sum') / (nb_conf_box + 1e-6)

        loss = loss_xy + loss_wh + loss_class + loss_conf

        '''
        Debug
        '''
        print("Loss xy: ", loss_xy)
        print("Loss wh: ", loss_wh)
        print("loss class: ", loss_class)
        print("loss conf: ", loss_conf)
        print("total loss: ", loss)

        return loss

    def build_target(self, ground_truth):
        """
        Build target output y_true with shape of [N, S*S*B, 5+1]
        """
        y_true = np.zeros([BATCH_SIZE, GRID_W, GRID_H, B,  4 + 1 + 1])

        for iframe in range(BATCH_SIZE):
            for obj in ground_truth[iframe]:
                if obj[2] == 0 and obj[3] == 0: 
                    # both w and h are zero
                    break
                center_x, center_y, w, h, class_index = obj

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                assert grid_x < GRID_W and grid_y < GRID_H and class_index < CLASS

                box = [center_x, center_y, w, h]
                best_anchor, best_iou = self.best_anchor_finder.find(w, h)

                y_true[iframe, grid_x, grid_y, best_anchor, :4] = box
                y_true[iframe, grid_x, grid_y, best_anchor, 4] = 1.
                y_true[iframe, grid_x, grid_y, best_anchor, 5] = class_index

        y_true = y_true.reshape([BATCH_SIZE, -1, 6])

        return torch.from_numpy(y_true)

    def load_weight(self, weight_file):
        pass

if __name__ == "__main__":
    args_ = args.arg_parse()
    net = YOLOv2(args_)
    # net.load_weight("./darknet19_448.conv.23")
    print(net.parameters)


    input = torch.randn(4, 3, IMAGE_H, IMAGE_W)
    output = net.forward(input)
    # print("Input shape: ", input.shape)
    print("Output shape: ", output.shape)