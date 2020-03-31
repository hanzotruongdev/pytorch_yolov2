import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.reorg import Reorg
import numpy as np
import args


class YOLOv2(nn.Module):
    def __init__(self, args):
        super(YOLOv2, self).__init__()

        self.NUM_CLASS = int(args.NUM_CLASS)
        self.ANCHORS = [[1, 3], [3, 4], [5, 6]]       # anchor box size relative to the final feature map
        self.NUM_ANCHOR = len(self.ANCHORS)
        self.INPUT_SIZE = 320
        self.GRID_SIZE = 10

        # define elements used
        self.max_pool = nn.MaxPool2d(2, 2)
        # first
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # second
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # third
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # fourth
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        # fiveth
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # sixth
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        # seventh
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        # eighth
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        # nineth
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        # tenth
        self.conv10 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        # 11th
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        # 12th
        self.conv12 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        # 13th
        self.conv13 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        # 14th
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn14 = nn.BatchNorm2d(1024)
        # 15th 
        self.conv15 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        # 16th
        self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn16 = nn.BatchNorm2d(1024)
        # 17th
        self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.bn17 = nn.BatchNorm2d(512)
        # 18th
        self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn18 = nn.BatchNorm2d(1024)
        # 19th
        self.conv19 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bn19 = nn.BatchNorm2d(1024)
        # 20th
        self.conv20 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bn20 = nn.BatchNorm2d(1024)
        # 21th
        self.conv21 = nn.Conv2d(512, 64, 1, 1, 0, bias=False)
        self.bn21 = nn.BatchNorm2d(64)
        # reorg 
        self.reorg = Reorg(2)
        # 22th
        self.conv22 = nn.Conv2d(1280, 1024, 3, 1, 1, bias=False)
        self.bn22 = nn.BatchNorm2d(1024)
        # 23th
        self.conv23 = nn.Conv2d(1024, self.NUM_ANCHOR * (5 + self.NUM_CLASS), 1, 1, 0)
        self.activation23 = nn.Linear(self.NUM_ANCHOR * (5 + self.NUM_CLASS), self.NUM_ANCHOR * (5 + self.NUM_CLASS))

        # save list module as a array for weight restoration
        self.module_list = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, self.conv5, self.bn5, self.conv6, self.bn6, self.conv7, self.bn7, self.conv8, self.bn8, self.conv9, self.bn9, self.conv10, self.bn10, self.conv11, self.bn11, self.conv12, self.bn12, self.conv13, self.bn13, self.conv14, self.bn14, self.conv15, self.bn15, self.conv16, self.bn16, self.conv17, self.bn17, self.conv18, self.bn18, self.conv19, self.bn19, self.conv20, self.bn20, self.conv21, self.bn21, self.conv22, self.bn22, self.conv23]

        return

    def transform_detection(self, pred, input_size):
        '''
        input shape [N, As * (5 + nclass), grid, grid]
        output shape [N, grid*grid*As, 5 + nclass]
        '''
        
        batch_size = pred.shape[0]
        grid_size = pred.shape[-1]

        # convert the feature map shape
        pred = pred.view( batch_size, self.NUM_ANCHOR * (5 + self.NUM_CLASS), grid_size * grid_size)
        pred = pred.transpose(1, 2).contiguous()
        pred = pred.view(batch_size, grid_size * grid_size * self.NUM_ANCHOR, -1)
        
        # calculate x, y, and obj_prob
        pred[:,:,0] = torch.sigmoid(pred[:,:,0])
        pred[:,:,1] = torch.sigmoid(pred[:,:,1])
        pred[:,:,4] = torch.sigmoid(pred[:,:,4])
        
        t = np.arange(grid_size)            # make the matrix of offset
        a, b = np.meshgrid(t, t)
        offset_x = torch.from_numpy(a).view(-1, 1)
        offset_y = torch.from_numpy(b).view(-1, 1)
        offset_xy = torch.cat((offset_x, offset_y), 1).repeat(1, self.NUM_ANCHOR).view(-1, 2).unsqueeze(0)
        
        pred[:,:,:2] += offset_xy           # then add it to the feature map

        # calculate the h, w
        t = torch.Tensor(self.ANCHORS)
        ANCHORS = t.repeat(grid_size * grid_size, 1).unsqueeze(0)   # make the matrix of anchor
        pred[:, :, 2:4] = torch.exp(pred[:,:,2:4]) * ANCHORS        # then use it to calcute the w and h

        # scale x, y, w, h to the size of the input
        pred[:,:,:4] *= input_size//grid_size

        # calculate the class prob
        pred[:, :, 5: 5+self.NUM_CLASS] = F.softmax(pred[:, :, 5: 5+self.NUM_CLASS], -1)

        return pred

    def forward(self, x):
        
        # First 18 layers of Darknet19
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)
        
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.max_pool(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv8(out)
        out = self.bn8(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.max_pool(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv10(out)
        out = self.bn10(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv11(out)
        out = self.bn11(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv12(out)
        out = self.bn12(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv13(out)
        out = self.bn13(out)
        y = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.max_pool(y)

        out = self.conv14(out)
        out = self.bn14(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv15(out)
        out = self.bn15(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv16(out)
        out = self.bn16(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv17(out)
        out = self.bn17(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv18(out)
        out = self.bn18(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)
        
        # YOLOv2 header
        out = self.conv19(out)
        out = self.bn19(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv20(out)
        out = self.bn20(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        z = self.conv21(y)
        z = self.bn21(z)
        z = nn.LeakyReLU(0.1, inplace = True)(z)

        reorg = self.reorg.forward(z)
        out = torch.cat((reorg, out), 1)

        out = self.conv22(out)
        out = self.bn22(out)
        out = nn.LeakyReLU(0.1, inplace = True)(out)

        out = self.conv23(out)

        # transform the detection feature map
        out = self.transform_detection(out, x.shape[-1])

        return out

    def loss(self, pred, true):
        """
        return YOLOv2 loss
        input:
            - pred: YOLOv2 prediction, the output feature map of forward() function
                    shape of [N, S*S*B, 5 + n_class]
            - label: ground truth in shape of [N, S*S*B, 5 + n_class]
        output:
            YOLOv2 loss includes coordinate loss, confidence score loss, and class loss.
        """

        # hyper parameters for loss function
        LAMBDA_OBJECT       = 0.5
        LAMBDA_NO_OBJECT    = 1.0
        LAMBDA_COORD        = 1.0
        LAMBDA_CLASS        = 1.0

        batch_size = pred.shape[0]

        pred_xy, pred_wh, pred_conf = pred[:,:,:2], pred[:,:,2:4], pred[4]
        pred_class_prob = pred[:,:,5:]

        true_xy, true_wh, true_conf = true[:,:,:2], true[:,:,2:4], true[4]
        true_class_id = true[:,:,5].squeeze()

        # loss_xywh
        mask_xy = true_conf.unsqueeze(-1)
        n_obj = true_conf.sum()
        loss_xy = (((true_xy - pred_xy)**2) * mask_xy).sum() / (n_obj + 1e-6)
        loss_wh = (((true_wh - pred_wh)**2) * mask_xy).sum() / (n_obj + 1e-6)
        loss_xywh = loss_xy + loss_wh

        # loss_class
        # loss_class = F.nll_loss(pred_class_prob, true_class_id)
        # If i calculate this way, then it includes the noobj to loss....
        pred_class_prob = pred_class_prob.view([-1, self.NUM_CLASS])
        true_class_id = true_class_id.view([-1])
        mask = true_conf.view([-1])

        have_object_pred_class_prob = pred_class_prob[mask > 0]
        have_object_true_class_id = true_class_id[mask>0]

        loss_class = F.nll_loss(have_object_pred_class_prob, have_object_true_class_id)

        # loss_confidence





    def load_weight(self, weight_file):
        """
        Load weight from pretrained model from Pjreddie website
        Weight file format: 
            - header: 4 inter32
            - remain: weight, with the order of weight file complies the following rule:
                if [conv+bn]: bias of BN, weight of BN, running mean of BN, running var of BN, weight of conv
                else only [conv]: bias of conv, weight of conv.
        """

        print("-" * 30)
        print("Loading weight from file %s" % weight_file)
        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, np.int32, count=4)

        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0

        # copy weight to the first 22 layers of [conv+bn]
        # if darknet19.23.weigh is provided, we just the weight of the first 18 layers
        for i in range(22):
            if start >= buf.size:
                break
            conv = self.module_list[2*i]
            bn = self.module_list[2*i + 1]

            print("[%4d]" % i, conv)
            print("[%4d]" % i, bn)
            
            n_weight = conv.weight.numel()
            n_bias = bn.bias.numel()
            
            bn.bias.data.copy_(torch.from_numpy(buf[start:start+n_bias]));                                      start += n_bias
            bn.weight.data.copy_(torch.from_numpy(buf[start:start+n_bias]));                                    start += n_bias
            bn.running_mean.copy_(torch.from_numpy(buf[start:start+n_bias]));                                   start += n_bias
            bn.running_var.copy_(torch.from_numpy(buf[start:start+n_bias]));                                    start += n_bias
            conv.weight.data.copy_(torch.from_numpy(buf[start:start+n_weight]).view_as(conv.weight.data));      start += n_weight
        
        # copy weight to the last layer of conv
        for i in range(22,23):
            if start >= buf.size:
                break

            conv = self.module_list[2*i]
            

            n_weight = conv.weight.numel()
            n_bias = conv.bias.numel()

            print("[%4d]" % i, (conv))

            conv.bias.data.copy_(torch.from_numpy(buf[start:start+n_bias]));        start += n_bias
            conv.weight.data.copy_(torch.from_numpy(buf[start:start+n_weight]).view_as(conv.weight.data));      start += n_weight

        print("Successfully loaded weight file!", (buf.size, start))
        print("-" * 30)

if __name__ == "__main__":
    args_ = args.arg_parse()
    net = YOLOv2(args_)
    net.load_weight("./darknet19_448.conv.23")


    input = torch.randn(3, 3, 320, 320)
    output = net.forward(input)
    print("Input shape: ", input.shape)
    print("Output shape: ", output.shape)