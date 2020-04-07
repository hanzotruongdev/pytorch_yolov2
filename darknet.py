
#--------------------------------------------------
# Darknet19
# Written by Noi Truong email: noitq.hust@gmail.com
#--------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers

class GlobalAvgPool2d(nn.Module):
    '''
    Module GlobalAvgPool2d
    note: it is used in YOLO, just define for full defined Darknet19
    '''
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class Darknet19(nn.Module):

    cfg = {
        'layer0': [32],
        'layer1': ['M', 64],
        'layer2': ['M', 128, 64, 128],
        'layer3': ['M', 256, 128, 256],
        'layer4': ['M', 512, 256, 512, 256, 512],
        'layer5': ['M', 1024, 512, 1024, 512, 1024]
    }

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.in_channels = 3
        self.module_list = []

        self.layer0 = self.make_layers(self.cfg['layer0'])
        self.layer1 = self.make_layers(self.cfg['layer1'])
        self.layer2 = self.make_layers(self.cfg['layer2'])
        self.layer3 = self.make_layers(self.cfg['layer3'])
        self.layer4 = self.make_layers(self.cfg['layer4'])
        self.layer5 = self.make_layers(self.cfg['layer5'])

        self.conv = nn.Conv2d(self.in_channels, num_classes, kernel_size=1, stride=1)
        self.avgpool = GlobalAvgPool2d()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv(x)
        x = self.avgpool(x)
        x = self.softmax(x)

        return x

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
        print("Loading darknet19 weights from file %s" % weight_file)
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
        
    
    def make_layers(self, layer_cfg):
        '''
        Create Block of layers base on layer config
        Block may contains Maxpool, Convlution with BN and LeakyReLU
        Return an array of them
        '''

        layers = []

        # set the kernel size of the first conv block = 3
        kernel_size = 3
        for v in layer_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                ls = conv_bn_leaky(self.in_channels, v, kernel_size)
                self.module_list  += ls[:-1]
                layers += ls

                kernel_size = 1 if kernel_size == 3 else 3
                self.in_channels = v
        return nn.Sequential(*layers)


if __name__ == "__main__":
    im = np.random.randn(1, 3, 224, 224)
    im_variable = Variable(torch.from_numpy(im)).float()
    model = Darknet19()
    model.load_weight("./darknet19_448.conv.23")
    out = model(im_variable)
    print(out.size())
    print(model)