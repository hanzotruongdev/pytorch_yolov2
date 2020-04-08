#---------------------------------------------
# Pytorch YOLOv2 - A simplified yolov2 version
# @Author: Noi Truong <noitq.hust@gmail.com>
#---------------------------------------------

import torch
from model import YOLOv2
import utils
from args import arg_parse

args = arg_parse()


## test case 1 ##
def test_forward_pass():
    net = YOLOv2(args)
    net.load_weight("./darknet19_448.conv.23")
    
    input  = torch.randn(3, 3, 320, 320)
    output = net.forward(input)
    final_output = utils.get_detection_result(output, 0.5, 0.4, net.NUM_CLASS)

    print("args.num_class: ", args.num_class)


all_test = {
    'test_forward_pass': test_forward_pass,
    }

if __name__ == "__main__":
    for k in all_test:
        print('Runing test case: ', k)
        all_test[k]()


