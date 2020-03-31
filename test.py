import torch
from model import YOLOv2
import utils
from args import arg_parse

args = arg_parse()

if __name__ == "__main__":
    net = YOLOv2(args)
    net.load_weight("./darknet19_448.conv.23")
    
    input  = torch.randn(3, 3, 320, 320)
    output = net.forward(input)
    final_output = utils.get_detection_result(output, 0.5, 0.4, net.num_class)

    print("args.num_class: ", args.num_class)

    # print("output data: ", final_output)
    # print("output shape: ", final_output.shape)



