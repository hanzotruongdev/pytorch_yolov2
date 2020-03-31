import argparse

def arg_parse():
    """
    Parse argument for YOLOv2 detector
    """
    parser = argparse.ArgumentParser(description="A YOLOv2 detector with Darknet19 backbone!")
    parser.add_argument("--images", dest='images', help="Path to input image folder", default="./test")
    parser.add_argument("--output", dest='output', help="Path to output folder of detection result", default="./output")
    parser.add_argument("--bs", dest='batchsize', help="Batch size", default=8)
    parser.add_argument("--cf_thres", dest="conf_thres", help="Confidence threshold", default=0.5)
    parser.add_argument("--nms_thres", dest="nms_thres", help="NMS threshold", default=0.4)
    parser.add_argument("--weight", dest='weight_file', help="Path to the pretrained Darknet19 weight file", default="./darknet19_448.conv.23")
    parser.add_argument("--model", dest="pretrained_model", help="Path to the pretrained model")
    parser.add_argument("--num_class", dest="num_class", help="Number of classes", default=80)

    return parser.parse_args()
