import argparse

def arg_parse():
    """
    Parse argument for YOLOv2 detector
    """
    parser = argparse.ArgumentParser(description="A simplified YOLOv2 detector with Darknet19 backbone!")
    parser.add_argument("--output", dest="output", help="Path to output folder of detection result", default="./output")
    parser.add_argument("--model_dir", dest="model_dir", help="Path to the directory for storing trained models", default="./models")

    parser.add_argument("--weights", dest="weights", help="Path to the pretrained model",)
    parser.add_argument("--darknet19_weights", dest="darknet19_weights", help="Path to the pretrained darknet19-weights file", default="./darknet19_448.conv.23")
    ### training hyper parameters
    parser.add_argument("--epoch", dest="epoch", help="Learning epoch", default=10)
    parser.add_argument("--batch_size", dest="batch_size", help="The training batch size", default=16)
    parser.add_argument("--lr", dest="lr", help="Learning rate", default=1e-6)
    parser.add_argument("--momentum", dest="momentum", help="Learning momentum", default=0.9)
    parser.add_argument("--decay", dest="decay", help="Learning weights decay", default=5e-4)
    
    return parser
