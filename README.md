<h1 align="center"> PYTORCH_YOLOV2 </h1> 

Implement a minimal YOLOv2 in Pytorch for learning purpose. 
The project focus on implementations of the logic code behind the YOLOv2 algorithms. 
Therefore, I would like to skip optional features as much as posible such as no loading config, uses only Darknet19 as a backbone network.

## Features

- [x] Implement Darknet19
- [x] Implement YOLO CNN using Darknet
- [x] Implement Reog layer
- [x] Implement region loss
- [x] load pre-trained darknet19 weights
- [x] load passcal voc 2012 dataset (generate input, label)
- [x] Implement train YOLO
- [ ] Save and load trained model
- [ ] Implement inference
- [x] Show loss graph using TensorBoard
- [x] Implement jupyter notebook for training on Google Colab
