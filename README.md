<h1 algin="center"> PYTORCH_YOLOV2 </h1> 

Implement a minimal YOLOv2 in Pytorch for learning purpose. 
The project forcus on implement the logic code behinds the YOLOv2 algorithms. 
Therefore, I will skip optional features as much as posible such as no loading config, uses only Darknet19 as a backbone network.

## Features

- [x] Implement forward pass darknet 19
- [x] Implement YOLO header layers
- [x] Implement region loss
- [x] load pre-trained darknet19 weights
- [ ] load passcal voc dataset (generate input, label)
- [ ] Define the region loss
- [ ] Implement train on VOC dataset
- [ ] Save and load trained model
- [ ] Implement inference