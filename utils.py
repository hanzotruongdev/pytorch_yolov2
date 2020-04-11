import numpy as np
import torch
import cv2
from torchvision.ops import nms

def draw_boxes(image, boxes, label, color=255):
    """
    draw boxes on image
    - image: np array or cv2 image
    - boxes: shape [50, 5 + 1]
    - label: array of label e.g. ["car", "dog", etc]
    """

    if boxes.shape[1] == 6:
        boxes = torch.cat([boxes[:,:4], boxes[:,5].unsqueeze(-1)], -1)
    for obj in boxes:
        if obj[2] == 0 and obj[3] == 0:
            break

        x, y, w, h, cls_idx = obj
        xmin, xmax = int(x - w/2), int(x + w/2)
        ymin, ymax = int(y - h/2), int(y + h/2)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label[int(cls_idx)], (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2, color)


    return image

def resize_image(image, label, image_size):
    """
    resize object detection image
    """
    height, width = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))
    width_ratio = float(image_size) / width
    height_ratio = float(image_size) / height
    new_label = []
    for lb in label:
        resized_xmin = lb[0] * width_ratio
        resized_ymin = lb[1] * height_ratio
        width = lb[2] * width_ratio
        height = lb[3] * height_ratio
        new_label.append([resized_xmin, resized_ymin, width, height, lb[4]])

    return image, new_label
    
def bbox_ious(boxes1, boxes2):
    """
    calculate iou scores
        - boxes1 [..., 4], (x, y, w, h)
        - boxes2 [..., 4], (x, y, w, h)
    """
    # extract xy, wh of box1 and box2
    box1_xy = boxes1[..., :2]
    box1_wh = boxes1[..., 2:4]
    box2_xy = boxes2[..., :2]
    box2_wh = boxes2[..., 2:4]

    # calculate min xy max xy of box1 and box2
    box1_wh_half = box1_wh / 2.0
    box1_mins = box1_xy - box1_wh_half
    box1_maxs = box1_xy + box1_wh_half
    box2_wh_half = box2_wh / 2.0
    box2_mins = box2_xy - box2_wh_half
    box2_maxs = box2_xy + box2_wh_half

    box1_mins.clamp_(0)
    box1_maxs.clamp_(0)
    box2_mins.clamp_(0)
    box2_maxs.clamp_(0)

    # calculate min xy of intersects, areas of intersect
    intersect_mins = torch.max(box1_mins, box2_mins)
    intersect_maxs = torch.min(box1_maxs, box2_maxs)
    intersect_wh = intersect_maxs - intersect_mins
    intersect_wh.clamp_(0)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # calculate areas of 2 boxes
    box1_areas = box1_wh[..., 0] * box1_wh[..., 1]
    box2_areas = box2_wh[..., 0] * box2_wh[..., 1]

    # calculate iou_scores
    union_areas = box1_areas + box2_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores

class BoundBox:
    def __init__(self, x, y, w, h, confidence=None,classes=None):
        self.xmin, self.ymin = x-w/2, y-h/2
        self.xmax, self.ymax = x+w/2, y+h/2
        ## the code below are used during inference
        # probability
        self.confidence      = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])

class BestAnchorFinder(object):
    
    def __init__(self, ANCHORS):
        '''
        ANCHORS: a np.array of even number length e.g.
        
        _ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                    2,4, ##  width=2, height=4,  tall large anchor box
                    1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]

    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union
    
    def find(self, center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0,center_w, center_h)
        ##  For given object, find the best anchor box!
        for i in range(len(self.anchors)): ## run through each anchor box
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return(best_anchor,max_iou)    


def get_detection_result(y_pred, anchors, classes, conf_thres=0.6, nms_thres=0.4):
    """
    input:
        - y_pred: [N, B*(5+CLASS), S, S]
        - conf_thres: confidence threshold
        - nms_thres: NMS threshold
        - classes: number of class
    return:
        - out: [N, 50, 5+1]
    """

    # config
    BATCH_SIZE, _, GRID_W, GRID_H = y_pred.shape
    ANCHORS = anchors
    BOX = len(anchors) // 2
    CLASS = classes
    
    output = y_pred.new_zeros([BATCH_SIZE, 50, 5+1])

    # prepare grid
    lin_x = torch.arange(0, GRID_W).repeat(GRID_H, 1).view(GRID_W * GRID_H)
    lin_y = torch.arange(0, GRID_H).repeat(1, GRID_W).view(GRID_W * GRID_H)
    
    t_anchors   = torch.Tensor(ANCHORS).view(-1, 2) #[BOX, 2]
    anchor_w = t_anchors[:, 0]
    anchor_h = t_anchors[:, 1]

    '''
    Adjust prediction
    '''
    ### y_pred has shape of [N, B*(5+CLASS), S, S], we need it transfromated 
    ### to [N, W, H, B * (5 + CLASS)]
    y_pred = y_pred.permute(0, 2, 3, 1).contiguous()

    ### adjust x, y, w, h
    y_pred          = y_pred.view(BATCH_SIZE, GRID_H * GRID_W, BOX, 5 + CLASS)     #[N, W*H, B, (5 + CLASS)]
    pred_box_x      = y_pred[..., 0].sigmoid() + lin_x.view(-1, 1)       # [N, W*H, B] + [W*H, 1]      =>   #[N, W*H, B]
    pred_box_y      = y_pred[..., 1].sigmoid() + lin_y.view(-1, 1)       # [N, W*H, B] + [W*H, 1]      =>   #[N, W*H, B]
    pred_box_w      = y_pred[..., 2].exp() * anchor_w.view(-1)           # [N, W*H, B] * [B]           =>   #[N, W*H, B]
    pred_box_h      = y_pred[..., 3].exp() * anchor_h.view(-1)           # [N, W*H, B] * [B]           =>   #[N, W*H, B]

    y_pred          = y_pred.view(BATCH_SIZE, GRID_H * GRID_W * BOX, 5 + CLASS)
    pred_box_xywh   = torch.cat([pred_box_x.view(BATCH_SIZE, -1, 1), pred_box_y.view(BATCH_SIZE, -1, 1), \
        pred_box_w.view(BATCH_SIZE, -1, 1), pred_box_h.view(BATCH_SIZE, -1, 1)], -1)
       
    # adjust confidence score
    pred_box_conf = (y_pred[..., 4]).sigmoid()

    # adjust class propabilities: 
    # - at train time: we do not Softmax cuz we call nn.CrossEntropyLoss
    #   this loss function takes care call to nn.Softmax
    # - at test time, we adjust by calling Softmax.
    pred_box_class = torch.nn.Softmax(dim=-1)(y_pred[..., 5:])

    # rescale x, y
    pred_box_xywh.mul_(416 / 13)

    y_pred[..., :4] = pred_box_xywh
    y_pred[..., 5] = pred_box_conf
    y_pred[..., 5:] = pred_box_class

    for iframe in range(BATCH_SIZE):
        total_keep = 0
        i_frame_pred = y_pred[iframe]     # [S*S*B, 5+1]

        ### first, remove boxes with confidence score < conf_thres
        i_frame_pred = i_frame_pred[i_frame_pred[:, 4] >= conf_thres]
        if i_frame_pred.shape[0] == 0:
            # skip frame with no detection
            continue

        clses_idx = i_frame_pred[:, 5:].argmax(-1)

        i_frame_pred = torch.cat([i_frame_pred[:,:5], clses_idx.float().unsqueeze(-1)], -1)


        ### second, run NMS on each class
        for cls_idx in range(classes):
            cls_i_pred = i_frame_pred[i_frame_pred[:, 5] == cls_idx].view(-1, 6)

            if cls_i_pred.shape[0] == 0:
                # if there is no detection of this class, we skip
                continue
            
            x_mins = (cls_i_pred[:,0] - cls_i_pred[:,2]/2).unsqueeze(-1)
            x_maxs = (cls_i_pred[:,0] + cls_i_pred[:,2]/2).unsqueeze(-1)
            y_mins = (cls_i_pred[:,1] - cls_i_pred[:,3]/2).unsqueeze(-1)
            y_maxs = (cls_i_pred[:,1] + cls_i_pred[:,3]/2).unsqueeze(-1)

            b = torch.cat([x_mins, y_mins, x_maxs, y_maxs], -1)
            scores = cls_i_pred[:,4]
            keep = nms(b, scores, iou_threshold = nms_thres)

            keep_boxes = cls_i_pred[keep]
            n_keep = keep_boxes.shape[0]

            if n_keep:
                # assert total_keep + n_keep < 50
                if total_keep + n_keep >=50:
                    break
                output[iframe, total_keep:total_keep+n_keep, :] = keep_boxes

    return output

if __name__ == "__main__":
    pass