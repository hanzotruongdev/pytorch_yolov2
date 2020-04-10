import numpy as np
import torch
import cv2
from torchvision.ops import nms

# VOC 2012 dataset
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']

CLASS = len(LABELS)

IMAGE_W, IMAGE_H    = 416, 416
GRID_W, GRID_H      = 13, 13


ANCHORS             = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
BOX                 = int(len(ANCHORS) / 2)


def draw_boxes(image, boxes, label=LABELS, color=255):
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

        print("object: ", obj)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, LABELS[int(cls_idx)], (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2, color)


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


def get_detection_result(y_pred, conf_thres=0.6, nms_thres=0.4, classes=len(LABELS)):
    """
    input:
        - y_pred: [N, B*(5+CLASS), S, S]
        - conf_thres: confidence threshold
        - nms_thres: NMS threshold
        - classes: number of class
    return:
        - out: [N, 50, 5+1]
    """

    print("*"*50)
    print("Get detection result: ", y_pred.shape)
    print("*"*50)
    

    batch_size = y_pred.shape[0]
    output = y_pred.new_zeros([batch_size, 50, 5+1])

    # prepare grid, and empty mask
    shift_x, shift_y = torch.meshgrid(torch.arange(0, GRID_W), torch.arange(0, GRID_H))
    c_x         = shift_x.t().contiguous().float()
    c_y         = shift_y.t().contiguous().float()
    grid_xy     = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], -1)  # [S*S, 2]
    grid_xy     = grid_xy.repeat(BOX, 1)                             # [S*S*B, 2]

    '''
    Adjust prediction
    '''
    # transform from shape of [N, B*(5+CLASS), S, S] to [N, S*S*B, 5 + CLASS]
    y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, GRID_H * GRID_W * BOX, 5 + CLASS)

    # adjust xy, wh
    pred_box_xy = y_pred[..., 0:2].sigmoid() + grid_xy         # [N, S*S*B, 2] + [S*S*B, 2] 


    print("pred_box_xy col 5 syle dif: ", pred_box_xy[0].view(BOX, GRID_W, GRID_H, 2)[:, 5, :, :2])
    print("pred_box_xy col 5 syle dif: ", pred_box_xy[0].view(BOX, GRID_W, GRID_H, 2)[:, :, 6, :2])
    pred_box_wh = y_pred[..., 2:4].exp().view(-1, BOX, 2) * torch.Tensor(ANCHORS).view(1, BOX, 2)
    pred_box_wh = pred_box_wh.view(-1, GRID_H * GRID_W * BOX, 2)
    pred_box_xywh = torch.cat([pred_box_xy, pred_box_wh], -1)

    y_pred[..., :4] = pred_box_xywh

    # adjust confidence score
    y_pred[..., 4].sigmoid_()

    # adjust class prob
    y_pred[..., 5:] = torch.nn.Softmax(dim=-1)(y_pred[..., 5:])

    # rescale x, y
    y_pred[:,:4] = y_pred[:, :4] * 416 / 13

    for iframe in range(batch_size):
        total_keep = 0
        i_frame_pred = y_pred[iframe]     # [S*S*B, 5+1]

        ### first, remove boxes with confidence score < conf_thres
        i_frame_pred = i_frame_pred[i_frame_pred[:, 4] >= conf_thres]

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

            print("keep: ", keep)

            keep_boxes = cls_i_pred[keep]
            n_keep = keep_boxes.shape[0]

            if n_keep:
                # assert total_keep + n_keep < 50
                if total_keep + n_keep >=50:
                    break
                output[iframe, total_keep:total_keep+n_keep, :] = keep_boxes

    return output

def get_detection_result_b(feats, threshold, iou_thres, n_classes):
    '''
    get true detection from final feature map
    - feat shape: [N, grid * grid * nAnchor, 5 + nClass]
    - output: the final detection result tensor of shape [#detection, attributes: 
    (image_id_in_batch, x1, y1, x2, y2, prob, class_id, class_prob)]
    '''
    batch_size = feats.shape[0]
    result = feats.new_zeros([batch_size, 50, 6])



    # Assign zero to boxes which have objective probability less than threshold
    mask = (feats[:,:,4] >= threshold).float().unsqueeze(2)
    feats = feats * mask

    result_exits = False
    batch_size = feats.size(0)

    # output = torch.randn(1, 1)
    for index in range(batch_size):
        feat = feats[index]
        # remove object with objectnesss = 0
        mask = torch.nonzero(feat[:, 4]).squeeze()
        try:
            feat = feat[mask]
        except:
            print("has exception in first step!")
            continue

        # convert result format from (x, y, w, h, pro, class[0], class[1])
        # to format (x1, y1, x2, y2, pro, max_classid, score)
        x1 = (feat[:, 0] - feat[:, 2]/2).unsqueeze(1)
        x2 = (feat[:, 0] + feat[:, 2]/2).unsqueeze(1)
        y1 = (feat[:, 1] - feat[:, 3]/2).unsqueeze(1)
        y2 = (feat[:, 1] + feat[:, 3]/2).unsqueeze(1)

        max_class_prob, class_id = torch.max(feat[:, 5:5+n_classes], 1)
        max_class_prob = max_class_prob.float().unsqueeze(1)
        class_id = class_id.float().unsqueeze(1)

        # cat all the features together
        new_feat = torch.cat((x1, y1, x2, y2, feat[:, 4].unsqueeze(1), class_id, max_class_prob), 1)

        # conducting nms
        classes_in_prediction = torch.unique(new_feat[:, 5]) # get list of unique classes in the class_id column

        for cls_id in classes_in_prediction:
            # perform NMS for each class
            # 1. get the detections corresponding to cls_id
            mask = (new_feat[:, 5] == cls_id).float()
            idx = torch.nonzero(mask).squeeze()
            predictions_of_cls = new_feat[idx].view(-1, 7)

            # 2. sort the predictions_of_cls in the descending order.
            sorted_idx = torch.sort(predictions_of_cls[:, 4])[1]       # sort by object_prob column
            predictions_of_cls = predictions_of_cls[sorted_idx]
            n_predictions_of_cls = predictions_of_cls.size(0)

            # 3. remove the boxes with the iou > threshold
            for i in range(n_predictions_of_cls):
                try:
                    ious = box_ious(predictions_of_cls[i].unsqueeze(0), predictions_of_cls[i+1:])
                except:
                    break
                
                iou_mask = (ious < iou_thres).float().unsqueeze(1)      # ious > thres will be masked as zero
                predictions_of_cls[i+1:] *= iou_mask

                # remove the zero entries
                non_zero_index = torch.nonzero(predictions_of_cls[:, 4]).squeeze()
                predictions_of_cls = predictions_of_cls[non_zero_index].view(-1, 7)

            # 4. assign batch index into the result 
            batch_index = predictions_of_cls.new(predictions_of_cls.size(0), 1).fill_(index)

            if not result_exits:
                output = torch.cat((batch_index, predictions_of_cls), 1)
                result_exits = True
            else:
                temp = torch.cat((batch_index, predictions_of_cls), 1)
                output = torch.cat((output, temp), 0)

    return output


if __name__ == "__main__":
    pass