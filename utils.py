import numpy as np
import torch
import cv2


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
        resized_xmax = lb[2] * width_ratio
        resized_ymax = lb[3] * height_ratio
        resize_width = resized_xmax - resized_xmin
        resize_height = resized_ymax - resized_ymin
        new_label.append([resized_xmin, resized_ymin, resize_width, resize_height, lb[4]])

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

    # calculate min xy of intersects, areas of intersect
    intersect_mins = torch.max(box1_mins, box2_mins)
    intersect_maxs = torch.min(box1_maxs, box2_maxs)
    intersect_wh = intersect_maxs - intersect_mins
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
    
    def __init__(self, anchors):
        """
        ANCHORS: a np.array of even number length e.g.
            
            _ANCHORS = [[4,2], ##  width=4, height=2,  flat large anchor box
                        [2,4], ##  width=2, height=4,  tall large anchor box
                        [1,1]] ##  width=1, height=1,  small anchor box
        """
        self.anchors = [BoundBox(ANCHORS[i][0]/2,  ANCHORS[i][1]/2, ANCHORS[i][0], ANCHORS[i][1]) 
                        for i in range(int(len(ANCHORS))]

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

def get_detection_result(feats, threshold, iou_thres, n_classes):
    '''
    get true detection from final feature map
    - feat shape: [N, grid * grid * nAnchor, 5 + nClass]
    - output: the final detection result tensor of shape [#detection, attributes: 
    (image_id_in_batch, x1, y1, x2, y2, prob, class_id, class_prob)]
    '''

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