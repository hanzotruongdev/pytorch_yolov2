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
    
def box_ious(box, boxes):
    """
    returns the IoU of box vs a number of other boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # calculate the corner position of the intersetion
    inter_rec_x1 = torch.max(b1_x1, b2_x1)
    inter_rec_y1 = torch.max(b1_y1, b2_y1)
    inter_rec_x2 = torch.min(b1_y1, b2_y1)
    inter_rec_y2 = torch.min(b1_y2, b2_y2)

    # calculate the area
    inter_area = torch.clamp(inter_rec_x2 - inter_rec_x1 + 1, 0) * torch.clamp(inter_rec_y2 - inter_rec_y1 + 1, min=0)

    box_area = (b1_x2 - b1_x1 +1) * (b1_y2 - b1_y1 + 1)
    boxes_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    ious = inter_area / (box_area + boxes_area - inter_area)

    return ious


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