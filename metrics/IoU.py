import numpy as np

def calculate_iou(gt, pr, form='pascal_voc'):
    # """Calculates the Intersection over Union.

    # Args:
    #     gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
    #     pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
    #     form: (str) gt/pred coordinates format
    #         - pascal_voc: [xmin, ymin, xmax, ymax]
    #         - coco: [xmin, ymin, w, h]
    # Returns:
    #     (float) Intersection over union (0.0 <= iou <= 1.0)
    # """
    
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = - gt[0] + gt[2]
        gt[3] = - gt[1] + gt[3]
        pr[2] = - pr[0] + pr[2]
        pr[3] = - pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    # print(">>>>> dx: ", dx)

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    # print(">>>>> dy: ", dy)
    
    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
        (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
        (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
        overlap_area
    )

    return overlap_area / union_area



def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None):
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1
    # print(f">>>>> gts is {gts}")

    for gt_idx in range(len(gts)):
        # print(f">>>> gt_idx is {gt_idx}")
        if type(gts[gt_idx])==int and gts[gt_idx] == -1:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return int(best_match_idx)


def calculate_precision(gts, preds, threshold=0.5, form='pascal_voc', ious=0):
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    # print(f">>>>> gts is {gts}")

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)
        # 
        # print(f">>>>> best_match_gt_idx is {best_match_gt_idx}")

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1
            print(f"best pred_idx is {pred_idx}")

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # print(f">>>>> gts is {gts}")
    fn = (gts.sum(axis=1) > 0).sum()
    # fn = sum(sum([gt > 0 for gt in gts]))

    return tp / (tp + fp + fn)



def calculate_image_precision(gts, preds, threshold=0.5, form='pascal_voc'):
    """Calculates image precision.
       The mean average precision at different intersection over union (IoU) thresholds.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    # n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
        
    image_precision = calculate_precision(gts.copy(), preds, threshold=threshold, form=form, ious=ious)

    return image_precision

# from collections import namedtuple
# import numpy as np
# import cv2
# # define the `Detection` object
# Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

# def bb_intersection_over_union(img_path, boxA, boxB):
#     	# determine the (x, y)-coordinates of the intersection rectangle
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[2], boxB[2])
# 	yB = min(boxA[3], boxB[3])
# 	# compute the area of intersection rectangle
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
# 	# compute the area of both the prediction and ground-truth
# 	# rectangles
# 	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
# 	# compute the intersection over union by taking the intersection
# 	# area and dividing it by the sum of prediction + ground-truth
# 	# areas - the interesection area
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
# 	# return the intersection over union value
# 	return iou
