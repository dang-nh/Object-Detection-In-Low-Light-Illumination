from collections import namedtuple
import cv2
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


def find_best_match(gt, preds, gt_idx, threshold=0.5, form='pascal_voc', ious=None):
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

    for pr_idx in range(len(preds)):

        if type(preds[pr_idx]) == int and preds[pr_idx] == -1:
            # Already matched GT-box
            continue

        iou = calculate_iou(preds[pr_idx], gt, form=form)

        # if ious is not None:
        #     ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = pr_idx

    return int(best_match_idx), best_match_iou


def calculate_precision(gts, preds, threshold=0.5, form='pascal_voc'):
    n = len(gts)
    # tp = 0
    # fp = 0

    best_matched_idxs = {}
    best_iou_list = []
    # for pred_idx, pred in enumerate(preds_sorted):
    for gt_idx in range(n):

        best_match_pred_idx, best_iou = find_best_match(gts[gt_idx], preds, gt_idx,
                                                        threshold=threshold, form=form)
        if best_match_pred_idx != -1:
            best_matched_idxs[gt_idx]=best_match_pred_idx
            best_iou_list.append(best_iou)
        # if best_match_gt_idx >= 0:
        #     # True positive: The predicted box matches a gt box with an IoU above the threshold.
        #     tp += 1
        #     # Remove the matched GT box
        #     gts[best_match_gt_idx] = -1

        # else:
        #     # No match
        #     # False positive: indicates a predicted box had no associated gt box.
        #     fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    # fn = (gts.sum(axis=1) > 0).sum()

    return best_matched_idxs, best_iou


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
    # print(f">>>>> gts is {gts}")
    # print(f">>>>> preds is {preds}")
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    # for threshold in thresholds:
    # precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
    #                                              form=form, ious=ious)
    # image_precision += precision_at_threshold / n_threshold

    image_precision = calculate_precision(
        gts.copy(), preds, threshold=threshold, form=form, ious=ious)

    return image_precision
