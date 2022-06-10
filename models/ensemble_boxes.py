from ensemble_boxes import *

#device = torch.device('cuda:0')


def make_ensemble_predictions(images, model):
    models = [model]
    images = list(image.to(device) for image in images)
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result


def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.5, weights=None, model=None):
    models = [model]
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1)
             for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy()
              for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0])
              for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(
        boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels
