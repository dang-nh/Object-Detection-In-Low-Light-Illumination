from utils import build_loaders, AverageMeter, EvalMeter, split_dataset
import pandas as pd
import numpy as np
import yaml
import torch
from tqdm import tqdm
from models import FasterRCNN, make_ensemble_predictions, run_wbf
from metrics import calculate_precision


def train(model, train_loader, epoch, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    tqdm_object = tqdm(
        train_loader, desc=f"Train Epoch {epoch}", total=len(train_loader))
    itr = 1
    for images, targets, img_paths in tqdm_object:
        # images, targets, image_path = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if k == 'labels' else v.float().to(
            device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # Return the loss

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_meter.send(loss_value)  # Average out the loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")
        itr += 1

    return loss_value


def eval(model, val_loader, device):
    validation_image_precisions = []
    iou_thresholds = 0.5
    detection_threshold = 0.5
    model.eval()
    tqdm_object = tqdm(
        val_loader, desc=f"Validating.....", total=len(val_loader))

    epoch_acc = []
    epoch_iou = []
    with torch.no_grad():
        for images, targets, imageids in tqdm_object:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if k == 'labels' else v.float().to(
                device) for k, v in t.items()} for t in targets]
            # print(f">>>>imageids: {imageids}")

            outputs = model(images)
            # predictions = make_ensemble_predictions(images, model=model, device=device)
            # print(">>>> Predictions: ", outputs)
            acc = []
            iou = []
            for i, image in enumerate(images):
                true = 0
                # Formate of the output's box is [Xmin,Ymin,Xmax,Ymax]
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                # print(f">>>> boxes: {boxes}")
                scores = outputs[i]['scores'].data.cpu().numpy()

                # Compare the score of output with the threshold and
                # boxes = boxes[scores >= detection_threshold].astype(np.int32)
                # scores = scores[scores >= detection_threshold]

                boxes = boxes.astype(np.int32).clip(min=0, max=1333)

                preds = boxes  # outputs[i]['boxes'].data.cpu().numpy()
                # print(f">>>> Predictions: {preds}")
                #scores = outputs[i]['scores'].data.cpu().numpy()
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = preds[preds_sorted_idx]
                # print(f">>>> Predictions sorted: {preds_sorted}")
                gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
                # image_precision = calculate_image_precision(preds=preds_sorted,
                #                                             gts=gt_boxes,
                #                                             threshold=iou_thresholds,
                #                                             form='pascal_voc')
                best_match_pred_idx, best_iou = calculate_precision(preds=preds_sorted,
                                                                    gts=gt_boxes,
                                                                    threshold=iou_thresholds,
                                                                    form='pascal_voc')

                for idx in best_match_pred_idx.keys():
                    if outputs[i]['labels'][best_match_pred_idx[idx]] == targets[i]['labels'][idx]:
                        true += 1
                    else:
                        best_iou[idx] = 0
                acc.append(true/len(best_match_pred_idx))
                iou.append(sum(best_iou)/len(best_match_pred_idx))

                validation_image_precisions.append(image_precision)
            epoch_acc.appen(sum(acc)/len(acc))
            epoch_iou.append(sum(iou)/len(iou))
    return sum(epoch_acc)/len(epoch_acc), sum(epoch_iou)/len(epoch_iou)


def main():
    device = torch.device("cuda:2"
                          if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f">>>> Torch device is {torch.cuda.current_device()}")
    print(f">>>> Preparing data....")

    batch_size = 2
    image_dir = 'Dataset'
    data = pd.read_csv('data/annotations.csv')
    train_df, test_df = split_dataset(data, 0.2)

    train_loader = build_loaders(
        dataframe=train_df, image_dir=image_dir, batch_size=batch_size, num_workers=16, mode="train")
    test_loader = build_loaders(dataframe=test_df, image_dir=image_dir,
                                batch_size=batch_size, num_workers=16, mode="test")

    model = FasterRCNN(num_classes=13).to(device)
    # model.load_state_dict(torch.load(
    #     'weights/fasterrcnn_resnet50_fpn_best.pth'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)

    best_iou = 0
    for epoch in range(1, 30):
        print(f">>>> Epoch {epoch}/30")
        train_loss = train(model, train_loader, epoch,
                           optimizer, device=device)

        print(f">>>> Validating.....")
        train_acc, train_iou = eval(model, train_loader, device=device)
        print(f">>>> Train IoU: {train_iou}")

        print(f">>>> Validating.....")
        test_acc, test_iou = eval(model, test_loader, device=device)
        print(f">>>> Test IoU: {test_iou}")

        if test_iou > best_iou:
            print(f">>>> Saving model....")
            torch.save(model.state_dict(),
                       "checkpoints/fasterrcnn_resnet50_fpn_best.pth")
            best_iou = test_iou
        print(
            f"Current IoU: {test_iou} - Currenct Accuracy: {test_acc} - Best IoU: {best_iou}")


if __name__ == "__main__":
    main()
