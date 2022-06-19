from utils import build_loaders, AverageMeter, EvalMeter, split_dataset
import pandas as pd
import numpy as np
import yaml
import torch
from tqdm import tqdm
from models import FasterRCNN, make_ensemble_predictions, run_wbf
from metrics import calculate_precision
# from vision.references.detection.engine import evaluate
from utils.engine import train_one_epoch, evaluate


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


# def eval(model, val_loader, device):
#     validation_image_precisions = []
#     iou_thresholds = 0.5
#     detection_threshold = 0.5
#     model.eval()
#     tqdm_object = tqdm(
#         val_loader, desc=f"Validating.....", total=len(val_loader))

#     epoch_acc = []
#     epoch_iou = []
#     with torch.no_grad():
#         for images, targets, imageids in tqdm_object:
#             images = list(image.to(device) for image in images)
#             targets = [{k: v.to(device) if k == 'labels' else v.float().to(
#                 device) for k, v in t.items()} for t in targets]
#             # print(f">>>>imageids: {imageids}")

#             outputs = model(images)
#             # predictions = make_ensemble_predictions(images, model=model, device=device)
#             # print(">>>> Predictions: ", outputs)
#             acc = []
#             iou = []
#             for i, image in enumerate(images):
#                 true = 0
#                 # Formate of the output's box is [Xmin,Ymin,Xmax,Ymax]
#                 boxes = outputs[i]['boxes'].data.cpu().numpy()
#                 # print(f">>>> boxes: {boxes}")
#                 scores = outputs[i]['scores'].data.cpu().numpy()

#                 # Compare the score of output with the threshold and
#                 # boxes = boxes[scores >= detection_threshold].astype(np.int32)
#                 # scores = scores[scores >= detection_threshold]

#                 boxes = boxes.astype(np.int32).clip(min=0, max=1333)

#                 preds = boxes  # outputs[i]['boxes'].data.cpu().numpy()
#                 # print(f">>>> Predictions: {preds}")
#                 #scores = outputs[i]['scores'].data.cpu().numpy()
#                 preds_sorted_idx = np.argsort(scores)[::-1]
#                 preds_sorted = preds[preds_sorted_idx]
#                 # print(f">>>> Predictions sorted: {preds_sorted}")
#                 gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#                 # image_precision = calculate_image_precision(preds=preds_sorted,
#                 #                                             gts=gt_boxes,
#                 #                                             threshold=iou_thresholds,
#                 #                                             form='pascal_voc')
#                 best_match_pred_idx, best_iou = calculate_precision(preds=preds_sorted,
#                                                                     gts=gt_boxes,
#                                                                     threshold=iou_thresholds,
#                                                                     form='pascal_voc')

#                 for idx in best_match_pred_idx.keys():
#                     if outputs[i]['labels'][best_match_pred_idx[idx]] == targets[i]['labels'][idx]:
#                         true += 1
#                     else:
#                         best_iou[idx] = 0
#                 acc.append(true/len(best_match_pred_idx))
#                 iou.append(sum(best_iou)/len(best_match_pred_idx))

#                 validation_image_precisions.append(image_precision)
#             epoch_acc.appen(sum(acc)/len(acc))
#             epoch_iou.append(sum(iou)/len(iou))
#     return sum(epoch_acc)/len(epoch_acc), sum(epoch_iou)/len(epoch_iou)


def eval(model, test_loader, epoch, optimizer, device):
    model.eval()

    loss_meter = AverageMeter()
    tqdm_object = tqdm(
        test_loader, desc=f"Validating...", total=len(train_loader))
    itr = 1

    with(torch.no_grad()):
        for images, targets, img_paths in tqdm_object:
            # images, targets, image_path = batch

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if k == 'labels' else v.float().to(
                device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # Return the loss

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_meter.send(loss_value)  # Average out the loss

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1

    return loss_value


def main():
    device = torch.device("cuda:1"
                          if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f">>>> Torch device is {torch.cuda.current_device()}")
    print(f">>>> Preparing data....")

    batch_size = 8
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
    optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
    best_iou = 1e9
    # for epoch in range(1, 30):
        # print(f">>>> Epoch {epoch}/30")
        # train_loss = train(model, train_loader, epoch,
        #                    optimizer, device=device)

        # print(f">>>> Validating.....")
        # # tr = eval(model, train_loader, device=device)
        # # print(f">>>> Train IoU: {train_iou}")

        # # print(f">>>> Validating.....")
        # # test_loss = eval(model, test_loader, device=device)
        # # print(f">>>> Test loss: {test_loss}")

        # evaluate(model, test_loader, device=device)

        # # if test_loss < best_loss:
        # #     print(f">>>> Saving model....")
        # #     torch.save(model.state_dict(),
        # #                "checkpoints/fasterrcnn_resnet50_fpn_best.pth")
        # #     best_loss = test_loss
        # print(
        #     f"Current loss: {test_loss} - Best loss: {best_loss}")
    best_iou = 0

    for epoch in range(1,31):
    # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

        lr_scheduler.step()
        # if epoch % 2 ==0:
        #     # save lastest model
        #     state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
        #      'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
        #     torch.save(state, 'checkpoints/lastest.pth')
    # update the learning rate
    # evaluate on the test dataset
        mean_iou = evaluate(model, test_loader, device=device)
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "checkpoints/best.pth")
        print(f">>> Epoch: {epoch} - Mean IoU: {mean_iou} - Best IoU: {best_iou}")


if __name__ == "__main__":
    main()
