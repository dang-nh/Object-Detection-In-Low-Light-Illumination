import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from metrics import calculate_image_precision
from models import FasterRCNN, make_ensemble_predictions, run_wbf
from utils import AverageMeter, EvalMeter, build_loaders, split_dataset


def train(model, train_loader, epoch, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    tqdm_object = tqdm(train_loader, desc=f"Train Epoch {epoch}", total=len(train_loader))

    for images, targets, img_paths in tqdm_object:
        # images, targets, image_path = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if k == 'labels' else v.float().to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # Return the loss

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_meter.send(loss_value)  # Average out the loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return loss_meter.avg


def eval(model, val_loader, device):
    validation_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    model.eval()
    tqdm_object = tqdm(
        val_loader, desc=f"Validating.....", total=len(val_loader))
    with torch.no_grad():
        for images, targets, imageids in tqdm_object:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if k == 'labels' else v.float().to(device) for k, v in t.items()} for t in targets]
            #outputs = model(images)

            predictions = make_ensemble_predictions(images, model=model)

            for i, image in enumerate(images):
                boxes, scores, labels = run_wbf(predictions, image_index=i, model=model)
                boxes = boxes.astype(np.int32).clip(min=0, max=1023)

                preds = boxes  # outputs[i]['boxes'].data.cpu().numpy()
                #scores = outputs[i]['scores'].data.cpu().numpy()
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = preds[preds_sorted_idx]
                gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
                image_precision = calculate_image_precision(preds_sorted,
                                                            gt_boxes,
                                                            thresholds=iou_thresholds,
                                                            form='coco')

                validation_image_precisions.append(image_precision)
    return np.mean(validation_image_precisions)

def main():
    device = torch.device("cuda:3"
                          if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f">>>> Torch device is {torch.cuda.current_device()}")
    print(f">>>> Preparing data....")

    batch_size = 1
    image_dir = 'Dataset'
    data = pd.read_csv('data/annotations.csv')
    train_df, test_df = split_dataset(data, 0.2)

    train_loader = build_loaders(dataframe=train_df, image_dir = image_dir, batch_size=batch_size, num_workers=4, mode="train")
    test_loader = build_loaders(dataframe=test_df, image_dir = image_dir, batch_size=batch_size, num_workers=4, mode="test")

    model = FasterRCNN(num_classes=13).to(device)
    # model.load_state_dict(torch.load(
    #     'weights/fasterrcnn_resnet50_fpn_best.pth'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    best_val = None
    for epoch in range(1, 10):
        print(f">>>> Epoch {epoch}/10")
        train_loss = train(model, train_loader, epoch, optimizer, device=device)

        print(f">>>> Validating.....")
        train_iou = eval(model, train_loader, device=device)
        print(f">>>> Train IoU: {train_iou}")

        print(f">>>> Validating.....")
        test_iou = eval(model, test_loader, device=device)
        print(f">>>> Test IoU: {test_iou}")

        if not best_val or test_iou > best_val:
            print(f">>>> Saving model....")
            # torch.save(model.state_dict(), "weights/fasterrcnn_resnet50_fpn_best.pth")
            best_val = test_iou

if __name__ == "__main__":
    main()
