from albumentations.core.serialization import load
from utils import build_loaders, split_dataset, option, load_config
import pandas as pd
import numpy as np
import yaml
import torch
from tqdm import tqdm
from models import FasterRCNN
# from vision.references.detection.engine import evaluate
import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from utils.utils import MetricLogger, SmoothedValue, reduce_dict
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset

IMAGE_DIR = 'Dataset'
ANNOTATION_PATH = 'data/annotations.csv'

def train(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, args=None):    
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch % 2 ==0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'losslogger': metric_logger}
            torch.save(state, f"checkpoints/latest_{args.model_name}.pth")

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

# @torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    res = coco_evaluator.summarize()
    print(f"results: {res}")
    torch.set_num_threads(n_threads)
    return sum(res) / len(res)


def main(args, config):

    device = torch.device("cuda:" + str(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    print(f">>>> Torch device is {torch.cuda.current_device()}")
    print(f">>>> Preparing data....")

    batch_size = args.batch_size

    data = pd.read_csv(ANNOTATION_PATH)
    train_df, test_df = split_dataset(data, args.train_size)

    train_loader = build_loaders(
        dataframe=train_df, image_dir=IMAGE_DIR, batch_size=batch_size, num_workers=config['global']['num_workers'], mode="train")
    test_loader = build_loaders(dataframe=test_df, image_dir=IMAGE_DIR,
                                batch_size=batch_size, num_workers=config['global']['num_workers'], mode="test")

    model = FasterRCNN(num_classes=13).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                            step_size=3,
    #                                            gamma=0.1)
    best_iou = 0

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, device, epoch, print_freq=50)

        mean_iou = evaluate(model, test_loader, device=device)
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), f"checkpoints/{args.checkpoint_name}")
        print(f">>> Epoch: {epoch} - Mean IoU: {mean_iou} - Best IoU: {best_iou}")


if __name__ == "__main__":

    args = option()
    config = load_config(args.config_path)
    main(args, config)
