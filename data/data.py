import os
from torchvision import datasets, transforms
import albumentations as A
import torch
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


class ExDarkDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transforms):
        super().__init__()
        self.dataframe = dataframe

        self.img_paths = dataframe['image_path'].unique()
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # Read image
        img_path = self.img_paths[idx]
        img = cv2.imread(os.path.join(self.image_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        records = self.dataframe[self.dataframe['image_path'] == img_path]

        boxes = records[["x_tl", "y_tl", "x_br", "y_br"]].values
        img_width, img_height = records[[
            "image_width", "image_height"]].values[0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = records["class"].values
        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(
            [int(img_path.split(".")[0].split('_')[-1])])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            img = sample['image']
            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return img, target, img_path


def get_transforms(mode):
    if mode == "train":
        transform = A.Compose([
            # A.Resize(224, 224, always_apply=True),
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(0.5),
            A.ShiftScaleRotate(p=0.5),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        transform = A.Compose([
            # A.Resize(224, 224, always_apply=True),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    return transform
