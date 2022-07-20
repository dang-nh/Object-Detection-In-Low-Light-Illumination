import os
from torchvision import datasets, transforms
import albumentations as A
import torch
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
# from preprocess.exposure_enhancement import enhance_image_exposure

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

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
        img_path_og = self.img_paths[idx]
        img_path = img_path_og.split('.')[0] + '.png'
        # print(f"img_path: {os.path.join(self.image_dir, img_path)}")
        img = cv2.imread(os.path.join(self.image_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = img.astype(np.float32)
        img /= 255.0
        
        img_width, img_height = img.shape[1], img.shape[0]
        records = self.dataframe[self.dataframe['image_path'] == img_path_og]

        # print(f"Records: {records}")
        
        og_width, og_height = records[[
            "image_width", "image_height"]].values[0]

        boxes = records[["x_tl", "y_tl", "x_br", "y_br"]].values
        try:
            boxes = boxes.astype(np.float32)
        except:
            print(f"Image error is {img_path}")

        if int(og_width) > 2000 or int(og_height) > 2000:
            boxes = (boxes * 0.6 ) - 1

        for box in boxes:
            # x_tl, y_tl, x_br, y_br = box
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(img_width, box[2])
            box[3] = min(img_height, box[3])
        # boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        # boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        # img_width, img_height = records[[
        #     "image_width", "image_height"]].values[0]

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
            try:
                target['boxes'] = torch.stack(
                    tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            except:
                print(f"img path error is {img_path}")

        return img, target


def get_transforms(mode):
    if mode == "train":
        transform = A.Compose([
            # A.Resize(224, 224, always_apply=True),
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(0.5),
            # A.ShiftScaleRotate(p=0.5),
            # A.OneOf(
            # [
            #     A.ShiftScaleRotate(
            #         rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
            #     ),
            #     A.IAAAffine(shear=15, p=0.5, mode="constant"),
            # ],
            # p=1.0,
            # ),
            # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            # A.Blur(p=0.1),
            # A.CLAHE(p=0.1),
            # A.Posterize(p=0.1),
            # A.ToGray(p=0.1),
            # A.ChannelShuffle(p=0.05),
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

# def test():
#     img_path = 'Dataset/2015_06963.jpg'
#     img = cv2.imread(img_path).cvtColor(cv2.COLOR_BGR2RGB)

