import os
from torchvision import datasets, transforms
import albumentations as A
import torch
import cv2


class ExDarkDataset(torch.utils.data.Dataset):
    def __init__(self, anno_path, transforms):
        self.anno_path = anno_path
        self.transforms = transforms
        
    def __len__(self):
        return len(self.anno_path)

    def load_image_paths(self, anno_path):
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        return [line.split(' ')[0] for line in lines]
    
    
    def load_bbox_class(self, line):
        bbox_list_with_class = [line.split(' ')[1:] for line in lines]
        bbox_list = [list(map(int, bbox.split(',')))[:4] for bbox in bbox_list_with_class]
        class_list = [list(map(int, bbox.split(',')))[-1] for bbox in bbox_list_with_class]
        return bbox_list, class_list
    
    def __getitem__(self, idx):
        
        lines = self.load_annotations(self.anno_path)
        image_paths = load_image_paths(self.anno_path)
        
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = self.transforms(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox_list, class_list = self.load_bbox_class(lines[idx])
        
        target = {}
        target['bboxs'] = torch.tensor(bbox_list)
        target['classes'] = torch.tensor(class_list)
        target['image'] = image
        
        return target
        
        
def get_transforms(mode):
    if mode == "train":
        transform = A.Compose([
                                A.Resize(224, 224, always_apply=True),
                                A.Rotate(limit=10, p=0.5),
                                A.HorizontalFlip(),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
                                ])
    else:
        transform = A.Compose([
                                A.Resize(224, 224, always_apply=True),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    return transform
