# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

class SSD(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.model = torchvision.models.detection.ssd300_vgg16(
            pretrained=True)
        
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
        
