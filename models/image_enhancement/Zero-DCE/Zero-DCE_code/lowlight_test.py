import cv2
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import skimage.io as io


 
def lowlight(image_path, model, device):

    data_lowlight = io.imread(image_path)
    # data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
    if len(data_lowlight.shape) == 3:
            _, _, channels = data_lowlight.shape
            if channels == 4:
                data_lowlight = data_lowlight[:,:,0:3]
    else:
        data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)

    data_lowlight = (np.asarray(data_lowlight)/255.0)


    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    start = time.time()
    _,enhanced_image,_ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('/Dataset/','/data/data_enhancement/Zero-DCE/')
    result_path = image_path
    # if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
    #     os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == "__main__":

    device = torch.device("cuda:0"
                if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"[INFO] Torch device is {torch.cuda.current_device()}")

    print("[INFO] Loading model...")
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    # DCE_net.load_state_dict(torch.load('checkpoints/best_model.pth'))

    print("[INFO] Infering...")
    with torch.no_grad():
        filePath = '/home/ubuntu/thanh.nt176874/dangnh/Object-Detection-In-Night-Vision/Dataset/'

        file_list = os.listdir(filePath)

        for image_path in file_list:
            # image = image
            print(image_path)
            image_path = filePath + image_path
            lowlight(image_path=image_path, model=DCE_net, device=device)

		

