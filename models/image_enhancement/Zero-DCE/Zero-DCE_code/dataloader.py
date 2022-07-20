import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from skimage import io

random.seed(1143)


def populate_train_list(lowlight_images_path):

    image_list_lowlight = glob.glob(lowlight_images_path + "*.*")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list



class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):

        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))


    def __getitem__(self, index):

        data_lowlight_path = self.data_list[index]

        data_lowlight = io.imread(data_lowlight_path)
        # print(f"Shape of data_lowlight: {data_lowlight.shape}")

        if len(data_lowlight.shape) == 3:
            _, _, channels = data_lowlight.shape
            if channels == 4:
                data_lowlight = data_lowlight[:,:,0:3]
        else:
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
            
        data_lowlight = cv2.resize(data_lowlight, (self.size, self.size))
        data_lowlight = np.array(data_lowlight)/255.0
        data_lowlight = torch.from_numpy(data_lowlight).float()

        # data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)

        # data_lowlight = (np.asarray(data_lowlight)/255.0) 
        # data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2,0,1)


    def __len__(self):
        return len(self.data_list)

