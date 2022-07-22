import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from .utils import *
from .filters import *
from skimage import io

def HDR(path):
    img = cv2.imread(path)

    img = img.astype(np.float32)

    S = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
    S = S + 1e-20
    # L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    img = 1.0*img/255

    # Comppute the illumination map
    I = cv2.GaussianBlur(S, (3, 3), 0)
    mI = np.mean(I)

    # Compute the reflectance map by apply the Retinex theory
    R = np.log(S+1e-20) - np.log(I+1e-20)
    R_eh = SRS(R, I)

    # Define 5 virtual exposure value vEV
    v1 = 0.2
    v3 = mI
    v5 = 0.8
    v2 = (mI+0.2)/2
    v4 = (mI+0.8)/2

    vEVs = [v1, v2, v3, v4, v5]

    I_inv = 1.0 - I
    I_virtuals = VIG(I, I_inv, vEVs)
    L_eh = tone_production(R_eh, I_virtuals)
    
    ratio = np.clip(L_eh/S, 0, 3)
    b,g,r = cv2.split(img)

    b_eh = ratio * b
    g_eh = ratio * g
    r_eh = ratio * r

    out = cv2.merge((b_eh, g_eh, r_eh))
    return np.clip(out, 0.0, 1.0)

def enhance(image_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    list_img = os.listdir(image_dir)
    for img_name in tqdm(list_img):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isdir(img_path):
            save_path = os.path.join(out_dir, img_name)
            out = HDR(img_path)
            cv2.imwrite(save_path, np.uint8(out*255))

