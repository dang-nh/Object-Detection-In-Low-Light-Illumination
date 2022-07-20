import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils import *
from filters import *
from skimage import io

def HDR(path):
    img = cv2.imread(path)

    # if len(img.shape) == 3:
    #     _, _, channels = img.shape
    #     if channels == 4:
    #         img = img[:,:,0:3]
    # else:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)

    S = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
    S = S + 1e-20
    # L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    img = 1.0*img/255

    I = cv2.GaussianBlur(S, (3, 3), 0)
    mI = np.mean(I)
    R = np.log(S+1e-20) - np.log(I+1e-20)
    R_eh = SRS(R, I)

    v_s = [0.2, (mI+0.2)/2, mI, (mI+0.8)/2, 0.8]

    I_vts = VIG(I, 1.0-I, v_s)
    L_eh = tone_production(R_eh, I_vts)
    
    ratio = np.clip(L_eh/ S, 0, 3)
    b,g,r = cv2.split(img)

    b_eh = ratio * b
    g_eh = ratio * g
    r_eh = ratio * r

    out = cv2.merge((b_eh, g_eh, r_eh))
    return np.clip(out, 0.0, 1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

	# Input Parameters

    parser.add_argument('--image_dir', type=str, default="/home/ubuntu/thanh.nt176874/dangnh/Object-Detection-In-Night-Vision/Dataset")
    # parser.add_argument('--filter', type=bool, default=True)
    parser.add_argument('--out_dir', type=str, default="/home/ubuntu/thanh.nt176874/dangnh/Object-Detection-In-Night-Vision/data/data_enhancement/HDRv3")
    args = parser.parse_args()

    list_img  = os.listdir(args.image_dir)

    for img in tqdm(list_img):

        path = os.path.join(args.image_dir, img)
        out = HDR(path)
        save_path = os.path.join(args.out_dir, img)
        cv2.imwrite(save_path, np.uint8(out*255))

