import cv2
import numpy as np


def SRS(r, i):
    mI = np.mean(i)
    r_eh = np.zeros_like(r)
    mask1 = r > mI
    mask2 = r <= mI

    r_changed = r * (i/mI)**0.5
    r_eh[mask1] = r_changed[mask1]
    r_eh[mask2] = r[mask2]

    return r_eh

def VIG(i, i_inv, vEVs):
    i_inv /=np.max(i_inv)
    mI = np.mean(i)
    M = np.max(i)
    r = 1.0 - mI/M
    f_vEVs = [r*( 1/(1+np.exp(-1.0*(v - mI))) - 0.5 ) for v in vEVs]

    I_k = [(1 + fv) * (i + fv * i_inv) for fv in f_vEVs]

    return I_k

def tone_production(R_eh, I_vts):
    L_s = [np.exp(R_eh) * I for I in I_vts]
    
    Ws = np.zeros_like(I_vts)
    for i, I in enumerate(I_vts):
        if i < 3:
            Ws[i,:,:] = I/np.max(I)
        else:
            Ws[i,:,:] = 1.0 - I/np.max(I)

    L_eh = np.zeros_like(R_eh)
    W_ = np.zeros_like(R_eh)
    for W, L in zip(Ws, L_s):
        L_eh += W * L
        W_ += W
    
    L_eh = L_eh/W_
    return L_eh
