import cv2 
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, lsqr
from scipy.linalg import solve_banded
from scipy import sparse
import time

def wlsFilter(IN, Lambda=1.0, Alpha=1.2):
    """
    IN        : Input image (2D grayscale image, type float)
    Lambda    : Balances between the data term and the smoothness term.
                Increasing lbda will produce smoother images.
                Default value is 1.0
    Alpha     : Gives a degree of control over the affinities by 
                non-lineary scaling the gradients. Increasing alpha 
                will result in sharper preserved edges. Default value: 1.2
    """
    start_time = time.time()
    L = np.log(IN+1e-22)        # Source image for the affinity matrix. log_e(IN)
    smallNum = 1e-6
    height, width = IN.shape
    k = height * width
    print(f"Time to initialize is: {time.time() - start_time}")

    start_time = time.time()
    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, n=1, axis=0)   # axis=0 is vertical direction
    print(f"Time to compute affinites is: {time.time() - start_time}")

    start_time = time.time()
    dy = -Lambda/(np.abs(dy)**Alpha + smallNum)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')    # add zeros row
    dy = dy.flatten(order='F')
    print(f"Time to compute dy is: {time.time() - start_time}")


    start_time = time.time()
    dx = np.diff(L, n=1, axis=1)

    dx = -Lambda/(np.abs(dx)**Alpha + smallNum)
    dx = np.pad(dx, ((0,0),(0,1)), 'constant')    # add zeros col 
    dx = dx.flatten(order='F')
    print(f"Time to compute dx is: {time.time() - start_time}")
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    
    start_time = time.time()
    B = np.concatenate([[dx], [dy]], axis=0)
    d = np.array([-height,  -1])

    A = spdiags(B, d, k, k) 
    print(f"Time to compute A is: {time.time() - start_time}")

    start_time = time.time()
    e = dx 
    w = np.pad(dx, (height, 0), 'constant'); w = w[0:-height]
    s = dy
    n = np.pad(dy, (1, 0), 'constant'); n = n[0:-1]

    D = 1.0 - (e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, k, k)

    A = sparse.csr_matrix(A)
    print(f">>>> Shape of A: {A.shape} and shape of b: {IN.flatten(order='F').shape}")
    print(f"Time to compute linh tinh is: {time.time() - start_time}")

    start_time = time.time()
    # Solve
    OUT = spsolve(A, IN.flatten(order='F'))
    print(f"Time to solve is: {time.time() - start_time}")
    return np.reshape(OUT, (height, width), order='F') 

def gdft(img, r):
    eps = 0.04;
    
    I = np.double(img);
    # I = I/255;
    I2 = cv2.pow(I,2);
    mean_I = cv2.boxFilter(I,-1,((2*r)+1,(2*r)+1))
    mean_I2 = cv2.boxFilter(I2,-1,((2*r)+1,(2*r)+1))
    
    cov_I = mean_I2 - cv2.pow(mean_I,2);
    
    var_I = cov_I;
    
    a = cv2.divide(cov_I,var_I+eps)
    b = mean_I - (a*mean_I)
    
    mean_a = cv2.boxFilter(a,-1,((2*r)+1,(2*r)+1))
    mean_b = cv2.boxFilter(b,-1,((2*r)+1,(2*r)+1))
    
    q = (mean_a * I) + mean_b;
    
    return q
