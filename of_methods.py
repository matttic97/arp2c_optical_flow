import numpy as np
from scipy import signal
import cv2
from ex1_utils import *


# im1 − first image matrix (grayscale)
# im2 − sec ond image mat rix (grayscale)
# N − size of the neighborhood (N x N)
def lucaskanade (im1, im2, N):
    im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    Ix, Iy = gaussderiv((im1+im2)/2, 1.4)
    It = gausssmooth(im2-im1, 1.4)
    
    Ix2 = np.power(Ix, 2)
    Iy2 = np.power(Iy, 2)
    Ixy = np.multiply(Ix, Iy)
    Ixt = np.multiply(Ix, It)
    Iyt = np.multiply(Iy, It)

    kernel_n = np.ones((N, N))
    Ix2N = signal.convolve2d(Ix2, kernel_n, boundary='symm', mode='same')
    Iy2N = signal.convolve2d(Iy2, kernel_n, boundary='symm', mode='same')
    IxyN = signal.convolve2d(Ixy, kernel_n, boundary='symm', mode='same')
    IxtN = signal.convolve2d(Ixt, kernel_n, boundary='symm', mode='same')
    IytN = signal.convolve2d(Iyt, kernel_n, boundary='symm', mode='same')
    
    D = np.multiply(Ix2N, Iy2N) - np.power(IxyN, 2) + 0.000001
    u = -(np.multiply(Iy2N, IxtN) - np.multiply(IxyN, IytN)) / D
    v = -(np.multiply(Ix2N, IytN) - np.multiply(IxyN, IxtN)) / D

    return (u.astype('float32'), v.astype('float32'))


# im1 − first image matrix (grayscale)
# im2 − sec ond image mat rix (grayscale)
# n − size of the neighborhood (N x N)
def pyramidal_lucaskanade (im1, im2, N, n):
    # image normalization
    im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # downsampled images
    P1 = [im1]
    P2 = [im2]
    for i in range(n):
        P1.append(gausssmooth(cv2.resize(P1[i], (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR), 1))
        P2.append(gausssmooth(cv2.resize(P2[i], (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR), 1))
     
    u, v = lucaskanade(P1[n], P2[n], N)
    for i in reversed(range(n)):
        shape = np.transpose(P2[i]).shape
        u = cv2.resize(u, shape, interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, shape, interpolation=cv2.INTER_LINEAR)
        flow = np.array((np.transpose(u), np.transpose(v))).T
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warp = cv2.remap(P2[i], np.transpose(flow)[0], np.transpose(flow)[1], cv2.INTER_LINEAR)
        u2,v2 = lucaskanade(np.transpose(warp), P2[i], N)
        u += u2
        v += v2        

    return (u, v)


# im1 − first image matrix (grayscale)
# im2 − sec ond image mat rix (grayscale)
# n_iters − number of iterations (try several hundred)
# lmbd - lambda
def hornschunck (im1, im2, n_iters, lmbd):
    im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    Ix, Iy = gaussderiv((im1+im2)/2, 1.4)
    It = gausssmooth(im2-im1, 1.4)
    
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    D = lmbd + np.power(Ix, 2) + np.power(Iy, 2)
    L = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for i in range(n_iters):
        ua = signal.convolve2d(u, L, boundary='symm', mode='same')
        va = signal.convolve2d(v, L, boundary='symm', mode='same')
        PD = (np.multiply(Ix, ua) + np.multiply(Iy, va) + It) / D
        u = ua - np.multiply(Ix, PD)
        v = va - np.multiply(Iy, PD)
        
    return (u, v)


# im1 − first image matrix (grayscale)
# im2 − sec ond image mat rix (grayscale)
# N - size of the neighborhood (N x N)
# n_iters − number of iterations (try several hundred)
# lmbd - lambda
def hornschunck_LK (im1, im2, N, n_iters, lmbd):
    im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # LK initialization
    Ix, Iy = gaussderiv((im1+im2)/2, 1.4)
    It = gausssmooth(im2-im1, 1.4)
    
    Ix2 = np.power(Ix, 2)
    Iy2 = np.power(Iy, 2)
    Ixy = np.multiply(Ix, Iy)
    Ixt = np.multiply(Ix, It)
    Iyt = np.multiply(Iy, It)

    kernel_n = np.ones((N, N))
    Ix2N = signal.convolve2d(Ix2, kernel_n, boundary='symm', mode='same')
    Iy2N = signal.convolve2d(Iy2, kernel_n, boundary='symm', mode='same')
    IxyN = signal.convolve2d(Ixy, kernel_n, boundary='symm', mode='same')
    IxtN = signal.convolve2d(Ixt, kernel_n, boundary='symm', mode='same')
    IytN = signal.convolve2d(Iyt, kernel_n, boundary='symm', mode='same')
    
    D = np.multiply(Ix2N, Iy2N) - np.power(IxyN, 2) + 0.000001
    u = -(np.multiply(Iy2N, IxtN) - np.multiply(IxyN, IytN)) / D
    v = -(np.multiply(Ix2N, IytN) - np.multiply(IxyN, IxtN)) / D
    
    D = lmbd + np.power(Ix, 2) + np.power(Iy, 2)
    L = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for i in range(n_iters):
        ua = signal.convolve2d(u, L, boundary='symm', mode='same')
        va = signal.convolve2d(v, L, boundary='symm', mode='same')
        PD = (np.multiply(Ix, ua) + np.multiply(Iy, va) + It) / D
        u = ua - np.multiply(Ix, PD)
        v = va - np.multiply(Iy, PD)
        
    return (u, v)