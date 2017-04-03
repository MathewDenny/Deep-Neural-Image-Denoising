'''
Created on Apr 3, 2017

@author: denny
'''


# 
# Parameters
# ----------
# image : ndarray
#     Input image data. Will be converted to float.
# mode : str
#     One of the following strings, selecting the type of noise to add:
# 
#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
#     'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.


import numpy as np
import os
import cv2
import numpy as np
import os, os.path
import cv2
import sys

if __name__ == '__main__':
    pass


def noisy(noise_typ,image, _mean, _sigma):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = _mean
        sigma = _sigma
#         gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = np.zeros((row,col,ch))
        cv2.randn(gauss, mean, sigma) 
        gauss = gauss.reshape(row,col,ch)
        
        noisy = image + gauss
        noisy = cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3 )
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
  