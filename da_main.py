
"""
Created on Apr 22, 2017

@author: denny
"""



from sklearn import datasets
import da_tf
from matplotlib import pyplot as plt
import pickle
import numpy as np
from noise_manage import noisy
import os, os.path
import cv2
import sys
from numpy.f2py.rules import arg_rules

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)

g_r = 0
g_c = 0
g_mean = 0
g_sigma10 = (10.0,10.0,10.0)
g_sigma15 = (15.0,15.0,15.0)
g_sigma25 = (25.0,25.0,25.0)
g_sigma35 = (35.0,35.0,35.0)
g_sigma45 = (45.0,45.0,45.0)
g_sigma50 = (50.0,50.0,50.0)

path = "/home/denny/NYU/IMAGE/imagedata/all"


def create_noisy_patches(patches):
    noisy_patches = []
    
#     noisy_patch = []
    for patch in patches:
        len = 0
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma10))
        
        len += 1
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma15))
        len += 1
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma25))
        len += 1
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma35))
        len += 1
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma45))
        len += 1
        noisy_patches.append(noisy("gauss", patch, g_mean, g_sigma50))
        len += 1
#         noisy_patches.append(noisy_patch)
    return len, noisy_patches

    
def create_dataset_patches(images, patchshape, patch_shift=4):
    """
    Given an image list, extract patches of a given shape Patch shift
    is the certain shift amount for the successive patch location
    """
    rowstart = 0; colstart = 0
 
    patches = []
    for active in images:
        rowstart = 0  
        while rowstart < active.shape[0] - patchshape[0]:
             
            colstart = 0       
            while colstart < active.shape[1] - patchshape[1]:
                # Slice tuple indexing the region of our proposed patch
                region = (slice(rowstart, rowstart + patchshape[0]),
                          slice(colstart, colstart + patchshape[1]))
                 
                # The actual pixels in that region.
                patch = active[region]
                
                # Accept the patch.
                patches.append(patch)
                colstart += patch_shift
     
             
            rowstart += patch_shift
 
     
    return patches
    
def plot_patches(patches, fignum=None, low=0, high=0):
    """
    Given a stack of 2D patches indexed by the first dimension, plot the
    patches in subplots. 
    'low' and 'high' are optional arguments to control which patches
    actually get plotted. 'fignum' chooses the figure to plot in.
    """
    try:
        istate = plt.isinteractive()
        plt.ioff()
        if fignum is None:
            fig = plt.gcf()
        else:
            fig = plt.figure(fignum)
        if high == 0:
            high = len(patches)
#         pmin, pmax = patches.min(), patches.max()
        dims = np.ceil(np.sqrt(high - low))
        for idx in xrange(high - low):
            spl = plt.subplot(dims, dims, idx + 1)
            ax = plt.axis('off')
            im = plt.imshow(patches[idx], cmap=matplotlib.cm.gray)
#             cl = plt.clim(pmin, pmax)
        plt.show()
    finally:
        plt.interactive(istate)
  
def load_images_from_folder(folder):
    """
    Given a folder load all the images to a list in python and return 
    the list
    """
    imgs = []
    valid_images = [".jpg",".pbm",".png",".ppm"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            print "NOTE: ", f, " avoided from dataset"
            continue
#         imgs.append(Image.open(os.path.join(path,f)))
        
        image = cv2.imread(os.path.join(path,f), cv2.IMREAD_COLOR )
        
        # swap to RGB format
        red = image[:,:,2].copy()
        blue = image[:,:,0].copy()             
        image[:,:,0] = red
        image[:,:,2] = blue
        image = np.asarray( image)
        global g_r 
        global g_c
        if g_r == 0:
            g_r,g_c,d = image.shape
        
        norm_image = gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         norm_image = cv2.normalize(gray_image, 
#                                    alpha=0, 
#                                    beta=1, 
#                                    dst = norm_image, 
#                                    norm_type=cv2.NORM_MINMAX, 
#                                    dtype=cv2.CV_32F)
        
        image_vector = norm_image.flatten()
        if len(imgs) > 0:            
            imgs = np.vstack((imgs, image_vector))            
            
        else:
            imgs = image_vector
            
        
#         imgs.append(image_vector)
        
    imgs = np.matrix(imgs)    
    return imgs

def main():

#     hidden_dim = 1
#     data = datasets.load_iris().data
#     input_dim = len(data[0])
#     ae = da_tf.Autoencoder(input_dim, hidden_dim)
#     ae.train(data)
#     ae.test([[8, 4, 6, 2]])
    
    
    
#     names = unpickle('./cifar-10-batches-py/batches.meta')['label_names']
#     data, labels = [], []
#     for i in range(1, 6):
#         filename = './cifar-10-batches-py/data_batch_' + str(i)
#         batch_data = unpickle(filename)
#         if len(data) > 0:
#             data = np.vstack((data, batch_data['data']))
#             labels = np.hstack((labels, batch_data['labels']))
#         else:
#             data = batch_data['data']
#             labels = batch_data['labels']
#     
#     data = grayscale(data)
    
    images = load_images_from_folder(path)
    print "shape of input images = ", images.shape
#     data = grayscale(images)
#     
#     x = np.matrix(data)
     
      
    print('Some examples of images we will feed to the autoencoder for training')
#     plt.rcParams['figure.figsize'] = (10, 10)
#     num_examples = 5
#     global g_r; global g_c
#     for i in range(num_examples):
#         in_image = np.reshape(images[i], (g_r, g_c))
#         print in_image.shape
#         plt.subplot(1, num_examples, i+1)
#         plt.imshow(in_image, cmap='Greys_r')
#     plt.show()
    print "our data = " , images
  
    input_dim = np.shape(images)[1]
    print "our input_dim = " , input_dim

    hidden_dim = 100
    ae = da_tf.Denoiser(input_dim, hidden_dim)
    ae.train(images)


if __name__== "__main__":
    main()
    
    

