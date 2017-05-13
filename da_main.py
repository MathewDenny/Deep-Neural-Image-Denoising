
"""
Created on Apr 22, 2017

@author: denny
"""



from sklearn import datasets
from sklearn import feature_extraction
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

# def grayscale(a):
#     return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)

g_r = 0
g_c = 0
g_mean = 0
g_sigma = 0.01
g_sigma10 = (10.0,10.0,10.0)
g_sigma15 = (15.0,15.0,15.0)
g_sigma25 = (25.0,25.0,25.0)
g_sigma35 = (35.0,35.0,35.0)
g_sigma45 = (45.0,45.0,45.0)
g_sigma50 = (50.0,50.0,50.0)
g_shape = (32,32)
g_imgs_set = [] 

path = "/home/denny/NYU/IMAGE/imagedata/all"
testimg_path = "/home/denny/NYU/IMAGE/imagedata/test"

def reconstruct_patches(input_patches, patch_shape):
    """ Creates an image from the patches
    """
    i = 0
    window_size_r = patch_shape[0]
    window_size_c = patch_shape[1]
    new_image = np.ones((g_r,g_c))
    print input_patches.shape
    for r in range(0,new_image.shape[0], window_size_r):
#         print "r value = ", r, window_size_r
        for c in range(0,new_image.shape[1], window_size_c):            
                recons_patch = np.reshape(input_patches[i], patch_shape)
                new_image[r:r+window_size_r,c:c+window_size_c] = recons_patch
                i += 1
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1 )                
    return new_image

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

    
def create_dataset_patches(image, patchshape, patch_shift):
    """
    Given an image list, extract patches of a given shape Patch shift
    is the certain shift amount for the successive patch location
    """
    rowstart = 0; colstart = 0
 
    patches = []
    active = image
    rowstart = 0 
     
    while rowstart <= active.shape[0] - patchshape[0]:
         
        colstart = 0       
        while colstart <= active.shape[1] - patchshape[1]:
            # Slice tuple indexing the region of our proposed patch
            region = (slice(rowstart, rowstart + patchshape[0]),
                      slice(colstart, colstart + patchshape[1]))
             
            # The actual pixels in that region.
            patch = active[region]
            # Accept the patch.
            patch_vector = patch.flatten()
            if len(patches) > 0:            
                patches = np.vstack((patches, patch_vector))            
            else:
                patches = patch_vector
                
#             patches.append(patch)
            colstart += patch_shift
            
         
        rowstart += patch_shift
 
    patches = np.matrix(patches)
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
#     del g_imgs_set[:]
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            print "NOTE: ", f, " avoided from dataset"
            continue
#         imgs.append(Image.open(os.path.join(path,f)))
        
        image = cv2.imread(os.path.join(folder,f), cv2.IMREAD_COLOR )
        
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
        norm_image = cv2.normalize(gray_image, 
                                   alpha=0, 
                                   beta=1, 
                                   dst = norm_image, 
                                   norm_type=cv2.NORM_MINMAX, 
                                   dtype=cv2.CV_32F)
        g_imgs_set.append(norm_image.flatten())
        patches = create_dataset_patches(norm_image, g_shape, patch_shift=16)
        num_patches = len(patches)
#         print "no of patches made:", num_patches, patches.shape

        if len(imgs) > 0:            
            imgs = np.vstack((imgs, patches))            
        else:
            imgs = patches
            
        
#         imgs.append(image_vector)
        
    imgs = np.matrix(imgs)    
    return imgs

def main():
    
#     images = load_images_from_folder(path)
#     print "no of images made:", len(images), images.shape
#     print "shape of input images = ", images.shape
#     data = grayscale(images)
#     
#     x = np.matrix(data)
     
     
#     print('Some examples of images we will feed to the autoencoder for training')
#     plt.rcParams['figure.figsize'] = (10, 10)
#     num_examples = 5
#     global g_r; global g_c
#     for i in range(num_examples):
#         in_image = np.reshape(images[i], g_shape)
#         print in_image.shape
#         plt.subplot(1, num_examples, i+1)
#         plt.imshow(in_image, cmap='Greys_r')
#     plt.show()
#   
#     input_dim = np.shape(images)[1]
#     print "our input_dim = " , input_dim
#     hidden_dim = 100
#     ae = da_tf.Denoiser(input_dim, hidden_dim)
    #ae.train(images, [0.019, 0.058, 0.098, 0.137])
    
    
    #TESTING IMAGES
    num_examples = 4
    hidden_dim = 100
    images = load_images_from_folder(testimg_path)
    
    input_dim = np.shape(images)[1]
    ae = da_tf.Denoiser(input_dim, hidden_dim)    
    
    print "no of images made:", len(images), images.shape
    print "shape of input images = ", images.shape
    
    data_noised = ae.add_noise([g_imgs_set[0], g_imgs_set[1]], 0, 0.12)
    print data_noised.shape
    for i in range(2):
        in_image = np.reshape(data_noised[i], (g_r, g_c))
        noisy_patches = create_dataset_patches(in_image, g_shape, patch_shift=32)
        reconstructed = ae.test(noisy_patches)
        denoised = reconstruct_patches(reconstructed,g_shape)
        denoised_image = np.reshape(denoised, (512, 512))
        print denoised_image.shape
        plt.subplot(1, num_examples, i+1)
        plt.imshow(denoised_image, cmap='Greys_r')
    plt.subplot(1, num_examples, 3)
    plt.imshow(np.reshape(data_noised[0], (512, 512)), cmap='Greys_r')
    plt.subplot(1, num_examples, 4)
    plt.imshow(np.reshape(data_noised[1], (512, 512)), cmap='Greys_r')
    plt.show()
    #reconstructed = ae.test(images)

if __name__== "__main__":
    main()
    
    

