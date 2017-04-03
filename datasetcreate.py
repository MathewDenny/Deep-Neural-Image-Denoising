'''
Created on Apr 2, 2017

@author: denny
'''
from sklearn.feature_extraction import image
from noise_adder import noisy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import cv2
import sys
from posix import abort

imgs = []
path = "/home/denny/NYU/IMAGE/imagedata/c512_002"

g_mean = 0

g_sigma10 = (10.0,10.0,10.0)
g_sigma15 = (15.0,15.0,15.0)
g_sigma25 = (25.0,25.0,25.0)
g_sigma35 = (35.0,35.0,35.0)
g_sigma45 = (45.0,45.0,45.0)
g_sigma50 = (50.0,50.0,50.0)

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
        
        imgs.append(image)
       
    return imgs


def main(argv):
    # My code here
    images = load_images_from_folder(path)
    print "size of input images = ", len(images)
    plot_patches(images)
    patches = create_dataset_patches(images, (64,64), 8)
    print "size of patches images = ", len(patches)

    print "Making Noisy patches:"  
    no_of_sets, noisy_patches = create_noisy_patches(patches)

    dataset = []
    
    ctr = 0
    for patch in patches:        
        for i in range(no_of_sets):
            temp = (patch, noisy_patches[ctr])
            dataset.append(temp)
            ctr += 1
            
    print "number of noise levels = ", (no_of_sets)       
    
    if (len(dataset) != (no_of_sets * len(patches))):
        print "Error in creating Noise patches"
        abort
    else:
        print "Made", len(dataset), "dataset entries" 
          
    plot_patches(noisy_patches)

if __name__ == "__main__":
    main(sys.argv)
    