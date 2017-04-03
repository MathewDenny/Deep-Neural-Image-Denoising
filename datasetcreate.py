'''
Created on Apr 2, 2017

@author: denny
'''
from sklearn.feature_extraction import image
from PIL import Image
import os, os.path
import cv2
import sys

imgs = []
path = "/home/denny/NYU/IMAGE/imagedata/c512_002"


def create_dataset_patches(images):
    for image_iter in images:
        patches = image.extract_patches_2d(image_iter, (2, 2))
    return patches
    
    
def extract_patches(images, patchshape, patch_shift=4):
    """
    Given an image list, extract patches of a given shape Patch shift
    is the certain shift amount for the successive patch location
    """
    
    rowstart = 0; colstart = 0

    patches = []
    for active in images:
         
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

    # Return a 3D array of the patches with the patch index as the first
    # dimension (so that patch pixels stay contiguous in memory, in a 
    # C-ordered array).
    return np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0)

    
def load_images_from_folder(folder):
     """
    Given a folder load all the images to a list in python and return it.
    """
    valid_images = [".jpg",".pbm",".png",".ppm"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            print "NOTE: ", f, " avoided from dataset"
            continue
#         imgs.append(Image.open(os.path.join(path,f)))
        imgs.append(cv2.imread(os.path.join(path,f)))
        return imgs

def main(argv):
    # My code here
    images = load_images_from_folder(path)
    print "size of input images = ", len(images)
    patches = create_dataset_patches(images)
    print "size of patches images = ", len(patches)

if __name__ == "__main__":
    main(sys.argv)
    