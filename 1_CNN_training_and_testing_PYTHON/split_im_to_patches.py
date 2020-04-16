# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:34:33 2019

@author: Neuroimmunology Unit
"""

import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.misc import imread, imsave, imresize


# the code produces read-only view of an image in 0.015s
# function taken from
# https://stackoverflow.com/questions/45960192/
def window_nd(a, window, steps = None, axis = None, outlist = False):
    """
    Create a windowed view over `n`-dimensional input that uses an 
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
    a : Array-like
        The array to create the view on

    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if 
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else 
            equal to `len(a.shape)`, or 
            1

    steps : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the steps along each `axis`.  
            `len(steps)` must me equal to `len(axis)`

    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    outlist : boolean
        If output should be as list of windows.  
        If False, it will be an array with 
            `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.  
        If True, output is a list of arrays with `a_view[0].ndim = a.ndim`
            Warning: this is a memory-intensive copy and not a view

    Returns
    -------

    a_view : ndarray
        A windowed view on the input array `a`, or copied list of windows   

    """
    ashp = np.array(a.shape)

    if axis != None:
        axs = np.array(axis, ndmin = 1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), \
            "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin = 1)
    assert (window.size == axs.size) | (window.size == 1), \
        "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin = 1)
        assert np.all(steps > 0), \
            "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), \
            "Steps and axes don't match"
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a, 
                                 shape = shape, 
                                 strides = strides, writeable=False))
    if outlist:
        return list(a_view.reshape((-1,) + tuple(wshp)))
    else:
        # return view (N, p_h, p_w, channels)
        return a_view.reshape((-1,) + tuple(wshp)) #a_view




# split image into patches. If padding is used, everything is ok
def patchify(img, patch_shape=(1000,1000), overlap=10):

    img_h, img_w = img.shape[:2]
    p_h, p_w = patch_shape[:2]

    # REMOVE: Padding makes the code work
    # calculate number of patches needed
    n_h = (img_h - overlap) // (p_h - overlap) 

    if (img_h - overlap) % (p_h - overlap) > 0:
        n_h += 1

    n_w = (img_w - overlap) // (p_w - overlap)

    if (img_w - overlap) % (p_w - overlap) > 0:
        n_w += 1

    h_new = (p_h - overlap)*n_h + overlap
    w_new = (p_w - overlap)*n_w + overlap

    pad_h, pad_w = h_new - img_h + 1, w_new - img_w + 1
    #img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'mean')   
    img = np.pad(img, ((0, pad_h), (0, pad_w)), "constant")     # TIGER-added


    return window_nd(img, (p_h, p_w), \
        steps=(p_h-overlap,p_w-overlap), axis=(0,1))



# simple loop to collect image back with overlap
def collect(patches, image_size, overlap=10):    

    '''

    If you have m windows of length k with an overlap of r, 
    then the total distance span, n, is given as

    n = (k-r)m + r

    Hence the number of windows, m, is

    m = (n-r)/(k-r)


    '''    

    img_h, img_w = image_size[:2]

    print('image_size {}'.format( image_size))

    n_p, p_h, p_w = patches.shape[:3]

    print('patches.shape {}'.format(patches.shape))

    # calculate number of patches needed, including overlapping ones
    n_h = (img_h - overlap) // (p_h - overlap) 

    if (img_h - overlap) % (p_h - overlap) > 0:
        n_h += 1

    n_w = (img_w - overlap) // (p_w - overlap)

    if (img_w - overlap) % (p_w - overlap) > 0:
        n_w += 1


    #img = np.zeros((img_h, img_w, image_size[2]), dtype=patches.dtype)
    img = np.zeros((img_h, img_w), dtype=patches.dtype)    # TIGER-added



    patch_idx = 0
    pos_h = 0
    pos_w = 0

    # we know that this image size is sufficient, 
    # so we cut everything which does not fit
    for i in range(n_h):

        patch_offset_h = overlap//2 if i > 0 else 0

        height_left = img_h - pos_h 

        # overlap is needed for correctness
        h_to_insert = np.min([p_h - patch_offset_h, height_left])

        for j in range(n_w):

            p = patches[patch_idx]

            patch_offset_w = overlap//2 if j > 0 else 0

            width_left = img_w - pos_w 

            w_to_insert = np.min([p_w - patch_offset_w, width_left])

            #print('h:{}, w:{}, h_i:{}, w_i:{}'.format(pos_h, pos_w,h_to_insert, w_to_insert))

            ## watching carefully the size of parts we copy
            #img[pos_h:(pos_h+h_to_insert),pos_w:(pos_w+w_to_insert),:] = p[patch_offset_h:(h_to_insert + patch_offset_h ),
            #    patch_offset_w:(w_to_insert + patch_offset_w), :]
            
            # TIGER-added
            img[pos_h:(pos_h+h_to_insert),pos_w:(pos_w+w_to_insert)] = p[patch_offset_h:(h_to_insert + patch_offset_h ),
                patch_offset_w:(w_to_insert + patch_offset_w)]

            pos_w += w_to_insert - overlap // 2

            patch_idx += 1

            print('patch {}/{}'.format(patch_idx, len(patches)))

            ### REMOVE: to save what we actually have if the number of patches is less
            if patch_idx > len(patches) - 1:
                return img

        pos_w = 0    
        pos_h += h_to_insert - overlap // 2


    return img




# Test

# this image has last row of patches not calculated
#image = imread('4960x3500.jpg')

# this image has both last row and last column not calculated
# You can see that by commenting out 
'''
if (img_w - overlap) % (p_w - overlap) > 0:
        n_w += 1
'''

#image = imread('4928x3264.jpg')
#
#start_time = time.clock()
#
#patches = patchify(image)
#
#if not type(patches) is np.ndarray:
#    patches = np.array(patches)
#
#print("Patchify took: {}s".format(time.clock() - start_time))
#
#start_time = time.clock()
#res_image = collect(patches, (image.shape[0], image.shape[1], \
#    image.shape[2]), overlap=10)
#
##res_image = collect(patches, (image.shape[0], image.shape[1]), overlap=10)
#
#print("Collect took: {}s".format(time.clock() - start_time))
#
#imsave('out.png', res_image)