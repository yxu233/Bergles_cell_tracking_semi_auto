# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:10:47 2020

@author: tiger
"""

import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import glob
import os
from natsort import natsort_keygen, ns
from plot_functions import *
from data_functions import *
from data_functions_3D import *

# - 80 dissection buffer 1st one

def mse(x, y):
    return np.linalg.norm(x - y)


images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST_SSIM_COMPARE/','*.tif'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)


for im_idx in range(0, len(images), 2):
     truth = open_image_sequence_to_3D(images[im_idx], input_size='default', depth='default')

     test = open_image_sequence_to_3D(images[im_idx + 1], input_size='default', depth='default')

     # img = img_as_float(data.camera())
     # rows, cols = img.shape
     
     # noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
     # noise[np.random.random(size=noise.shape) > 0.5] *= -1
     
     
     # img_noise = img + noise
     # img_const = img + abs(noise)
     
     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                               sharex=True, sharey=True)
     ax = axes.ravel()
     
     mse_none = mse(truth, truth)
     ssim_none = ssim(truth, truth, data_range=truth.max() - truth.min())
     psnr_none = psnr(truth, truth,  data_range=truth.max() - truth.min())
     
     mse_noise = mse(truth, test)
     ssim_noise = ssim(truth, test,
                       data_range=test.max() - test.min())
     psnr_noise = psnr(truth, test, data_range=test.max() - test.min())
     
     # mse_const = mse(truth, img_const)
     # ssim_const = ssim(truth, img_const,
     #                   data_range=img_const.max() - img_const.min())
     
     label = 'MSE: {:.2f}, SSIM: {:.2f}, PSNR: {:.2f}'
     
     
     truth_max = np.amax(truth, axis=0)
     ax[0].imshow(truth_max)
     ax[0].set_xlabel(label.format(mse_none, ssim_none, psnr_none))
     ax[0].set_title('Original image')
     
     test_max = np.amax(test, axis=0)
     test_max = np.asarray(test_max, np.uint8)
     ax[1].imshow(test_max)
     ax[1].set_xlabel(label.format(mse_noise, ssim_noise, psnr_noise))
     ax[1].set_title('Image with noise')
     
     # ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
     # ax[2].set_xlabel(label.format(mse_const, ssim_const))
     # ax[2].set_title('Image plus constant')
     
     plt.tight_layout()
     plt.show()





