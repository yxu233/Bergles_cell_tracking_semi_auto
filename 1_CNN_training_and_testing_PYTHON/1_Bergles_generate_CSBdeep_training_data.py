# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:18:44 2019

@author: tiger
"""

from csbdeep import data
import numpy as np
import matplotlib.pyplot as plt
import datetime


from csbdeep import io
from plot_functions import *
from data_functions import *
from data_functions_3D import *
from natsort import natsort_keygen, ns





def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

""" Load .czi into imageJ ==> then split channels and save
     must put each channel in diff folder, and rename so starts with same name!!!

We further observed that even a small number of training images 
(for example, 200 patches of size 64 × 64 × 16) led to an acceptable image restoration quality (Supplementary Fig. 6).


 26 training stacks (of size ∼700 × 700 × 50) using different samples at different developmental stages. 
 From that, we randomly sampled around 15,000 patches of size 64 × 64 × 16 and trained a 3D network as before
 
 
 
 - Image with less zoom
 - training data larger crops
 - ***get to training online!!!
 - don't train with high SNR? ==> maybe instead should just have with other training level
 
 
 
To train with "mssim_mae" or "mssim" must copy/paste current:
     - config.py
     - care_standard.py
     - train.py
     - losses.py


 
"""


#cd ... make sure in correct folder!

#%load_ext tensorboard
#%tensorboard --logdir==training:./my_model_timeseries/ --host=127.0.0.1
#%tensorboard --logdir=./my_model_with_highSNR-tensorboard/
#netstat -ano | findstr :6006
#taskkill /PID 15488 /F

#tensorboard --logdir=./my_model_HUGANIR_63x_ONLY_WORSE_Z_depth_4  --port=8002

#my_model_HUGANIR_63x_ONLY_WORSE_Z

# image size == 2048 x 2048

""" To get a cube in size:
     
     depth == 3 um/pixel * 32 slices ==> 96 um
     x_scale == 0.83 um/pixel * 128 ==> 106.24 um
     
     
     """
     
     
     
patch_size = (32, 128, 128)     # ALL MUST BE DIVISIBLE BY 4

n_patches_per_image = 50    # ideally should be 15,000
#""" Load and generate training data """
raw_data = data.RawData.from_folder(basepath='E:/7) Bergles lab data/RemyelinationData/Training_data_CSBDEEP/', source_dirs=['Raw'], 
                                    target_dir='Truth', axes='CZYX', pattern='*.tif*')

X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
                                     patch_filter=None, shuffle=True)

#io.save_training_data('training_data_CONFOCAL_PINHOLE', X, Y, XY_axes)
io.save_training_data('training_data_Bergles_cell_tracking', X, Y, XY_axes)





