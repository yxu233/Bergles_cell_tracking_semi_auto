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

patch_size = (16, 64, 64)     # ALL MUST BE DIVISIBLE BY 4
n_patches_per_image = 5000    # ideally should be 15,000
#""" Load and generate training data """
#raw_data = data.RawData.from_folder(basepath='./Training FULL data/', source_dirs=['High SNR - Train','Medium SNR - Train','Low SNR - Train'], 
#                                    target_dir='High SNR - Truth', axes='CZYX', pattern='*.tif*')
#raw_data = data.RawData.from_folder(basepath='./Training confocal and pinhole/', source_dirs=['Train channel 2 - worst','Train channel 3 - low digital gain', 'Train channel 4 - high digital gain'], 
#                                    target_dir='Truth', axes='CZYX', pattern='*.tif*')

#raw_data = data.RawData.from_folder(basepath='./Training REGISTERED/', source_dirs=['Train'], 
#                                    target_dir='Truth', axes='CZYX', pattern='*.tif*')

#X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
#                                     patch_filter=None, shuffle=True)


#patch_size = (4, 64, 64)     # ALL MUST BE DIVISIBLE BY 4
#n_patches_per_image = 8000    # ideally should be 15,000
raw_data = data.RawData.from_folder(basepath='./Training data - Huganir - LIVE/', source_dirs=['Train-medium'], 
                                   target_dir='Truth', axes='CZYX', pattern='*.tif*')





""" If need to generate training data, UNCOMMENT """
X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
                                      shuffle=True)
                                        # CAN TURN BACK ON PATCH_FILTER ==> to reduce background patches
io.save_training_data('training_data_HUGANIR_LIVE', X, Y, XY_axes)



""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf

if not tf.__version__ == '2.1.0':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


""" Start training """
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
(X,Y), (X_val,Y_val), axes = load_training_data('training_data_HUGANIR_LIVE.npz', validation_split=0.1)

#config = Config(axes, n_dim=3, train_batch_size=8, train_epochs=1000, train_steps_per_epoch=200,
#                train_learning_rate=0.0004, train_tensorboard=True,  unet_kern_size=5, unet_n_depth=2)


config = Config(axes, n_dim=3, train_batch_size=16, train_epochs=1000, train_steps_per_epoch=400,
                train_learning_rate=0.0004, train_tensorboard=True,  unet_kern_size=5, unet_n_depth=2)

model = CARE(config, 'my_model_HUGANIR_LIVE')
model.train(X,Y, validation_data=(X_val,Y_val))

model.export_TF()


""" Test Medium SNR
images = glob.glob(os.path.join('./Training data/Medium SNR - Train/','*.tif'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)

idx = 0
for filename in images:
     im = open_image_sequence_to_3D(filename, input_size='default', depth='default')

     pred_med_snr = model.predict(im, 'ZYX', n_tiles=(1,25,25))
     
     plot_max(pred_med_snr)
     imsave(filename + '_' + str(idx) + '_high_snr.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     
     plot_max(im)
     idx += 1


#Test Low SNR
images = glob.glob(os.path.join('./Training data/Low SNR - Train/','*.tif'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)

idx = 0
for filename in images:
     im = open_image_sequence_to_3D(filename, input_size='default', depth='default')

     pred_low_snr = model.predict(im, 'ZYX', n_tiles=(1,25,25))
     
     plot_max(pred_low_snr)
     imsave(filename + '_' + str(idx) + '_low_snr.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     plot_max(im)
     idx += 1



#Get ground truth
images = glob.glob(os.path.join('./Training data/High SNR - Truth/','*.tif'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)

for filename in images:
     im_truth = open_image_sequence_to_3D(filename, input_size='default', depth='default')

     plot_max(im_truth)


"""


