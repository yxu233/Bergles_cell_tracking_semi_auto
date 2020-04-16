# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:10:54 2019

@author: tiger


Installation notes:
    1) Anaconda
    2) pip install (everything) + 

    
    pip install matplotlib scipy scikit-image pillow numpy natsort opencv-python tifffile keras pandas
    
    
    
    pip install csbdeep numba

    mahotas? - failed
    conda config --add channels conda-forge
    conda install mahotas
    
    pip install skan *** NEW!!! allows skeleton analysis
    
    Graphics card driver
    CUDA Toolkit ==> needs VIsual studio (base package sufficient???)
    CuDnn SDK ==> 
    
    Ignore step about putting cudnn with Visual Studio

"""



""" For 2p images, try:
     - scale vs. no scale
     - z resolution scale vs. no scale
     - 20 x vs. 63x objective registered

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




""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf

""" UNNECESSARY FOR TENSORFLOW 2.1.0 """
if not tf.__version__ == '2.1.0':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

""" load model again """

# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
#(X,Y), (X_val,Y_val), axes = load_training_data('training_data_FULL.npz', validation_split=0.1)
#X = []; Y = []; X_val = []; Y_val = [];
config = Config('SZYXC', n_dim=3, train_batch_size=8, train_epochs=1000, train_steps_per_epoch=200,
                train_learning_rate=0.0004, train_tensorboard=True,  unet_kern_size=5, unet_n_depth=2)
#model = CARE(config, 'my_model_FULL_DATASET with registered') 
#model = CARE(config, 'BELUGA_my_model_FULL_DATASET') 
#model = CARE(config, 'BELUGA_my_model_FULL_DATASET_CLAHE') 

#model = CARE(config, 'my_model_CONFOCAL_PINHOLE') 

#model = CARE(config, 'my_model_2p') 
#model = CARE(config, 'my_model_timeseries') 



""" Good Huganir models to use """
#model = CARE(config, 'my_model_HUGANIR') 
#model = CARE(config, 'my_model_HUGANIR_LESS') 
#model = CARE(config, 'my_model_HUGANIR_63x') 
#model = CARE(config, 'my_model_HUGANIR_63x_ONLY_WORSE_Z') 

#model = CARE(config, 'my_model_HUGANIR_63x_cellfill_ONLY') 

#model = CARE(config, 'my_model_HUGANIR_63x_ONLY_WORSE_Z_mssim_mae_lambda_1_0') 

#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_airy')

#config = Config('SZYXC', n_dim=3, train_batch_size=8, train_epochs=1000, train_steps_per_epoch=200,
#                train_learning_rate=0.0004, train_tensorboard=True,  unet_kern_size=5, unet_n_depth=4)
#model = CARE(config, 'my_model_HUGANIR_63x_ONLY_WORSE_Z_depth_4') 


#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED') 

#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_MSSIM_MAE_0_5_redo') 

#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_MSSIM_MAE_1') 




#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_downsampled') 


#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_MSSIM_MAE_0_5_downsampled') 

#model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_MSSIM_MAE_1_downsampled') 

model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_downsampled_00952') 




model.load_weights('weights_now.h5')
# evaluate the model

""" Test Cody's images """
#images = glob.glob(os.path.join('./Codys images to enhance/','*.tif'))
#images = glob.glob(os.path.join('./Validation data/','*.tif'))
#images = glob.glob(os.path.join('./Training FULL data/Low SNR - Train/','*.tif'))
#images = glob.glob(os.path.join('../Bergles-lab/Input raw/augmented/','*.tif'))

#images = glob.glob(os.path.join('./Training CONFOCAL/Full-open pinhole/','*.tif'))

#images = glob.glob(os.path.join('./Training confocal and pinhole/Train channel 2 - worst/','*.tif'))
#images = glob.glob(os.path.join('./Training 2p confocal/Train - 2p full open pinhole/','*.tif'))
#images = glob.glob(os.path.join('./LSM test/','*.tif'))


#images = glob.glob(os.path.join('./Training data - timeseries/Train - low power/','*.tif'))

#images = glob.glob(os.path.join('./Training data - HUGANIR/Test/','*.tif'))

#images = glob.glob(os.path.join('E:/11) Huganir lab microscope 880/Testing_63x','*.tif'))


images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/','*.tif'))



#images = glob.glob(os.path.join('E:/For Daniel/attempted_registration_LIVE_RESONANCE/reconstruct/','*.tif'))



#images = glob.glob(os.path.join('./Test_folder','*.tif'))

#images = glob.glob(os.path.join('D:/Full data/preprocessed/','*.tif'))



images = glob.glob(os.path.join('E:/14_Bergles_GIAN/To_super_resolve/','*.tif'))






natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)

 
#timeseries_z_size = 10
timeseries_z_size = 0
idx = 0
for filename in images:
     im = open_image_sequence_to_3D(filename, width_max='default', height_max='default', depth='default')


     """ Old xy resolution ==> 0.0598188
         old z resolution ==> 0.2873224 um
     """
     #old_xy = 0.0598188
     #old_z = 0.2873224

     old_xy = 0.0952
     old_z = 1

     new_xy = 0.0598188
     #new_xy = 0.08
     new_z = 0.2873224
            
     
     """ Downsample using image resize """
     z_scale_factor = old_z/new_z;
     xy_scale_factor = old_xy/new_xy;


     # #old_x_pixels = im.shape[1]
     # #old_y_pixels = im.shape[-1]
 
     # #old_z_pixels = im.shape[0]
     
      
     # from skimage.transform import rescale, resize, downscale_local_mean

      
     # upsampled = rescale(im, [z_scale_factor, xy_scale_factor, xy_scale_factor], order=3, mode='reflect', cval=0, clip=True, 
     #                            preserve_range=False, multichannel=False, anti_aliasing=True, anti_aliasing_sigma=None)
      

     # im = upsampled

     


     if timeseries_z_size:
          output_combined = np.zeros(np.shape(im))
          for idx in range(0, len(im), timeseries_z_size):
               seg_im = im[idx: idx + timeseries_z_size, :, :]
               pred_med_snr = model.predict(seg_im, 'ZYX', n_tiles=(1,5,5))
               
               output_combined[idx: idx + timeseries_z_size, :, :] = pred_med_snr
          pred_med_snr = output_combined
          
     else:
          pred_med_snr = model.predict(im, 'ZYX', n_tiles=(4,16,16))
          #pred_med_snr = model.predict(im, 'ZYX')
          
          
     """ Gets rid of weird donut holes if you normalize it"""    
     # im = np.copy(pred_med_snr)
     # m,M = im.min(),im.max()
     # im_norm = (im - m) / (M - m)
     # pred_med_snr = im_norm * 255
     pred_med_snr[pred_med_snr < 0] = 0
     
     plot_max(pred_med_snr)
     #imsave(filename + '_output_' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint16))
     imsave(filename + '_output_CLAHE_MAE_ONLY' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     
     plot_max(im)
     idx += 1
     
     
     from skimage.metrics import structural_similarity as ssim
     
     ssim = ssim(im, pred_med_snr)
     
     print(ssim)
     

#im = pred_med_snr
#m,M = im.min(),im.max()
#t = ((im - m) / (M - m))
#plt.show()








