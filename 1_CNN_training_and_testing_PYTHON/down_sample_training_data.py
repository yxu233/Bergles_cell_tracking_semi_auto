# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:10:54 2019

@author: tiger

     SimpleElastix installation:
               - install in path name WITHOUT SPACES (and not too many characters either!!!)
               - have visual studio C++ package
               - ***install target language dependencies first
                    sudo apt-get install cmake swig monodevelop r-base r-base-dev ruby ruby-dev python python-dev tcl tcl-dev tk tk-dev
               - run windows 64x native shell as ADMINISTRATOR
               - follow the GUI instructions to the DOT!!! Even deselecting all other options in the GUI to only keep Python wrappings
               - remember to go to SUPERBUILD folder within simpleElastix when setting cMake path
               - at the end, must python install
                    ==> will need to move file
                         _SimpleITK.pyd from the ...\Python\ directory to ...\Python\Packaging. 
                    and then run the installation INSIDE CONDA prompt!!!
                         or else will not install to anaconda Spyder
"""


""" Things to register:
          (1) LIVE resonance vs. airy
          (2) LIVE 2p 20x vs. airy
          (3) 63x confocal vs. resonance ==> scaled!!!

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




def plot_max(im, axis = 0):
     ma = np.amax(im, axis=axis)
     plt.figure(); plt.imshow(ma)

""" load model again """

images = glob.glob(os.path.join('./to_register/','*.tif'))

images = glob.glob(os.path.join('./to_register/3) LIVE confocal 63x vs airy/to register frame by frame/','*.tif'))


images = glob.glob(os.path.join('C:/Users/tiger/Documents/GitHub/Huganir_lab_ilastix/5) data_register_truth_to_others/to register/','*.tif'))




images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/REGISTER/','*.tif'))

images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/29) LIVE - take 3/BEST FOR REGISTRATION/for affine registration/','*.tif'))
#images = glob.glob(os.path.join('E:/Austin SEP GluA2/Full data/attempt to register_0_2um_z_scaled_RECONSTRUCTED/to watershed/affine registration/','*.tif'))


#images = glob.glob(os.path.join('E:/10) Huganir-CORE microscope lab 880/20200130-Live-take-2/Training data/to register affine/','*.tif'))
images = glob.glob(os.path.join('E:/10) Huganir-CORE microscope lab 880/Train-bad/','*.tif'))



#images = glob.glob(os.path.join('E:/10) Huganir-CORE microscope lab 880/Train-bad_to_downsample/','*.tif'))







natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)


""" Frame by frame """
timeseries_z_size = 10
timeseries_z_size = 0
idx = 0
for file_idx in range(0, len(images)):
      filename = images[file_idx]
      im = open_image_sequence_to_3D(images[file_idx], width_max='default', height_max='default', depth='default')


      """ Old xy resolution ==> 0.0598188
          old z resolution ==> 0.2873224 um
      """
      old_xy = 0.0598188
      old_z = 0.2873224

      #new_xy = 0.06346666666
      new_xy = 0.0952
      new_z = 1.0
      
      
      """ Downsample with decimate """
      z_scale_factor = round(new_z/old_z);
      xy_scale_factor = round(new_xy/old_xy);


      #downsampled_z = scipy.signal.decimate(im, 5, n=None, ftype='fir', axis=0, zero_phase=True)
      
      
      
      """ Downsample using image resize """
      z_scale_factor = old_z/new_z;
      xy_scale_factor = old_xy/new_xy;


      old_x_pixels = im.shape[1]
      old_y_pixels = im.shape[-1]
      
      old_z_pixels = im.shape[0]
      
      
      #new_xy_pixels = old_xy_pixels * xy_scale_factor
      #new_z_pixels = old_z_pixels * z_scale_factor
      
      #downsampled = skimage.transform.resize(im, [], order=3, mode='reflect', cval=0, clip=True, preserve_range=False, 
      #                         anti_aliasing=True, anti_aliasing_sigma=None)
      
      
      from skimage.transform import rescale, resize, downscale_local_mean

      
      downsampled = rescale(im, [z_scale_factor, xy_scale_factor, xy_scale_factor], order=3, mode='reflect', cval=0, clip=True, 
                                preserve_range=False, multichannel=False, anti_aliasing=True, anti_aliasing_sigma=None)
      

      """ Then upscale again """
      upsampled = resize(downsampled, [old_z_pixels, old_x_pixels, old_y_pixels], order=3, mode='reflect', cval=0, clip=True, preserve_range=False, 
                               anti_aliasing=True, anti_aliasing_sigma=None)
      
      
      
      """ Must normalize it back for some reason? """
      #plot_max(im_truth_static); plot_max(im_restored_moving); plot_max(registered_output)
     
          
      print('registered #: ' + str(idx) + ' of total: ' + '')
      imsave(filename, np.asarray(upsampled, dtype=np.uint8))
      #imsave(filename + '_output_CLAHE_' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     
      idx += 1
      
      
     



