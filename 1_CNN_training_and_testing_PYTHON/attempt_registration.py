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




def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

""" load model again """

images = glob.glob(os.path.join('./to_register/','*.tif'))

images = glob.glob(os.path.join('./to_register/3) LIVE confocal 63x vs airy/to register frame by frame/','*.tif'))


images = glob.glob(os.path.join('C:/Users/tiger/Documents/GitHub/Huganir_lab_ilastix/5) data_register_truth_to_others/to register/','*.tif'))




images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/REGISTER/','*.tif'))

images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/29) LIVE - take 3/BEST FOR REGISTRATION/for affine registration/','*.tif'))
#images = glob.glob(os.path.join('E:/Austin SEP GluA2/Full data/attempt to register_0_2um_z_scaled_RECONSTRUCTED/to watershed/affine registration/','*.tif'))


#images = glob.glob(os.path.join('E:/10) Huganir-CORE microscope lab 880/20200130-Live-take-2/Training data/to register affine/','*.tif'))
images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/43) LIVE validation FOR REAL/2) Affine registrations frame-by-frame/','*.tif'))


images = glob.glob(os.path.join('E:/Austin SEP GluA2/TEST/43) LIVE validation FOR REAL/6) Run test/Compare TRUTH v RESTORED/New folder/','*.tif'))




natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)


""" Frame by frame """
timeseries_z_size = 10
timeseries_z_size = 0
idx = 0
for file_idx in range(0, len(images), 2):
      filename = images[file_idx + 1]
      im_truth_static = open_image_sequence_to_3D(images[file_idx], width_max='default', height_max='default', depth='default')

      im_restored_moving = open_image_sequence_to_3D(images[file_idx + 1], width_max='default', height_max='default', depth='default')

      registered_output = [];
      for i in range(0, len(im_truth_static)):
            frame_truth_static = im_truth_static[i, :, :]
            frame_restored_moving = im_restored_moving[i, :, :]
           
            import SimpleITK as sitk
            """ Using object oriented sitk """
          
            #transform_type = 'rigid'
            transform_type = 'affine'
            #transform_type = 'non_rigid'
            #transform_type = 'rigid_affine'
            sitk_truth_static = sitk.GetImageFromArray(frame_truth_static/255)
            sitk_restored_moving = sitk.GetImageFromArray(frame_restored_moving/255)
          
          
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.SetFixedImage(sitk_truth_static)
            elastixImageFilter.SetMovingImage(sitk_restored_moving)
          
          
            parameterMapVector = sitk.GetDefaultParameterMap(transform_type)
               
            elastixImageFilter.SetParameterMap(parameterMapVector)
               
               
            resultImage = elastixImageFilter.Execute()
          
            resultImage = elastixImageFilter.GetResultImage()
            resultImage = sitk.GetArrayFromImage(resultImage)
            transformParameterMap = elastixImageFilter.GetTransformParameterMap()
           
            registered_output.append(resultImage)


      """ Must normalize it back for some reason? """
      im = np.asarray(registered_output)
      
      im[im < 0] = 0;
      m,M = im.min(),im.max()
      im_norm = (im - m) / (M - m)
      registered_output = im_norm * 255
      
      plot_max(im_truth_static); plot_max(im_restored_moving); plot_max(registered_output)
     
          
      print('registered #: ' + str(idx) + ' of total: ' + '')
      imsave(filename + '_frame_by_frame_' + transform_type + '_' + str(idx) + '.tif', np.asarray(registered_output, dtype=np.uint8))
      #imsave(filename + '_output_CLAHE_' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     
      idx += 1
      
      
     









""" Volume to volume registration """

# #timeseries_z_size = 10
# timeseries_z_size = 0
# idx = 0
# for file_idx in range(0, len(images), 2):
#       filename = images[file_idx + 1]
#       im_truth_static = open_image_sequence_to_3D(images[file_idx], width_max='default', height_max='default', depth='default')

#       im_restored_moving = open_image_sequence_to_3D(images[file_idx + 1], width_max='default', height_max='default', depth='default')

#       import SimpleITK as sitk
#       """ Using object oriented sitk """
     
#       #transform_type = 'rigid'
#       transform_type = 'affine'
#       #transform_type = 'non_rigid'
#       #transform_type = 'rigid_affine'
#       sitk_truth_static = sitk.GetImageFromArray(im_truth_static/255)
#       sitk_restored_moving = sitk.GetImageFromArray(im_restored_moving/255)
     
     
#       elastixImageFilter = sitk.ElastixImageFilter()
#       elastixImageFilter.SetFixedImage(sitk_truth_static)
#       elastixImageFilter.SetMovingImage(sitk_restored_moving)
     
     
#       if transform_type == 'non_rigid':
#           parameterMapVector = sitk.VectorOfParameterMap()
#           parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#           parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
#           elastixImageFilter.SetParameterMap(parameterMapVector)

#       elif transform_type == 'rigid_affine':
          
#           elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('rigid'))
#           elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
           
#       else:
#           parameterMapVector = sitk.GetDefaultParameterMap(transform_type)
          
#           elastixImageFilter.SetParameterMap(parameterMapVector)
          
          
#       resultImage = elastixImageFilter.Execute()
     
#       resultImage = elastixImageFilter.GetResultImage()
#       resultImage = sitk.GetArrayFromImage(resultImage)
#       transformParameterMap = elastixImageFilter.GetTransformParameterMap()


#       """ Must normalize it back for some reason? """
#       im = resultImage
#       m,M = im.min(),im.max()
#       im_norm = (im - m) / (M - m)
#       resultImage = im_norm * 255
      
#       plot_max(im_truth_static); plot_max(im_restored_moving); plot_max(resultImage)
      
#       #sitk.WriteImage(elastixImageFilter.GetResultImage())
      
      
      
#       # if timeseries_z_size:
#       #      output_combined = np.zeros(np.shape(im))
#       #      for idx in range(0, len(im), timeseries_z_size):
#       #           seg_im = im[idx: idx + timeseries_z_size, :, :]
#       #           pred_med_snr = model.predict(seg_im, 'ZYX', n_tiles=(1,5,5))
               
#       #           output_combined[idx: idx + timeseries_z_size, :, :] = pred_med_snr
#       #      pred_med_snr = output_combined
          
#       # else:
#       #      pred_med_snr = model.predict(im, 'ZYX', n_tiles=(1,4,4))
#       #      #pred_med_snr = model.predict(im, 'ZYX')
          
#       print('registered #: ' + str(idx) + ' of total: ' + '')
#       imsave(filename + '_registered_' + transform_type + '_' + str(idx) + '.tif', np.asarray(resultImage, dtype=np.uint8))
#       #imsave(filename + '_output_CLAHE_' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint8))
     
#       idx += 1
     

""" Register all to first image """

# timeseries_z_size = 0
# idx = 0
# im_truth_static = open_image_sequence_to_3D(images[0], input_size='default', depth='default')
# for file_idx in range(0, len(images), 1):
#      filename = images[file_idx]
#      im_restored_moving = open_image_sequence_to_3D(images[file_idx], input_size='default', depth='default')

#      # import SimpleITK as sitk


#      # resultImage = sitk.Elastix(sitk_truth_static, sitk_restored_moving, "translation")
#      # resultImage = sitk.GetArrayFromImage(resultImage)
     
#      # plot_max(im_truth_static); plot_max(im_restored_moving); plot_max(resultImage)
     

#      import SimpleITK as sitk
#      """ Using object oriented sitk """
     
#      #transform_type = 'rigid'
#      transform_type = 'affine'
#      #transform_type = 'non_rigid'
#      sitk_truth_static = sitk.GetImageFromArray(im_truth_static)
#      sitk_restored_moving = sitk.GetImageFromArray(im_restored_moving)
     
     
#      elastixImageFilter = sitk.ElastixImageFilter()
#      elastixImageFilter.SetFixedImage(sitk_truth_static)
#      elastixImageFilter.SetMovingImage(sitk_restored_moving)
     
     
#      if transform_type == 'non_rigid':
#           parameterMapVector = sitk.VectorOfParameterMap()
#           parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#           parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
               
#      else:
#           parameterMapVector = sitk.GetDefaultParameterMap(transform_type)
          
#      elastixImageFilter.SetParameterMap(parameterMapVector)
#      resultImage = elastixImageFilter.Execute()
     
#      #resultImage = elastixImageFilter.GetResultImage()
#      resultImage = sitk.GetArrayFromImage(resultImage)
#      transformParameterMap = elastixImageFilter.GetTransformParameterMap()

#      plot_max(im_truth_static); plot_max(im_restored_moving); plot_max(resultImage)

#      # if timeseries_z_size:
#      #      output_combined = np.zeros(np.shape(im))
#      #      for idx in range(0, len(im), timeseries_z_size):
#      #           seg_im = im[idx: idx + timeseries_z_size, :, :]
#      #           pred_med_snr = model.predict(seg_im, 'ZYX', n_tiles=(1,5,5))
               
#      #           output_combined[idx: idx + timeseries_z_size, :, :] = pred_med_snr
#      #      pred_med_snr = output_combined
          
#      # else:
#      #      pred_med_snr = model.predict(im, 'ZYX', n_tiles=(1,4,4))
#      #      #pred_med_snr = model.predict(im, 'ZYX')
          
#      imsave(filename + '_registered_' + transform_type + '_' + str(idx) + '.tif', np.asarray(resultImage, dtype=np.uint8))
#      #imsave(filename + '_output_CLAHE_' + str(idx) + '.tif', np.asarray(pred_med_snr, dtype=np.uint8))
#      print('registered #: ' + str(idx) + ' of total: ' + '')
     
#      idx += 1




