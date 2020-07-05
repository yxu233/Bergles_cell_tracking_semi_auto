# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
"""

#import tensorflow as tf
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import scipy
import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
#from UNet import *
#from UNet_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

#from csbdeep.internals import predict
from tifffile import *
import tkinter
from tkinter import filedialog

import pandas as pd
from skimage import measure


""" Required to allow correct GPU usage ==> or else crashes """
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *
from UNet_functions_PYTORCH import *
from matlab_crop_function import *

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
s_path = './(1) Checkpoints_full_auto_no_spatialW/'

crop_size = 160
z_size = 32
num_truth_class = 2

lowest_z_depth = 135;
lowest_z_depth = 150;

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
num_check = int(num_check[0])

check = torch.load(s_path + checkpoint, map_location=device)

unet = check['model_type']
unet.load_state_dict(check['model_state_dict'])
unet.to(device)
unet.eval()
#unet.training # check if mode set correctly

print('parameters:', sum(param.numel() for param in unet.parameters()))

# """ load mean and std """  
input_path = './normalize_pytorch_CLEANED/'
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"

initial_dir = '/media/user/storage/Data/'
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
    print('Do you want to select another folder? (y/n)')
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
    initial_dir = input_path
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO'

    """ For testing ILASTIK images """
    #images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    #images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    #examples = [dict(input=i,seg=i.replace('_single_channel.tif','_single_channel_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


    images = glob.glob(os.path.join(input_path,'*_input_im.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,seg=i.replace('_input_im.tif','_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]




    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    batch_size = 1;
    
    batch_x = []; batch_y = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    
    

   
    """ Initialize matrix of cells """
    num_timeseries = len(examples)
    columns = list(range(0, num_timeseries))
    matrix_timeseries = pd.DataFrame(columns = columns)
    
   
    input_name = examples[0]['input']            
    input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
    depth_total, empty, empty = input_im.shape
    
    input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
 
    seg_name = examples[0]['seg']  
    cur_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
    cur_seg = cur_seg[0:lowest_z_depth, ...]
    cur_seg = np.moveaxis(cur_seg, 0, -1)
     
    #height_tmp, width_tmp, depth_tmp = input_im.shape
     
    
    ### HACK: ??? #####################################################################################################################################
    """ WHICH WAY IS CORRECT??? """
    width_tmp, height_tmp, depth_tmp = input_im.shape
    
    """ Get truth from .csv as well """
    truth = 1
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'
         truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv'   # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv'
         
         
         truth_name = '680_syGlass_10x.csv'
         
         
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         
         
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})
         
     
    TN = 0; TP = 0; FN = 0; FP = 0; doubles = 0; extras = 0; skipped = 0; blobs = 0;
    for i in range(1, len(examples)):

       
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[i]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            next_input = next_input[0:lowest_z_depth, ...]
            next_input = np.moveaxis(next_input, 0, -1)
     
   
            seg_name = examples[i]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg = next_seg[0:lowest_z_depth, ...]
            next_seg = np.moveaxis(next_seg, 0, -1)
            
            
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=i, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                     



            """ loop through each cell in cur_seg and find match in next_seg
            
                 ***keep track of double matched cells
                 ***append to dataframe
            """
            cur_seg[cur_seg > 0] = 1
            labelled = measure.label(cur_seg)
            cur_cc = measure.regionprops(labelled)
            
            
            iterator = 0;
            for cell in cur_cc:
                 
                 x, y, z = cell['centroid']
                 x = int(x); y = int(y); z = int(z);
                 coords = cell['coords']
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 if len(coords) < 20:
                      continue;
                 
                 
                 blank_im = np.zeros(np.shape(input_im))
                 blank_im[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                 
                 
                 crop_cur_seg, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(cur_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 
                 """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
                 if z + z_size/2 >= lowest_z_depth:
                      print('skip')
                      skipped += 1
                      continue
                 
                 """ TRY REGISTRATION??? """
                 # import SimpleITK as sitk
                 # elastixImageFilter = sitk.ElastixImageFilter()
                 # elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(crop_im))
                 # elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(crop_next_input))
                 # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
                 # im = elastixImageFilter.Execute()
                 # im_arr = sitk.GetArrayFromImage(im)
                 # im_arr[im_arr >= 255] = 255
                 # im_arr[im_arr < 0] = 0
                 # #sitk.WriteImage(elastixImageFilter.GetResultImage())
                 # crop_next_input = im_arr;
                 
                 
                 
                 
                 
                 crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(blank_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_im, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(input_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_cur_seg[crop_cur_seg > 0] = 10
                 crop_cur_seg[crop_seed > 0] = 50                 
                    
                 crop_next_input, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(next_input, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
                 crop_next_seg, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_next_seg[crop_next_seg > 0] = 10
               
                 
                 """ Get ready for inference """
                 batch_x = np.zeros((4, ) + np.shape(crop_im))
                 batch_x[0,...] = crop_im
                 batch_x[1,...] = crop_cur_seg
                 batch_x[2,...] = crop_next_input
                 batch_x[3,...] = crop_next_seg
                 batch_x = np.moveaxis(batch_x, -1, 1)
                 batch_x = np.expand_dims(batch_x, axis=0)
                 
                 
                 batch_x = normalize(batch_x, mean_arr, std_arr)
               
               
                 #batch_y = np.zeros([1, num_truth_class, quad_depth, quad_size, quad_size])

           
                 """ Convert to Tensor """
                 inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
                 #labels_val = torch.tensor(batch_y, dtype = torch.long, device=device, requires_grad=False)
     
                 # forward pass to check validation
                 output_val = unet(inputs_val)

                 """ Convert back to cpu """                                      
                 output_val = output_val.cpu().data.numpy()            
                 output_val = np.moveaxis(output_val, 1, -1)
                 seg_train = np.argmax(output_val[0], axis=-1)  
                 seg_train = np.moveaxis(seg_train, 0, -1)
                 
                 iterator += 1
                 
                 """ Find coords of identified cell and scale back up, then add to df_matrix
                 
                      also create an actual image matrix to keep track as you go?
                      
                      
                      ***KEEP TRACK OF DOUBLES
                      
                      
                      Testing cell: 1427 of total: 1473
                                TPs = 530; FPs = 61; TNs = 460; FNs = 107
                      
                         
                      
                      Testing cell: 1427 of total: 1473
                         TPs = 542; FPs = 84; TNs = 422; FNs = 84
                         
                         
                         355
                 """
                       
                 
                    
                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!"""
                 label = measure.label(seg_train)
                 cc_seg_train = measure.regionprops(label)
                 if len(cc_seg_train) > 1:
                      doubles += 1
                                 
                    
                 
                    
                 

                 """ Check if TP, TN, FP, FN """
                 if truth:
                      crop_truth_cur, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(truth_cur_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
                      crop_truth_next, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(truth_next_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
     
                      """ DEBUG """
                      # plot_max(crop_cur_seg, ax=-1)
                      # plot_max(crop_im, ax=-1)
                      # plot_max(crop_next_input, ax=-1)
                      # plot_max(crop_next_seg, ax=-1)
                      # plot_max(seg_train, ax=-1) 
                      # plot_max(crop_truth_cur, ax=-1)
                      # plot_max(crop_truth_next, ax=-1)
                      # plt.pause(0.0005)
                      # plt.pause(0.0005)                      
                      print('TPs = ' + str(TP) + '; FPs = ' + str(FP) + '; TNs = ' + str(TN) + '; FNs = ' + str(FN) + '; extras = ' + str(extras))
                      


                      seg_train = dilate_by_ball_to_binary(seg_train, radius = 3)  ### DILATE A BIT
                      crop_next_seg = dilate_by_ball_to_binary(crop_next_seg, radius = 3)  ### DILATE A BIT
                      crop_seed = dilate_by_ball_to_binary(crop_seed, radius = 3)
                      
                      
                      """ REMOVE EVERYTHING IN CROP_NEXT_SEG THAT DOES NOT MATCH WITH SOMETHING CODY PUT UP, to prevent FPs  of unknown checking"""
                      
                      
                      # if nothing in the second frame
                      value_cur_frame = np.unique(crop_truth_cur[crop_seed > 0])
                      value_cur_frame = np.delete(value_cur_frame, np.where(value_cur_frame == 0)[0][0])  # DELETE zero
                      
                      values_next_frame = np.unique(crop_truth_next[crop_next_seg > 0])
                      
                      ### skip if no match on cur frame in truth
                      if len(value_cur_frame) == 0:
                           continue
                      
                      
                      
                      if not np.any(value_cur_frame == values_next_frame):   ### if it does NOT exist on next frame               
                      
                      
                           ### BUT IF EXISTS IN GENERAL ON 2nd frame, just not in the segmentation, then skip ==> is segmentation missed error
                           values_next_frame_all = np.unique(crop_truth_next[crop_truth_next > 0])
                           if np.any(value_cur_frame == values_next_frame_all):
                                continue; # SKIP
                           
                      
                           
                           ### count blobs:
                           if len(value_cur_frame) > 1:
                                blobs += 1
                         
                           ### AND if seg_train says it does NOT exist ==> then is TRUE NEGATIVE
                           if np.count_nonzero(seg_train) == 0:       
                                TN += 1     
                           else:
                                ### otherwise, all of the objects are False positives
                                
                                #FP += len(cc_seg_train)
                                FP += 1
                                
                      else:
                           
                           ### (1) if seg_train is empty ==> then is a FALSE NEGATIVE
                           if np.count_nonzero(seg_train) == 0:
                                print('depth_cur: ' + str(np.where(crop_truth_cur == value_cur_frame)[-1][0])) 
                                print('depth_next: ' + str(np.where(crop_truth_next == value_cur_frame)[-1][0]))
                                FN += 1
                                
                           else:
                                
                              ### (2) find out if seg_train has identified point with same index as previous frame
                              values_next_frame = np.unique(crop_truth_next[seg_train > 0])
                                                          
                              values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == 0)[0][0])  # delete zeros
                           
                                
                              if np.any(value_cur_frame == values_next_frame):     
                                     TP += 1
                                     values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == values_next_frame)[0][0])
                                
                                
                                
                                
                              """ 
                                   Add indices to a truth array using 
                                
                                   
                                     truth_array[np.where(truth_array.SERIES == value_cur_frame) : + 1]
                              """
                              row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == 0)]
                              truth_output_df = truth_output_df.append(row)                       
                                
                              # but if have more false positives
                              if len(values_next_frame) > 0:
                                #FP += len(values_next_frame)
                                extras += len(values_next_frame)
                 
                           
                 
                      plt.close('all')  
                      
                                    
                 print('Testing cell: ' + str(iterator) + ' of total: ' + str(len(cur_cc))) 

            print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples))) 
            
            
            
            
            
            """ Set next frame to be current frame """
            input_im = next_input
            cur_seg = next_seg
            truth_cur_im = truth_next_im
            
            
    """ Compare dataframes to see of the tracked cells, how well they were tracked """     
    if truth:
         all_lengths = []
         for cell_num in np.unique(truth_output_df.SERIES):
               track_length_SEG =  len(np.where(truth_output_df.SERIES == cell_num)[0])
                
               track_length_TRUTH = len(np.where(truth_array.SERIES == cell_num)[0])


               all_lengths.append(track_length_TRUTH - track_length_SEG - 1)

    plt.figure(); plt.plot(all_lengths)
    len(np.where(np.asarray(all_lengths) > 0)[0])

    #truth_output_df = truth_output_df.sort_values(by=['SERIES'])
    
    
    ### overall 130 overtracked by 1 or 2 frames out of 1080
    ### no negatives??? nothing was undertracked???
    