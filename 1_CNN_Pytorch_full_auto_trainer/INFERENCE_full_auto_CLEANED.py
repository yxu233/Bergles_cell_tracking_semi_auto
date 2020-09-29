# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
"""

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pandas as pd
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
from tifffile import *
import tkinter
from tkinter import filedialog

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from UNet_pytorch_online import *
from PYTORCH_dataloader import *
from UNet_functions_PYTORCH import *

from matlab_crop_function import *
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
from functions_cell_track_auto import *
from skimage.transform import rescale, resize, downscale_local_mean

""" Define transforms"""
import torchio
from torchio.transforms import (
   RescaleIntensity,
   RandomFlip,
   RandomAffine,
   RandomElasticDeformation,
   RandomMotion,
   RandomBiasField,
   RandomBlur,
   RandomNoise,
   Interpolation,
   Compose
)
from torchio import Image, Subject, ImagesDataset
                

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True


""" Set globally """
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)


""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
#s_path = './(1) Checkpoints_full_auto_no_spatialW/'
#s_path = './(2) Checkpoints_full_auto_spatialW/'


#s_path = './(4) Checkpoints_full_auto_no_spatialW_large_TRACKER/'; next_bool = 1;

#s_path = './(6) Checkpoints_full_auto_no_spatialW_large_TRACKER_NO_NEXT_SEG/'; next_bool = 0;

s_path = './(7) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_SEG/'; next_bool = 0;


#s_path = './(7) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_SEG_skipped/'; next_bool = 0;


#s_path = './(8) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_YES_NEXT_SEG/'; next_bool = 1;

crop_size = 160
z_size = 32
num_truth_class = 2

lowest_z_depth = 180;

scale_for_animation = 0

#min_size = 80

min_size = 10

both = 0


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

unet = check['model_type']; unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
#unet.training # check if mode set correctly

print('parameters:', sum(param.numel() for param in unet.parameters()))








""" Load 2nd model """
if both:
    s_path = './(6) Checkpoints_full_auto_no_spatialW_large_TRACKER_NO_NEXT_SEG/';
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
    
    unet_NO_NEXT = check['model_type']; unet_NO_NEXT.load_state_dict(check['model_state_dict'])
    unet_NO_NEXT.to(device); unet.eval()
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

net_two_identified = 0;
net_two_tested = 0;
    
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_sub_z_SINGLE_UNET_crop_pads_lowest_180_PREDICTIONS'

    """ For testing ILASTIK images """
    # images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,seg=i.replace('_single_channel.tif','_single_channel_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]

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
    

    """ Initialize matrix of cells """   
    input_name = examples[0]['input']            
    input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
    depth_total, empty, empty = input_im.shape
    
    #input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
    width_tmp, height_tmp, depth_tmp = input_im.shape
    
    
    
    
    seg_name = examples[0]['seg']  
    cur_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
    #cur_seg = cur_seg[0:lowest_z_depth, ...]
    cur_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
    
    cur_seg = np.moveaxis(cur_seg, 0, -1)
     
    """ loop through each cell in cur_seg and find match in next_seg
    """
    cur_seg[cur_seg > 0] = 1
    labelled = measure.label(cur_seg)
    cur_cc = measure.regionprops(labelled)
    tracked_cells_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z', 'coords', 'visited'})     
    
    
    """ add the cells from the first frame into "tracked_cells" matrix """ 
    for cell in cur_cc:
         if not np.isnan(np.max(tracked_cells_df.SERIES)):
              series = np.max(tracked_cells_df.SERIES) + 1
         else:
                   series = 1
         centroid = cell['centroid']
  
         """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
         # if int(centroid[2]) + z_size/2 >= lowest_z_depth:
         #       continue
         
         coords = cell['coords']
         
         
         """ DONT TEST IF TOO SMALL """
         if len(coords) < min_size:
              continue;         
         
         row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': 0, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
         tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
          
    
    """ Get truth from .csv as well """
    truth = 0
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'; scale = 0
         #truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv';  scale = 1  # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv';    # gets hazy at the end
         truth_name = '680_syGlass_10x.csv'                  

         #truth_name = 'MOBPF_190106w_5_cuprBZA_10x.tif - T=0_650_syGlass_10x.csv'   # well registered and clean window except for single frame

         
         
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})
         
    else:
        truth_cur_im = 0; truth_array = 0; truth_output_df = 0;
         
    

    """ Start looping through segmented nuclei """
    list_exclude = [];
    TN = 0; TP = 0; FN = 0; FP = 0; doubles = 0; extras = 0; skipped = 0; blobs = 0; not_registered = 0; double_linked = 0; seg_error = 0;
        
   

    for frame_num in range(1, len(examples)):
         print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
         all_dup_indices = [];
        
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[frame_num]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            #next_input = np.moveaxis(next_input[0:lowest_z_depth, ...], 0, -1)
            next_input = np.moveaxis(next_input, 0, -1)
            
            
            seg_name = examples[frame_num]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
            next_seg = np.moveaxis(next_seg, 0, -1)


            """ Plot for animation """
            if scale_for_animation and frame_num <= 2:
                track_cur_seg = np.zeros(np.shape(next_seg))
                
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
           
            """ Iterate through all cells """
            iterator = 0;            
            
            cell_size = []
            for cell_idx in np.where(tracked_cells_df.visited == 0)[0]: 
                 
                 cell = tracked_cells_df.iloc[cell_idx]
                 
                 ### go to unvisited cells
                 x = cell.X; y = cell.Y; z = cell.Z;

                 ### SO DON'T VISIT AGAIN
                 tracked_cells_df.visited[cell_idx] = 1
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 cell_size.append(len(cell.coords))
                 if len(cell.coords) < min_size:
                           continue;
                 
                 """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
                 # if z + z_size/2 >= lowest_z_depth:
                 #      print('skip'); skipped += 1
                 #      continue
                 

                 """ Crop and prep data for CNN, ONLY for the first 2 frames so far"""
                 batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
                                                                                                         next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
                                                                                                         height_tmp, width_tmp, depth_tmp, next_bool=next_bool)
                 
                
                 
                 ### Convert to Tensor
                 inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)

                 # forward pass to check validation
                 output_val = unet(inputs_val)

                 """ Convert back to cpu """                                      
                 output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                 seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)

                 iterator += 1



                 """ For trying out different warping functions 
                 
                         cell_idx in image timeseries 680 that are good:
                             - 600 ==> nice clean simple cell
                             
                 """

                 test_permutations = 0
                 if test_permutations:
                     
                   
                    plot_max(crop_im, ax=-1)
                    plot_max(crop_cur_seg, ax=-1)
                    plot_max(crop_next_input, ax=-1)
                    #plot_max(crop_next_seg, ax=-1)
                    crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
                    crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2              
                    plot_max(crop_next_seg_non_bin, ax=-1)               
                    plot_max(seg_train, ax=-1)
                  
                    p = 1 
                  
                    ### (1) try with different flips
                    #transforms = [RandomFlip(axes = 0, flip_probability = 1, p = p, seed = None)]; transform = Compose(transforms)
                    
                    ### (2) try with different blur
                    #transforms = [RandomBlur(std = (0, 4), p = p, seed=None)]; transforms = Compose(transforms)
                    
                    
                    ### (3) try with different warp (affine transformatins)
                    transforms = [RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
                                        default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
                                        p = p, seed=None)]; transforms = Compose(transforms)                    

                    ### (4) try with different warp (elastic transformations)
                    #transforms = [RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
                    #                                locked_borders = 2, image_interpolation = Interpolation.LINEAR,
                    #                                p = p, seed = None),]; transforms = Compose(transforms)

                    ### (5) try with different motion artifacts
                    #transforms = [RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = Interpolation.LINEAR,
                    #                    p = p, seed = None),]; transforms = Compose(transforms)


                    ### (6) try with different noise artifacts
                    #transforms = [RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None)]; transforms = Compose(transforms)

                     
                    ### transforms to apply to crop_im                     
                    inputs = crop_im
                    inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
                    #labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
                    labels = inputs
                
                    subject_a = Subject(
                            one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
                            a_segmentation=Image(None, torchio.LABEL, labels))
                      
                    subjects_list = [subject_a]
            
                    subjects_dataset = ImagesDataset(subjects_list, transform=transforms)
                    subject_sample = subjects_dataset[0]
                      
                      
                    """ MUST ALSO TRANSFORM THE SEED IF IS ELASTIC, rotational transformation!!!"""
                      
                    X = subject_sample['one_image']['data'].numpy()
                    Y = subject_sample['a_segmentation']['data'].numpy()
                     
                    if next_bool:
                        batch_x = np.zeros((4, ) + np.shape(crop_im))
                        batch_x[0,...] = X
                        batch_x[1,...] = crop_cur_seg
                        batch_x[2,...] = crop_next_input
                        batch_x[3,...] = crop_next_seg
                        batch_x = np.moveaxis(batch_x, -1, 1)
                        batch_x = np.expand_dims(batch_x, axis=0)
                
                    else:
                        batch_x = np.zeros((3, ) + np.shape(crop_im))
                        batch_x[0,...] = X
                        batch_x[1,...] = crop_cur_seg
                        batch_x[2,...] = crop_next_input
                        #batch_x[3,...] = crop_next_seg
                        batch_x = np.moveaxis(batch_x, -1, 1)
                        batch_x = np.expand_dims(batch_x, axis=0)
                
                    
                    ### NORMALIZE
                    batch_x = normalize(batch_x, mean_arr, std_arr)                 

                    ### Convert to Tensor
                    inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
    
                    # forward pass to check validation
                    output_val = unet(inputs_val)
    
                    """ Convert back to cpu """                                      
                    output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                    seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
    
                    plot_max(X[0], ax=-1)
                    plot_max(seg_train, ax=-1)
                     
                 

                 
                   
                   

                 """ Use other neural network if current seg is EMPTY """
                 if both:
                     if len(np.where(seg_train > 0)[0]) == 0:
                         batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
                                                                                                             next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
                                                                                                             height_tmp, width_tmp, depth_tmp, next_bool=0)
                         inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
                         output_val = unet_NO_NEXT(inputs_val)
                         """ Convert back to cpu """                                      
                         output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                         seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
        
                         #iterator += 1
                         net_two_tested += 1
                              
                         if len(np.where(seg_train > 0)[0]) > 0:
                             net_two_identified += 1
                
                
                
                
                
                

                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!
                 
                 
                         *** NEED TO FIX THIS TOO!!!
                 
                 """
                 cc_seg_train, seg_train, crop_next_seg = select_one_from_excess(seg_train, crop_next_seg)
                 
                 """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                 """
                 new = 0
                 if len(cc_seg_train) > 0:
                      next_coords = cc_seg_train[0].coords
                      
                      # if len(next_coords) > 2000:
                      #     print('check large blobs')
                      
                      next_coords = scale_coords_of_crop_to_full(next_coords, box_xyz, box_over)
                      
                      next_centroid = np.asarray(cc_seg_train[0].centroid)
                      next_centroid = scale_single_coord_to_full(next_centroid, box_xyz, box_over)
                      
                      
                      """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
                      next_coords = check_limits([next_coords], width_tmp, height_tmp, depth_tmp)[0]
                      next_centroid = check_limits_single([next_centroid], width_tmp, height_tmp, depth_tmp)[0]


                      ### add to matrix 
                      row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                      tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     
 
                      """ FIND DOUBLES EARLY TO CORRECT AS YOU GO """
                      if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250): ### if this place has already been visited in the past
                           print('double_linked'); double_linked += 1
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                           """ should also switch to use prediction, rather than distance???
                                   or just flag for association later???
                           """
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                           tracked_cells_df, dup_series = sort_double_linked(tracked_cells_df, next_centroid, frame_num)
                           #all_dup_indices.append(dup_series)
                           all_dup_indices = np.concatenate((all_dup_indices, dup_series))
                           
                           # plot_max(crop_im, ax=-1)
                           # plot_max(crop_cur_seg, ax=-1)
                           # plot_max(crop_next_input, ax=-1)

                           # crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
                           # crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
                           
                           # plot_max(crop_next_seg_non_bin, ax=-1)
                            
                           # plot_max(seg_train, ax=-1)
                           # len(np.where(crop_seed)[0])                            
                           
                           
                           
                      """ set current one to be value 2 so in future will know has already been identified """
                      next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
                      new = 1;
                      
                 else:
                           
                           ####   DEBUG if not matched

                           len(np.where(crop_seed)[0])         
                      
                        
                 """ Check if TP, TN, FP, FN """
                 if truth:
                     
                      TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude = parse_truth(truth_cur_im,  truth_array, truth_output_df, truth_next_im, 
                                                                                                              seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size,
                                                                                                              blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp)
                 """ Plot for animation """
                 if scale_for_animation and frame_num < 2:
                      input_name = examples[0]['input']
                      filename = input_name.split('/')[-1]
                      filename = filename.split('.')[0:-1]
                      filename = '.'.join(filename)
                      low_crop = 0.3; high_crop = 0.7; 
                      z_crop_h = 0.8
                      
                      
                      """ Skip if not within middle crop"""
                      if np.min(cell.coords[:, 0]) < track_cur_seg.shape[0] * low_crop or np.max(cell.coords[:, 0]) > track_cur_seg.shape[0] * high_crop or  np.min(cell.coords[:, 1]) < track_cur_seg.shape[1] * low_crop or np.max(cell.coords[:, 1]) > track_cur_seg.shape[1] * high_crop or np.max(cell.coords[:, 2]) > track_cur_seg.shape[2] * z_crop_h:
                          print('not animated')
                      
                      else:
                          ### PLOT CUR FRAME
                          
                          
                          
                          """ 
                                  how about carving out (setting ot 0) the intensities on the RAW data so can better 
                                      see intensity values???
                          
                          
                          """
                          
                          
                          
                          
                          
                          track_cur_seg[track_cur_seg > 0] = 150    ### MAKE THE BACKGROUND PAST TRACES DIMMER
                          track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                          
                          im = convert_matrix_to_multipage_tiff(input_im)
                          
                          
                          
                          ### or just crop it
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + 'ANIMATION_iterator_' + str(iterator) + '_frame_num_' + str(frame_num - 1) + '_cell_num_' + str(cell_idx) + '_cur_input.tif',  np.asarray(im * 255, dtype=np.uint8))
        
                          im = convert_matrix_to_multipage_tiff(track_cur_seg)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + 'ANIMATION_iterator_' + str(iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
    
    
    
                          ### PLOT NEXT FRAME
                          plot_next = np.copy(next_seg)
                          plot_next[plot_next != 250] = 0     ### delete everything that hasn't been visited
                          plot_next[plot_next > 0] = 150      ### MAKE THE BACKGROUND PAST TRACES DIMMER
                          if new:  ### if a new cell was added, use those coords to highlight it brighter than the rest
                              plot_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 255; 
                                                  
                        
                          im = convert_matrix_to_multipage_tiff(next_input)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + 'ANIMATION_iterator_' + str(iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_input.tif',  np.asarray(im * 255, dtype=np.uint8))
                          
                          im = convert_matrix_to_multipage_tiff(plot_next)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + 'ANIMATION_iterator_' + str(iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
                                 



                
                 print('Testing cell: ' + str(iterator) + ' of total: ' + str(len(np.where(tracked_cells_df.visited == 0)[0]))) 
                 #print(all_dup_indices)





            """ Re-try all the doubly-associated cells 
            
                    - things to fix:
                            - while looping, if has > 2 cells identified, ratio might not be best way to settle it???
                                    - look for distance? and whether or not has been pre-identified?/doubled?
            
            
                    - add error checking at the end using the distance/direction shift???
                            ***find # that go in wrong direction and fix these FIRST???
                    
                    
                    - what happens if predicted next cell is already pre-occupied???
                    
                    
                    
                    
                    ***plot to see how well you can predict next cell location!!! based on surroundings!!!!
                            *** use truth_array???
                    
            
            """
            
            
            """ #1 == check all cells with tracked predictions """
            tmp = tracked_cells_df.copy()
            tmp_next_seg = np.copy(next_seg)
            
            
            zzz
            


            """  Identify all potential errors of tracking (over the predicted distance error threshold) """
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            plt.figure(); plt.hist(all_dist)
            
            """ Keep looping above and start searnchig from LARGEST distance cells to correct """
            concat = np.transpose(np.asarray([check_series, dist_check]))
            sortedArr = concat[concat[:,1].argsort()[::-1]]
            check_sorted_series = sortedArr[:, 0]
             

            """ Check all cells with distance that is going against prediction """
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, check_sorted_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=12)
            
            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            plt.figure(); plt.hist(all_dist)
            
            
            new_candidates = np.concatenate((check_series, recheck_series))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, new_candidates, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)

                


            
            """ #2 == Check duplicates """
            all_dup_indices = np.unique(all_dup_indices)
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, all_dup_indices, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            plt.figure(); plt.hist(all_dist)

            ### check on recheck_series
            recheck_series = np.unique(recheck_series)
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)
            print(recheck_series)
            
            # for dup_idx in all_dup_indices:
            #      index = np.where((tracked_cells_df["SERIES"] == dup_idx) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
                
            #      if len(index) == 0:
            #          continue;
                     
            #      cell = tracked_cells_df.iloc[index[0]]
                 
            #      ### go to unvisited cells
            #      x = cell.X; y = cell.Y; z = cell.Z;

            #      ### SO DON'T VISIT AGAIN
            #      tracked_cells_df.visited[dup_idx] = 1
                 
                 
            #      """ DONT TEST IF TOO SMALL """
            #      # cell_size.append(len(cell.coords))
            #      # if len(cell.coords) < min_size:
            #      #          continue;
                 
            #      """ TO HELP WITH DEBUG, delete later"""
            #      batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
            #                                                                                              next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
            #                                                                                              height_tmp, width_tmp, depth_tmp, next_bool=next_bool,
            #                                                                                              retry=1)
                 
                 
            #      # ### Convert to Tensor
            #      # inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)

            #      # # forward pass to check validation
            #      # output_val = unet(inputs_val)

            #      # """ Convert back to cpu """                                      
            #      # output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
            #      # seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)

            #      # iterator += 1
                 


            #      label = measure.label(crop_next_seg)
            #      cc = measure.regionprops(label)
            #      next_coords = []
            #      if len(cc) > 0:                 
            #          """ Associate based on distance of shift of nearby cells!!! (if available) otherwise, do by distance?
                     
            #                  need to get series number of all cells in the current frame and then get coordinates of current + next frame
            #          """
                 
            #          pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size, z_size, frame_num)
                         
            #          ### EXPAND CROP SIZE if less than 5 cells tracked within current crop
            #          if num_tracked < 5:
            #              pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size  +  crop_size * 0.25, z_size + z_size * 0.25)
                       
                         
            #          ### continue to use predicted xyz only if num_tracked > 5:
            #          if num_tracked >= 5:
            #              cell_next, next_coords, seg_train, next_centroid, min_dist = associate_to_closest(tracked_cells_df, cc, seg_train, pred_x, pred_y, pred_z, box_xyz, box_over, dup_idx, 
            #                                                                                               frame_num, width_tmp, height_tmp, depth_tmp, min_dist=20)   

            #              plot_max(crop_im, ax=-1)
            #              plot_max(crop_cur_seg, ax=-1)
            #              plot_max(crop_next_input, ax=-1)

            #              crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
            #              crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
                       
            #              plot_max(crop_next_seg_non_bin, ax=-1)
                        
            #              plot_max(seg_train, ax=-1)
                         
            #          else:  ### just use original distances
            #              cell_next, next_coords, seg_train, next_centroid, min_dist = associate_to_closest(tracked_cells_df, cc, seg_train, x, y, z, box_xyz, box_over, dup_idx, 
            #                                                                                                frame_num, width_tmp, height_tmp, depth_tmp, min_dist=20)
                         
            #              print('not enough cells')



            #          """ Change next coord only if something close was found
                    
            #                  otherwise...??? drop it???
            #          """
            #          if len(next_coords) > 0:   ### only add if not empty
            #              cell_next.coords = next_coords
            #              cell_next.X = next_centroid[0]
            #              cell_next.Y = next_centroid[1]
            #              cell_next.Z = next_centroid[2]
              
               
            #              next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
    

            
            #      if len(cc) == 0 or len(next_coords) == 0:   ### OTHERWISE, DROP THE CELL

            #         """ drop individual frame """
            #         tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(((tracked_cells_df["SERIES"] == dup_idx) & (tracked_cells_df["FRAME"] == frame_num)))[0]])
                
            #         """  OR DROP THE ENTIRE TRACK MAYBE??? BECAUSE THESE ARE UNCERTAIN CELLS ANYWAYS..."""
                
            #         tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(((tracked_cells_df["SERIES"] == dup_idx)))])
                     
            
            
            """ #3 == check all terminated cells """
            all_cur_frame = np.where(tracked_cells_df["FRAME"] == frame_num - 1)[0]
            cur_series = tracked_cells_df.iloc[all_cur_frame].SERIES
            term_series = []
            for cur in cur_series:
                index_cur = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
                index_next = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num))[0]
                 
                """ if next frame is empty, then terminated """
                if len(index_next) == 0:
                    term_series.append(cur)

            
            term_series = np.concatenate((term_series, recheck_series))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, term_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            plt.figure(); plt.hist(all_dist)
            
            
            
            ### check on recheck_series
            recheck_series = np.unique(recheck_series)
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)
            print(recheck_series)
            
            
            
            
            
            """ #4 repeat until no more recheck series """
            
            # ### THESE ARE LIKELY CELLS THAT ARE DOUBLED UP!!!
            # while len(recheck_series) > 0:
            #     recheck_series = np.unique(recheck_series)
            #     tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)
            #     print(recheck_series)
                
            
            # num_checked = 0
            # matched = 0;
            # for cur in cur_series:
            #      index_cur = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
            #      index_next = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num))[0]
                 
            #      """ if next frame is empty, then terminated """
            #      if len(index_next) == 0:
                     
            #          cell = tracked_cells_df.iloc[index_cur[0]]
                     
            #          ### go to unvisited cells
            #          x = cell.X; y = cell.Y; z = cell.Z;
    
            #          ### SO DON'T VISIT AGAIN
            #          tracked_cells_df.visited[dup_idx] = 1
                     
                     
            #          """ DONT TEST IF TOO SMALL """
            #          # cell_size.append(len(cell.coords))
            #          # if len(cell.coords) < min_size:
            #          #          continue;
                     
            #          """ Crop and prep data for CNN, ONLY for the first 2 frames so far"""
            #          # batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
            #          #                                                                                         next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
            #          #                                                                                         height_tmp, width_tmp, depth_tmp, next_bool=next_bool,
            #          #                                                                                     retry=1)

            #          # inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
    
            #          # # forward pass to check validation
            #          # output_val = unet(inputs_val)
    
            #          # """ Convert back to cpu """                                      
            #          # output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
            #          # seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)




            #          """ Associate with nearest within 20 pixels """
            #          label = measure.label(crop_next_seg)
            #          cc = measure.regionprops(label)
            #          next_coords = []
            #          if len(cc) > 0:
            #             cell_next, next_coords, seg_train, next_centroid, min_dist = associate_to_closest(tracked_cells_df, cc, seg_train, x, y, z, box_xyz, box_over, cur,
            #                                                                                               frame_num, width_tmp, height_tmp, depth_tmp, min_dist=20)

                    
            #          if len(next_coords) > 0:
            #             row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
            #             tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     
    
            #             """ Change next coord """
            #             next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
            #             matched += 1
                        # plot_max(crop_im, ax=-1)
                        # plot_max(crop_cur_seg, ax=-1)
                        # plot_max(crop_next_input, ax=-1)
                        # plot_max(crop_next_seg, ax=-1)
                 
      
                        # plot_max(seg_train, ax=-1)
                        # len(np.where(crop_seed)[0])
                        # crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
                        # crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
                      
                        # plot_max(crop_next_seg_non_bin, ax=-1)   
                        # print(next_centroid[-1])                     

            #          num_checked += 1
                     
            #          # if num_checked == 5:
            #          #     zzz
                     
            #      # ### Convert to Tensor            
    
            

            """ associate remaining cells that are "new" cells and add them to list to check as well as the TRUTH tracker """
            if not truth:
                truth_next_im = 0
            tracked_cells_df, truth_output_df, truth_next_im, truth_array = associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, min_size,
                                                                           truth=truth, truth_output_df=truth_output_df, truth_next_im=truth_next_im, truth_array=truth_array)

                             
                    
            """ delete any 100% duplicated rows
            
            
                    ***figure out WHY they are occuring??? probably from re-checking???
            """
            tmp_drop = tracked_cells_df.copy()
            tmp_drop = tmp_drop.drop(columns='coords')
            dup_idx = np.where(tmp_drop.duplicated(keep="first")) ### keep == True ==> means ONLY tells you subseqent duplicates!!!
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[dup_idx])
            
            
            
            """ Set next frame to be current frame """
            ### for debug
            tmp_cur = np.copy(next_seg)
            plot_max(next_seg, ax=-1)
            
            input_im = next_input
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 255    ### WAS WRONGLY SET TO 0 BEFORE!!!
            truth_cur_im = truth_next_im
            
            

            
            
            
            
    """ DONT DO SIZE ELIM WITHIN LOOP, DO AS POST-PROCESSING INSTEAD???
    
            - this way removes entire track if there's even one part that's too small
    """
            
            
            
            
    """ POST-PROCESSING """
            
    """ Parse the old array and SAVE IT: """
    print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
    
    print('double_linked throughout analysis: ' + str(double_linked))
    #print('num_blobs: ' + str(num_blobs))
    #print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
    #tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]

    
    ### (1) unsure that all of 'RED' or 'YELLOW' are indicated as such
    ### ***should be fine, just turn all "BLANK" into "GREEN"  
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'BLANK'] = 'GREEN'
    
    num_YELLOW = 0; num_RED = 0 
    for cell_num in np.unique(tracked_cells_df.SERIES):
       
        color_arr = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR)
        
        if np.any(color_arr == 'RED'):
            tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'RED'
            num_RED += 1
            
        elif np.any(color_arr == 'YELLOW'):
            tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'YELLOW'
            num_YELLOW += 1
        
    
    
    """ Pre-save everything """
    tracked_cells_df = tracked_cells_df.sort_values(by=['SERIES', 'FRAME'])
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_RAW.csv', index=False)
    
    tracked_cells_df.to_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    #tracked_cells_df = pd.read_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         


               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
               
               #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
               #if track_length_SEG > 0:
               if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
               
               #if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                           singles.append(cell_num)
                           tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
                           #print(cell_num)
                           continue;
                        


    tmp = tracked_cells_df.copy()
    """ (3) ALSO clean up bottom of image so that no new cell can appear in the last 20 stacks
    
                also maybe remove cells on edges as well???
    """
    
    
    num_edges = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
                
        idx = np.where(tracked_cells_df.SERIES == cell_num)[0]
        
        Z_cur_cell = tracked_cells_df.iloc[idx].Z
        
        X_cur_cell = tracked_cells_df.iloc[idx].X
        
        Y_cur_cell = tracked_cells_df.iloc[idx].Y
        
        
        if np.any(Z_cur_cell > lowest_z_depth - 20):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
        
      
        elif np.any(X_cur_cell > width_tmp - 40) or np.any(X_cur_cell < 40):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            num_edges += 1
            
        
        elif np.any(Y_cur_cell > height_tmp - 40) or np.any(Y_cur_cell < 40):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
            num_edges += 1
                

    """ Also remove by min_size """
    num_small = 0;
    min_size = 100;
    for cell_num in np.unique(tracked_cells_df.SERIES):
                
        idx = np.where(tracked_cells_df.SERIES == cell_num)
        for cell_obj in tracked_cells_df.iloc[idx].coords:
            if len(cell_obj) < min_size:  
                tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])   ### DROPS ENTIRE CELL SERIES
                num_small += 1
                break;






    """  Save images in output """
    input_name = examples[0]['input']
    filename = input_name.split('/')[-1]
    filename = filename.split('.')[0:-1]
    filename = '.'.join(filename)
    
    for frame_num, im_dict in enumerate(examples):
         
          output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
          im = convert_matrix_to_multipage_tiff(output_frame)
          imsave(sav_dir + filename + '_' + str(frame_num) + '_output_CLEANED.tif', im)
         
         
          output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
          im = convert_matrix_to_multipage_tiff(output_frame)
          imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)


          output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=0)
          im = convert_matrix_to_multipage_tiff(output_frame)
          imsave(sav_dir + filename + '_' + str(frame_num) + '_output_TERMINATED.tif', im)


          output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=1)
          im = convert_matrix_to_multipage_tiff(output_frame)
          imsave(sav_dir + filename + '_' + str(frame_num) + '_output_NEW.tif', im)
         

          """ Also save image with different colors for RED/YELLOW and GREEN"""
         
     ### (3) drop other columns
    tracked_cells_df = tracked_cells_df.drop(columns=['visited', 'coords'])
    
    
    ### and reorder columns
    cols =  ['SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z']
    tracked_cells_df = tracked_cells_df[cols]

    ### (4) save cleaned
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)               
            
            
            
            
            
           
            
            
    """ If want to reload quickly without running full analysis """            
    #tracked_cells_df = pd.read_csv(sav_dir + 'tracked_cells_df_RAW.csv', sep=',')           
    #truth_output_df = pd.read_csv(sav_dir + 'truth_output_df.csv', sep=',')            
    #truth_array = pd.read_csv(sav_dir + 'truth_array.csv', sep=',')               
    
    ### ALSO NEED LIST_EXCLUDE???      
            
            
            
            
    """ Plot compare to truth """
            


    if truth:
        truth_array.to_csv(sav_dir + 'truth_array.csv', index=False)
        truth_output_df = truth_output_df.sort_values(by=['SERIES'])
        truth_output_df.to_csv(sav_dir + 'truth_output_df.csv', index=False)
     
            
        """ Compare dataframes to see of the tracked cells, how well they were tracked """    
        itera = 0
        if truth:
             all_lengths = []
             truth_lengths = []
             output_lengths = []
             for cell_num in np.unique(truth_output_df.SERIES):
                   
                  
                   
                   ### EXCLUDE SEG_ERRORS
                   if not np.any( np.in1d(list_exclude, cell_num)):
                   #if not np.any( np.in1d(list_exclude, cell_num)) and np.any( np.in1d(all_cell_nums, cell_num)):
                       #track_length_SEG =  len(np.where(truth_output_df.SERIES == cell_num)[0])
                        #track_length_TRUTH = len(np.where(truth_array.SERIES == cell_num)[0])
         
         
                        track_length_TRUTH  = len(np.unique(truth_array.iloc[np.where(truth_array.SERIES == cell_num)].FRAME))
                        track_length_SEG = len(np.unique(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME))         
         
         
                        """ remove anything that's only tracked for length of 1 timeframe """
                        """ excluding if that timeframe is the very first one """
                        
                        #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
                        if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                            continue;
    
         
                        all_lengths.append(track_length_TRUTH - track_length_SEG)
                        
                        truth_lengths.append(track_length_TRUTH)
                        output_lengths.append(track_length_SEG)
                        
                        
                        if track_length_TRUTH - track_length_SEG > 0 or track_length_TRUTH - track_length_SEG < 0:
                             #print(truth_array.FRAME[truth_array.SERIES == cell_num])
                             #print("truth is: " + str(np.asarray(truth_array.iloc[np.where(truth_array.SERIES == cell_num)].FRAME)))
                             #print("output is: " + str(np.asarray(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME)))
                             
                             itera += 1
                             
                             #if len(np.asarray(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME)) == 0:
                             #           zzz
                        
                        
    
                        
        #plt.figure(); plt.plot(all_lengths)
        print(len(all_lengths))
        print(len(np.where(np.asarray(all_lengths) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths) < 0)[0]))
        #truth_output_df = truth_output_df.sort_values(by=['SERIES'])
    
        """ Sort by errors """
        # sorted_lens = np.sort(all_lengths)
        # plt.figure(); 
        # plt.plot(sorted_lens)
        
        
        ### Figure out proportions
        total = len(all_lengths)
        prop = []; track_diff = [];
        uniques = np.unique(all_lengths)
        for num in uniques:
            len_num = len(np.where(all_lengths == num)[0])
            
            if len(prop) == 0:
                prop.append(0)
                prop.append(len_num/total)
                
                
            else:
                prop.append(prop[-1])
                prop.append(len_num/total + prop[-2])
            
            track_diff.append(num)
            track_diff.append(num)
            
        plt.figure(); plt.plot(prop, track_diff)
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)
        ax.margins(x=0)
        ax.margins(y=0.02)
        
        plt.xlabel("proportion of tracks", fontsize=14)
        plt.ylabel("track difference (# frames)", fontsize=14)
        


        """ Load old .csv FROM MATLAB OUTPUT and plot it??? 
        """
        #MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO_MATLAB.csv'

        #MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'; ### CUPRIZONE
        #MATLAB_name = 'output.csv'
        
        MATLAB_name = '680_MATLAB_output.csv'
        
        
        MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
        
        
        all_cells_MATLAB = np.unique(MATLAB_auto_array.SERIES)
        all_cells_TRUTH = np.unique(truth_array.SERIES)
    
    
        all_lengths_MATLAB = []
        truth_lengths = []    
        MATLAB_lengths = []
        
        
        all_cell_nums = []
        for frame_num in range(len(examples)):
             print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
           
             seg_name = examples[frame_num]['seg']  
             seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
             seg = seg[0:lowest_z_depth, ...]
             seg = np.moveaxis(seg, 0, -1)       
             
             
        
             truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                                       input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                              
             
             MATLAB_next_im, MATLAB_auto_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=MATLAB_name, 
                                       input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp,
                                       depth_tmp=depth_total, scale=0, swap=1)
             
             
             """ region props on seg ==> then loop through each individual cell to find out which numbers are matched
             
                  then delete those numbers from the "all_cells_MATLAB" and "all_cells_TRUTH" matrices
                  
                  while finding out the lengths
             """
             label_seg = measure.label(seg)
             cc_seg = measure.regionprops(label_seg)
             
             for cell in cc_seg:
                  coords = cell['coords']
                  
                  
                  if np.any(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0) and np.any(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0):
                       
                       num_truth = np.unique(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                       if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0:
                            num_new_truth = np.max(np.intersect1d(all_cells_TRUTH, num_truth)) ### only keep cells that haven't been tracked before                   
                            track_length_TRUTH = len(truth_array[truth_array.SERIES == num_new_truth])
                            all_cells_TRUTH = all_cells_TRUTH[all_cells_TRUTH != num_new_truth]   # remove this cell so cant be retracked
                            
                            
                       """ get cell from MATLAB """
                       num_MATLAB = np.unique(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])   
                       if len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                            num_new_MATLAB = np.max(np.intersect1d(all_cells_MATLAB, num_MATLAB)) ### only keep cells that haven't been tracked before
                            track_length_MATLAB = len(truth_array[MATLAB_auto_array.SERIES == num_new_MATLAB])
                            all_cells_MATLAB = all_cells_MATLAB[all_cells_MATLAB != num_new_MATLAB]   # remove this cell so cant be retracked
                            
                       
                       if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0 and len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                       
                            all_lengths_MATLAB.append(track_length_TRUTH - track_length_MATLAB)
                            truth_lengths.append(track_length_TRUTH)
                            MATLAB_lengths.append(track_length_MATLAB)   
                            
                            all_cell_nums.append(num_new_truth)
                       
                       
        #plt.figure(); plt.plot(all_lengths)
        print(len(np.where(np.asarray(all_lengths_MATLAB) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths_MATLAB) < 0)[0]))         
                                       
        """ Sort by errors """
        #sorted_lens_CNN = np.sort(all_lengths)
        #plt.figure(); 
        #plt.plot(sorted_lens_CNN)
        

        ### Figure out proportions
        total = len(all_lengths_MATLAB)
        prop = []; track_diff = [];
        uniques = np.unique(all_lengths_MATLAB)
        for num in uniques:
            len_num = len(np.where(all_lengths_MATLAB == num)[0])
            
            if len(prop) == 0:
                prop.append(0)
                prop.append(len_num/total)
                
                
            else:
                prop.append(prop[-1])
                prop.append(len_num/total + prop[-2])
            
            track_diff.append(num)
            track_diff.append(num)
            
        plt.plot(prop, track_diff)
        plt.savefig(sav_dir + 'plot.png')

        
        """ Parse the old array: """
        print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
        MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
        
        #print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
        #MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]
        
                   #duplicates: 378  
        
    
         
        """ **** SEE ONLY THE CELLS THAT WERE TRACKED IN MATLAB, TRUTH, and CNN!!! """
        """ Things to fix still:
    
             (2) ***blobs
             
             
             does using del [] on double tracked cells do anything bad???
        
            ***FINAL OUTPUT:
                    - want to show on tracked graph:
                            (a) cells tracked over time, organize by mistakes highest at top horizontal bar graph ==> also only used cells matched across all 3 matrices
                            (b) show number of cells tracked by each method
                            (c) show # of double_linked to be resolved
                            
        
        
        """
        
        
        
        """ Scale x-axis so is % of tracked successfully """
        
    


    """ Plot """
        
        
    def plot_timeframes(tracked_cells_df, add_name='OUTPUT_'):
        new_cells_per_frame =  np.zeros(len(np.unique(tracked_cells_df.FRAME)))
        terminated_cells_per_frame =  np.zeros(len(np.unique(tracked_cells_df.FRAME)))
        num_total_cells_per_frame = np.zeros(len(np.unique(tracked_cells_df.FRAME)))
        for cell_num in np.unique(tracked_cells_df.SERIES):
            
            frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
            
            beginning_frame = np.min(frames_cur_cell)
            if beginning_frame > 0:   # skip the first frame
                new_cells_per_frame[beginning_frame] += 1

                        
            term_frame = np.max(frames_cur_cell)
            if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
                terminated_cells_per_frame[term_frame] += 1
            
            for num in frames_cur_cell:
                num_total_cells_per_frame[num] += 1    
            
            
            


        y_pos = np.unique(tracked_cells_df.FRAME)
        plt.figure(); plt.bar(y_pos, new_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'new cells per frame'
        #plt.title(name);
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# new cells', fontsize=16)
        # ax.set_xticklabels(x_ticks, rotation=0, fontsize=12)
        # ax.set_yticklabels(y_ticks, rotation=0, fontsize=12)
        plt.savefig(sav_dir + add_name + name + '.png')

        plt.figure(); plt.bar(y_pos, terminated_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'terminated cells per frame'
        #plt.title(name)
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# terminated cells', fontsize=16)
        plt.savefig(sav_dir + add_name + name + '.png')

        
        plt.figure(); plt.bar(y_pos, num_total_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'number cells per frame'
        #plt.title(name)
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# cells', fontsize=16)
        plt.savefig(sav_dir + add_name + name + '.png')



        """ Normalize to proportions like Cody did
        
        """
        new_cells_per_frame
        terminated_cells_per_frame
        num_total_cells_per_frame
        
        
        baseline = num_total_cells_per_frame[0]
        
        norm_tots = num_total_cells_per_frame/baseline
        norm_new = new_cells_per_frame/baseline

        width = 0.35       # the width of the bars: can also be len(x) sequence
        plt.figure()
        p1 = plt.bar(y_pos, norm_tots, yerr=0, color='k')
        p2 = plt.bar(y_pos, norm_new, bottom=norm_tots, yerr=0, color='g')
        
        line = np.arange(-5, len(y_pos) + 5, 1)
        plt.plot(line, np.ones(len(line)), 'r--', linewidth=2, markersize=10)
        
        plt.ylabel('Proportion of cells', fontsize=16)
        plt.xlabel('weeks', fontsize=16); 
        plt.xticks(np.arange(0, len(y_pos), 1))
        plt.xlim(-1, len(y_pos))
        plt.ylim(0, 1.4)
        plt.yticks(np.arange(0, 1.4, 0.2))
        plt.legend((p1[0], p2[0]), ('Baseline', 'New cells'))
        
        
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'normalized recovery'
        plt.savefig(sav_dir + add_name + name + '.png')

            
            

    """ plot timeframes """
    plot_timeframes(tracked_cells_df, add_name='OUTPUT_')
    plot_timeframes(MATLAB_auto_array, add_name='MATLAB_')
    plot_timeframes(truth_array, add_name='TRUTH_')
    
    
    
    
    
    

    
    
    """ 
        Also do density analysis of where new cells pop-up???
    
    """
    neighbors = 10
    
    #for neighbors in range(4, 50, 2):
    
    new_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)))]
    terminated_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)))]
    
    for cell_num in np.unique(tracked_cells_df.SERIES):
        
        frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
        
        beginning_frame = np.min(frames_cur_cell)
        if beginning_frame > 0:   # skip the first frame
            new_cells_per_frame[beginning_frame].append(cell_num)
                    
        term_frame = np.max(frames_cur_cell)
        if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
            terminated_cells_per_frame[term_frame].append(cell_num)
        

    ### loop through each frame and all the new cells and find "i.... i + n" nearest neighbors
    scale_xy = 0.83; scale_z = 3
    

    

    for frame_num, cells in enumerate(new_cells_per_frame):
        new_dists = []
        new_z = []
        total_dists = []
        total_z = []
        
        ### get list of all cell locations on current frame
        all_x = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].X
        all_y =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Y
        all_z =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Z
        
        all_series_num = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].SERIES
        
        all_centroids = [np.asarray(all_x) * scale_xy, np.asarray(all_y) * scale_xy, np.asarray(all_z) * scale_z]
        all_centroids = np.transpose(all_centroids)
        
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(all_centroids)
        distances, indices = nbrs.kneighbors(all_centroids)        
        
        
        
        """ Get all distances to see distribution comparing DEPTH and density/distance """
        for obj in distances:
            cur_dist = obj[1:-1]
            mean = np.mean(cur_dist)
            #total_dists.append(cur_dist)
            total_dists = np.concatenate((total_dists, [mean]))
            
        total_z = np.concatenate((total_z, all_centroids[:, -1]))
        
        
        
        """ Go cell by cell through NEW cells only """
        for cur_cell in cells:
            
            dist_idx = np.where(all_series_num == cur_cell)
            cur_dists = distances[dist_idx][0][1:-1]
            mean_dist = np.mean(cur_dists)
            new_dists = np.concatenate((new_dists, [mean_dist]))
            
            ### compare it with all the other cells at the same depth that are NOT new
            cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
            
            
            new_z = np.concatenate((new_z, [cur_z]))
            
            if cur_z < 200 and cur_z > 150:
                loc_cells_on_cur_z = np.where((all_centroids[:, -1] < cur_z + 5 * scale_z) & (all_centroids[:, -1] > cur_z - 5 * scale_z))[0]
                
                ### exclude cells that are NEW (unlikely)
                ###
                ###
                ###
                ### then find distribution of densities and see where current density lies
                all_dists = []
                for cell_z in loc_cells_on_cur_z:
                    #dist_idx = np.where(all_series_num == cur_cell)
                    cur_dists = distances[cell_z][1:-1]
                    mean_dist_z = np.mean(cur_dists)       
                    all_dists.append(mean_dist_z)

            
            
        plt.figure(neighbors + frame_num); 
        plt.scatter(total_z, total_dists, s=5, marker='o');
        plt.scatter(new_z, new_dists, s=10, marker='o');
        plt.xlabel('depth (um)'); plt.ylabel('density (mean distance 10 nn um - smaller = dense)')
        plt.title('num neighbors: ' + str(neighbors))    
        plt.savefig(sav_dir + 'DENSITY_new_' + str(neighbors + frame_num) + '.png')
    
    
    """ Also get cells that die
    """
    """ Go cell by cell through NEW cells only """

    for frame_num, cells in enumerate(terminated_cells_per_frame):   
        term_z  = [];
        term_dists = [];
        total_dists = [];
        total_z = [];
        
        
        
        all_x = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].X
        all_y =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Y
        all_z =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Z
        
        all_series_num = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].SERIES
        
        all_centroids = [np.asarray(all_x) * scale_xy, np.asarray(all_y) * scale_xy, np.asarray(all_z) * scale_z]
        all_centroids = np.transpose(all_centroids)
        
        
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(all_centroids)
        distances, indices = nbrs.kneighbors(all_centroids)      
        
        
        """ Get all distances to see distribution comparing DEPTH and density/distance """
        for obj in distances:
            cur_dist = obj[1:-1]
            mean = np.mean(cur_dist)
            #total_dists.append(cur_dist)
            total_dists = np.concatenate((total_dists, [mean]))
            
        total_z = np.concatenate((total_z, all_centroids[:, -1]))            

        for cur_cell in cells:
            
            dist_idx = np.where(all_series_num == cur_cell)
            cur_dists = distances[dist_idx][0][1:-1]
            mean_dist = np.mean(cur_dists)
            term_dists = np.concatenate((term_dists, [mean_dist]))
            
            ### compare it with all the other cells at the same depth that are NOT new
            cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
            
            
            term_z = np.concatenate((term_z, [cur_z]))
                

        plt.figure(neighbors * 100 + frame_num); 
        plt.scatter(total_z, total_dists, s=5, marker='o');            
        plt.scatter(term_z, term_dists, s=10, marker='o', color='k');
        plt.xlabel('depth (um)'); plt.ylabel('density (mean distance 10 nn um - smaller = dense)')
        plt.title('num neighbors: ' + str(neighbors))
        plt.xlim(0, 350)
        plt.ylim(50, 200)
        
        plt.savefig(sav_dir + 'DENSITY_term_' + str(neighbors * 100 + frame_num)  + '.png')
    
    

    """ Also get cell size vs. depth
    """
    """ Go cell by cell through NEW cells only """
    plt.close('all')
    for frame_num, cells in enumerate(terminated_cells_per_frame):   
        total_vols = []
        total_z = []
        
        all_coords = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].coords)

        term_z  = [];
        term_vol = [];
        
        all_z =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Z
        all_series_num = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].SERIES
        

        ### CURRENTLY UNSCALED
        
        
        """ Get all distances to see distribution comparing DEPTH and density/distance """
        for obj in all_coords:
            cur_vol = len(obj)

            total_vols = np.concatenate((total_vols, [cur_vol]))
            
        total_z = np.concatenate((total_z, np.asarray(all_z) * scale_z))            


        """ Get terminated cells """
        for cur_cell in cells:
            
            dist_idx = np.where(all_series_num == cur_cell)
            cur_vol = len(all_coords[dist_idx][0])
            

            term_vol = np.concatenate((term_vol, [cur_vol]))
            
            ### compare it with all the other cells at the same depth that are NOT new
            cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
            
            
            term_z = np.concatenate((term_z, [cur_z]))


        plt.figure(neighbors + frame_num); 
        plt.scatter(total_z, total_vols, s=5, marker='o');            
        plt.scatter(term_z, term_vol, s=10, marker='o', color='k');
        plt.xlabel('depth (um)'); plt.ylabel('size')
        plt.title('num neighbors: ' + str(neighbors))
        #plt.xlim(0, 350)
        plt.ylim(0, 4000)   
        plt.savefig(sav_dir + 'SIZE_new_' + str(neighbors + frame_num) + '.png')


    ### NEW CELL
    for frame_num, cells in enumerate(new_cells_per_frame):   
        total_vols = []
        total_z = []
        
        all_coords = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].coords)

        new_z  = [];
        new_vol = [];
        
        all_z =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Z
        all_series_num = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].SERIES
        

        ### CURRENTLY UNSCALED
        
        
        """ Get all distances to see distribution comparing DEPTH and density/distance """
        for obj in all_coords:
            cur_vol = len(obj)

            total_vols = np.concatenate((total_vols, [cur_vol]))
            
        total_z = np.concatenate((total_z, np.asarray(all_z) * scale_z))            


        """ Get terminated cells """
        for cur_cell in cells:
            
            dist_idx = np.where(all_series_num == cur_cell)
            cur_vol = len(all_coords[dist_idx][0])
            

            new_vol = np.concatenate((new_vol, [cur_vol]))
            
            ### compare it with all the other cells at the same depth that are NOT new
            cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
            
            
            new_z = np.concatenate((new_z, [cur_z]))


        plt.figure(neighbors * 100 + frame_num); 
        plt.scatter(total_z, total_vols, s=5, marker='o');            
        plt.scatter(new_z, new_vol, s=10, marker='o');
        plt.xlabel('depth (um)'); plt.ylabel('size')
        plt.title('frame num: ' + str(frame_num))
        #plt.xlim(0, 350)

        plt.ylim(0, 4000)   
        
        plt.savefig(sav_dir + 'SIZE_term_' + str(neighbors * 100 + frame_num) + '.png')
    
    """ 
        Also split by depths
    """
    
    
    
    
    
    
    """
        Test each new cell and see what's wrong
    
    """
    
    
    
    
    """
        Test a good looking cell and introduce noise:
                - degrees of warping
                - degrees of noise
                - degrees of intensity changes
                
                - random translations of the second image???
                
        ALSO TRAIN NEURAL NETWORK WITH THESE PERTURBATIONS???
            - translate the second image randomly???
        
    
    """
        
        
        
        
        
        
        
        
    
    
    
    
    
    