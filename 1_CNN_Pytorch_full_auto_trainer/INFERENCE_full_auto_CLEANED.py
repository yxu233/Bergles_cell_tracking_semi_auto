# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger


To run:
    
    (1) press run
    (2) select folder where the OUTPUT of the segmentation-CNN in the previous step is located
    
    (3) change the "lowest_z_depth" variable as needed (to save computational time)
            - i.e. if you want to go down to 100 slices, then select 100 + 2 == 120 for lowest_z_depth
                b/c the last 20 z-slices are discarded to account for possible large shifts in tissue movement
                so always segment 20 slices more than you actually care about
    
    



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

s_path = './(7) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_SEG_skipped/'; next_bool = 0;
#s_path = './(8) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_YES_NEXT_SEG/'; next_bool = 1;


s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT/'; next_bool = 0;


lowest_z_depth = 180;

crop_size = 160
z_size = 32
num_truth_class = 2
min_size = 10
both = 0




scale_for_animation = 0


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
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_seg_CLEANED_10_125762'


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
    
    if scale_for_animation:
        copy_input_im = np.copy(input_im)
    
    
    
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
    
    animator_iterator = 0;
    for frame_num in range(1, len(examples)):
         print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
         all_dup_indices = [];
        
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[frame_num]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            #next_input = np.moveaxis(next_input[0:lowest_z_depth, ...], 0, -1)
            next_input = np.moveaxis(next_input, 0, -1)
            if scale_for_animation:
                copy_next_input = np.copy(next_input)
            
            
            seg_name = examples[frame_num]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
            next_seg = np.moveaxis(next_seg, 0, -1)


            """ Plot for animation """
            if scale_for_animation:
                track_cur_seg = np.zeros(np.shape(next_seg))
                track_new_seg = np.zeros(np.shape(next_seg))
                track_term_seg = np.zeros(np.shape(next_seg))
                plot_next = np.zeros(np.shape(next_seg))
                
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
           
            """ Iterate through all cells """
            iterator = 0;            
            
            cell_size = []
            
            for cell_idx in progressbar.progressbar(np.where(tracked_cells_df.visited == 0)[0], max_value=len(np.where(tracked_cells_df.visited == 0)[0]), redirect_stdout=True): 
                 
                 cell = tracked_cells_df.iloc[cell_idx]
                 
                 ### go to unvisited cells
                 x = cell.X; y = cell.Y; z = cell.Z;

                 ### SO DON'T VISIT AGAIN
                 tracked_cells_df.visited[cell_idx] = 1
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 cell_size.append(len(cell.coords))
                 if len(cell.coords) < min_size:
                           continue;
                 
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


                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!

                         *** NEED TO FIX THIS TOO!!!
                 
                 """
                 cc_seg_train, seg_train, crop_next_seg = select_one_from_excess(seg_train, crop_next_seg)
                 
                 """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                 """
                 new = 0
                 if len(cc_seg_train) > 0:
                      next_coords = cc_seg_train[0].coords
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
                           #print('double_linked'); 
                           double_linked += 1                    
                           tracked_cells_df, dup_series = sort_double_linked(tracked_cells_df, next_centroid, frame_num)                           
                           all_dup_indices = np.concatenate((all_dup_indices, dup_series))

                      """ set current one to be value 2 so in future will know has already been identified """
                      next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
                      new = 1;
                      
                 else:                           
                      ####   DEBUG if not matched
                      len(np.where(crop_seed)[0])         
                      
                        
                 """ Check if TP, TN, FP, FN """
                 if False:
                      TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude = parse_truth(truth_cur_im,  truth_array, truth_output_df, truth_next_im, 
                                                                                                              seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size,
                                                                                                              blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp)
                 """ Plot for animation """
                 if scale_for_animation and (frame_num == 1 or frame_num == 4): 
                      input_name = examples[0]['input']
                      filename = input_name.split('/')[-1]
                      filename = filename.split('.')[0:-1]
                      filename = '.'.join(filename)
                      low_crop = 0.3; high_crop = 0.7; 
                      z_crop_h = 0.6
                      
                      
                      """ Skip if not within middle crop"""
                      #if np.min(cell.coords[:, 0]) < track_cur_seg.shape[0] * low_crop or np.max(cell.coords[:, 0]) > track_cur_seg.shape[0] * high_crop or  np.min(cell.coords[:, 1]) < track_cur_seg.shape[1] * low_crop or np.max(cell.coords[:, 1]) > track_cur_seg.shape[1] * high_crop or np.max(cell.coords[:, 2]) > track_cur_seg.shape[2] * z_crop_h:
                          
                      if cell.X > track_cur_seg.shape[0] * low_crop and cell.X < track_cur_seg.shape[0] * high_crop and cell.Y > track_cur_seg.shape[1] * low_crop and cell.Y < track_cur_seg.shape[1] * high_crop and cell.Z < track_cur_seg.shape[2] * z_crop_h:
                          ### PLOT current and next frame
                          #plot_next = np.copy(next_seg)
                          #plot_next[plot_next != 250] = 0     ### delete everything that hasn't been visited
                          #plot_next[plot_next > 0] = 150      ### MAKE THE BACKGROUND PAST TRACES DIMMER
                          from random import randint
                          rand = randint(1, 6)
                          if new:  ### if a cell was tracked on second frame, use those coords to highlight it brighter than the rest
                               track_new_seg = np.zeros(np.shape(track_cur_seg))
                               track_new_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                              
                               track_new_seg_next = np.zeros(np.shape(track_cur_seg))
                               track_new_seg_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 255       
                              
                              
                               # track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = rand
                               # plot_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = rand    
                               copy_next_input[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 0
                              
                          else:   ### otherwise, if NOT TRACKED, then change the color of the cell on the FIRST FRAME!!!
                              track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 0
                              track_term_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                          
                              
                              
                                                  
                          """ Print out animation for 2nd frame """
                          copy_next_input[plot_next > 0] = 0      ### set old cells to blank so color comes through better!
                          
                        
                          im = convert_matrix_to_multipage_tiff(copy_next_input)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_input.tif',  np.asarray(im * 255, dtype=np.uint8))
                          
                          im = convert_matrix_to_multipage_tiff(plot_next)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_seg.tif',  np.asarray(im * 255, dtype=np.uint8))


                          im = convert_matrix_to_multipage_tiff(track_new_seg_next)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_CUR_CHECK.tif',  np.asarray(im * 255, dtype=np.uint8))
                           
                          

                          """ Print out animation for 1st frame """
                          copy_input_im[track_cur_seg > 0] = 0   ### set old cells to blank so color comes through better!
                          im = convert_matrix_to_multipage_tiff(copy_input_im)
                          
                          ### or just crop it
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) + '_cell_num_' + str(cell_idx) + '_cur_input.tif',  np.asarray(im * 255, dtype=np.uint8))
        
                          im = convert_matrix_to_multipage_tiff(track_cur_seg)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
    
                          im = convert_matrix_to_multipage_tiff(track_new_seg)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_CURRENT_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
    
    
                          im = convert_matrix_to_multipage_tiff(track_term_seg)
                          im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                          im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                          imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_TERM.tif',  np.asarray(im * 255, dtype=np.uint8))
    
    
                          animator_iterator += 1
                          
                          
                          ### add rainbow color index AFTER iterating through current index
                          if new:
                               track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = rand
                               plot_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = rand
                             
                          print('YOOOOOOOOOOO')



            """ POST-PROCESSING on per-frame basis """            
            
            """ #1 == check all cells with tracked predictions """
            tmp = tracked_cells_df.copy()
            tmp_next_seg = np.copy(next_seg)
            
            
            ### DEBUG:
            tracked_cells_df = tmp.copy()
            next_seg = np.copy(tmp_next_seg)
            
            """  Identify all potential errors of tracking (over the predicted distance error threshold) """
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)
            
            """ Keep looping above and start searnchig from LARGEST distance cells to correct """
            concat = np.transpose(np.asarray([check_series, dist_check]))
            sortedArr = concat[concat[:,1].argsort()[::-1]]
            check_sorted_series = sortedArr[:, 0]
            
            ### ^^^this order is NOT being kept right now!!!
             

            """ Check all cells with distance that is going against prediction """
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, check_sorted_series,
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)
            new_candidates = recheck_series
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, new_candidates,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im,
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)            
            """ #2 == Check duplicates """
            all_dup_indices = np.unique(all_dup_indices)
            
            ### only keep what's hasn't been deleted above
            
            #l3 = [x for x in l1 if x not in l2]
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, all_dup_indices,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)

            ### check on recheck_series
            recheck_series = np.unique(recheck_series)
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)
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
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, term_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)          

            ### check on recheck_series
            recheck_series = np.concatenate((check_series, np.unique(recheck_series)))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)  
  

            """ #4 repeat until no more recheck series """
            
            # ### THESE ARE LIKELY CELLS THAT ARE DOUBLED UP!!!
            # while len(recheck_series) > 0:
            #     recheck_series = np.unique(recheck_series)
            #     tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)
            #     print(recheck_series)
                
            
     
    
            

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
            #plot_max(next_seg, ax=-1)
            
            input_im = next_input
            
            if scale_for_animation:
                copy_input_im = np.copy(next_input)
                
            
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 255    ### WAS WRONGLY SET TO 0 BEFORE!!!
            truth_cur_im = truth_next_im
            
            
            
    """ DONT DO SIZE ELIM WITHIN LOOP, DO AS POST-PROCESSING INSTEAD???
    
            - this way removes entire track if there's even one part that's too small
    """
            
    

    """
        Things to do:
            (1) find out which cells being eliminated and why
            (2) fix few duplicates left
            (3) fix XY coordinates
            (4) fix large blobs being segmented together
    
            (5) mark "new" cells to ensure they are "new"
            
            
            (6) looking at density ==> layer 2 maybe partition into different layers?
                - so that layer 2 isn't taking into account cells from lower/upper layers?
    
            (7) ***plot to see how well you can predict next cell location!!! based on surroundings!!!!
                            *** use truth_array???
    
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
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'BLANK'] = 'Green'
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'GREEN'] = 'Green'
    
    num_YELLOW = 0; num_RED = 0; num_new_color = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
       
        color_arr = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR)
        
        cells = tracked_cells_df.loc[tracked_cells_df["SERIES"].isin([cell_num])]
        index = cells.index
        
        if np.any(color_arr == 'RED'):
            tracked_cells_df["COLOR"][index] = 'RED'
            num_RED += 1
            
        elif np.any(color_arr == 'RED'):  # originally YELLOW
            tracked_cells_df["COLOR"][index]= 'RED'
            num_YELLOW += 1
            
        ### Mark all new cells on SINGLE FRAME as 'YELLOW'
        if len(color_arr) == 1:
            tracked_cells_df["COLOR"][index] = 'YELLOW'
            num_new_color += 1
            #print(color_arr)
     

    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'YELLOW'] = 'Yellow' 
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'RED'] = 'Red' 

    

    ### (4) re-name X and Y columns
    tracked_cells_df = tracked_cells_df.rename(columns={'X': 'Y', 'Y': 'X'})


    
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


    tmp = tracked_cells_df.copy()



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


          # output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=0)
          # im = convert_matrix_to_multipage_tiff(output_frame)
          # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_TERMINATED.tif', im)


          # output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=1)
          # im = convert_matrix_to_multipage_tiff(output_frame)
          # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_NEW.tif', im)
         

          """ Also save image with different colors for RED/YELLOW and GREEN"""
         
     ### (3) drop other columns
    tracked_cells_df = tracked_cells_df.drop(columns=['visited', 'coords'])
    
    
    ### and reorder columns
    cols =  ['SERIES', 'COLOR', 'FRAME', 'Y', 'X', 'Z']
    tracked_cells_df = tracked_cells_df[cols]


    ### (5) save cleaned
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)               
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    """ Plot out 3D line plot where each cell is tracked through a line """
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for series in np.unique(tracked_cells_df.SERIES):   
    #     all_cur_series = tracked_cells_df.loc[tracked_cells_df["SERIES"].isin([series])]
    #     all_x = all_cur_series.X
    #     all_y = all_cur_series.Y
    #     all_z = all_cur_series.Z
    #     ax.plot(all_x, all_y, all_z, linewidth=2, linestyle='dashed')
    #     size = all_cur_series.FRAME
    #     #ax.scatter([all_x], [all_y], [all_z], s=  ((np.asarray(size, dtype=np.float32) + 1)/4)**2, marker='o', c=;r;, label='the data')
    #     ax.scatter([all_x], [all_y], [all_z], c=np.asarray(size, dtype=np.float32)/np.max(tracked_cells_df.FRAME), marker='o', s=5, label='the data')
            
               
    """ If want to reload quickly without running full analysis """            
    #tracked_cells_df = pd.read_csv(sav_dir + 'tracked_cells_df_RAW.csv', sep=',')           
    #truth_output_df = pd.read_csv(sav_dir + 'truth_output_df.csv', sep=',')            
    #truth_array = pd.read_csv(sav_dir + 'truth_array.csv', sep=',')               
    
    ### ALSO NEED LIST_EXCLUDE???      


    """ Set plot size and DPI """
    plt.rcParams['figure.figsize'] = [8.0, 6.0]
    plt.rcParams['figure.dpi'] = 140

    



    """ plot timeframes """
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_', depth_lim_lower=0, depth_lim_upper=120)


    """ 
        Also split by depths
    """
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_0-32', depth_lim_lower=0, depth_lim_upper=32, only_one_plot=1)
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_33-65', depth_lim_lower=33, depth_lim_upper=65, only_one_plot=1)
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_66-99', depth_lim_lower=66, depth_lim_upper=99, only_one_plot=1)
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_100-132', depth_lim_lower=100, depth_lim_upper=132, only_one_plot=1)
    plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_133-165', depth_lim_lower=133, depth_lim_upper=165, only_one_plot=1)


    
    if truth:
        #MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO_MATLAB.csv'

        #MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'; ### CUPRIZONE
        #MATLAB_name = 'output.csv'
        
        MATLAB_name = '680_MATLAB_output.csv'  
        MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
        plot_timeframes(MATLAB_auto_array, sav_dir, add_name='MATLAB_', depth_lim_lower=0, depth_lim_upper=120)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_', depth_lim_lower=0, depth_lim_upper=120)


        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_0-32', depth_lim_lower=0, depth_lim_upper=32, only_one_plot=1)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_33-65', depth_lim_lower=33, depth_lim_upper=65, only_one_plot=1)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_66-99', depth_lim_lower=66, depth_lim_upper=99, only_one_plot=1)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_100-132', depth_lim_lower=100, depth_lim_upper=132, only_one_plot=1)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_133-165', depth_lim_lower=133, depth_lim_upper=165, only_one_plot=1)


    

    """ Also plot rate of loss??? """




    """ Show that truth predicitions are accurate! """
    
    
    if truth:
         distances = []
         for frame_num in range(len(examples)):
        
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(truth_array, frame_num, crop_size, z_size, dist_error_thresh=10)
                
            if frame_num == 1:
                plt.figure(); plt.hist(all_dist, color='k')
                ax = plt.gca()
                plt.xlabel('distance of prediction to truth (px)', fontsize=ax_title_size); plt.ylabel('number of cells', fontsize=ax_title_size)
                rs = ax.spines["right"]; rs.set_visible(False)
                ts = ax.spines["top"]; ts.set_visible(False)


                
                plt.savefig(sav_dir + 'prediction_accuracy_truth.png')
            distances = np.concatenate((distances, np.asarray(all_dist)))


         #plt.figure(); plt.hist(distances)
         num_above_10 =  len(np.where(distances > 10)[0])

         num_total = len(distances)
         
         #plt.savefig(sav_dir + 'prediction_accuracy_truth.png')

         print('% cells above 10 pixels: ' + str((num_above_10/num_total)  * 100))


            
    """ Plot compare to truth """
    if truth:

        """ Load .csv from tracked_output
        """
        CNN_name = 'tracked_cells_df_clean.csv'
        all_lengths_CNN = load_and_compare_csvs_to_truth(sav_dir, CNN_name, examples, 
                                                         lowest_z_depth, truth_array, truth_name, truth_path=input_path,
                                                         input_im=input_im, height_tmp=height_tmp, width_tmp=width_tmp, depth_total=depth_total,
                                                         scale=scale)
        
        ### length == TRUTH - test
        ### > 0 ==> TRUTH is longer (undertracked)
        ### < 0 ==> test is longer (overtracked)
        
        errs_CNN_under = len(np.where(np.asarray(all_lengths_CNN) > 0)[0])
        errs_CNN_under_2 = len(np.where(np.asarray(all_lengths_CNN) > 1)[0])
        
        errs_CNN_over = len(np.where(np.asarray(all_lengths_CNN) < 0)[0])
        errs_CNN_over_2 = len(np.where(np.asarray(all_lengths_CNN) < -1)[0])
        
        perc_errs = (errs_CNN_under + errs_CNN_over) / len(all_lengths_CNN) * 100
        perc_errs_over_2 = (errs_CNN_under_2 + errs_CNN_over_2) / len(all_lengths_CNN) * 100
        
        

        ### Figure out proportions
        total = len(all_lengths_CNN)
        prop = []; track_diff = [];
        uniques = np.unique(all_lengths_CNN)
        for num in uniques:
            len_num = len(np.where(all_lengths_CNN == num)[0])
            
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
                
        
        """ Load .csv from MATLAB run and plot it
        """
                  
        all_lengths_MATLAB = load_and_compare_csvs_to_truth(input_path, MATLAB_name, examples, 
                                                            lowest_z_depth, truth_array, truth_name, truth_path=input_path,
                                                            input_im=input_im, height_tmp=height_tmp, width_tmp=width_tmp, depth_total=depth_total,
                                                            scale=scale)
        errs_CNN_under = len(np.where(np.asarray(all_lengths_MATLAB) > 0)[0])
        errs_CNN_under_2 = len(np.where(np.asarray(all_lengths_MATLAB) > 1)[0])
        
        errs_CNN_over = len(np.where(np.asarray(all_lengths_MATLAB) < 0)[0])
        errs_CNN_over_2 = len(np.where(np.asarray(all_lengths_MATLAB) < -1)[0])
        
        perc_errs_MATLAB = (errs_CNN_under + errs_CNN_over) / len(all_lengths_MATLAB) * 100
        perc_errs_over_2_MATLAB = (errs_CNN_under_2 + errs_CNN_over_2) / len(all_lengths_MATLAB) * 100


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
        plt.xlabel("proportion of tracks", fontsize=14)
        plt.ylabel("track difference (# frames)", fontsize=14)
        
        ax.legend(['CNN tracker', 'Heuristic'])
        
        plt.savefig(sav_dir + 'plot.png')

        
        """ Parse the old array: """
        print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
        MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
        



        """ Plot errors  """
        fig = plt.figure()
        ax = plt.gca()
        
        errs = [perc_errs, perc_errs_MATLAB]
        errs_over_2 = [perc_errs_over_2, perc_errs_over_2_MATLAB]
        
        X = np.arange(len(errs))
        ax.bar(X + 0.00, errs, color = 'k', width = 0.25)
        ax.bar(X + 0.25, errs_over_2, color = 'g', width = 0.25)

        ind = np.arange(len(errs))
        width = 0.25
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('CNN tracker', 'Heuristic'))
        ax.legend(['All errors', 'errors > 1 frame'])

        #plt.xlabel("proportion of tracks", fontsize=14)
        plt.ylabel("% cells tracked with errors", fontsize=14)
        plt.yticks(np.arange(0, max(errs)+1, 5))
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)
        plt.savefig(sav_dir + 'cell_tracking_errors' + '.png')



    """ SCALE CELL COORDS??? 
    """
    scale_xy = 0.83
    scale_z = 3
    tmp = np.zeros(np.shape(input_im))
    tracked_cells_df['vol_rescaled'] = np.nan
    print('scaling cell coords')
    for idx in range(len(tracked_cells_df)):
        
        cell = tracked_cells_df.iloc[idx]
        
        coords = cell.coords   
        tmp[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        
        crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(tmp, cell.X, cell.Y, cell.Z, 50/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
   
        crop_rescale = resize(crop, (crop.shape[0] * scale_xy, crop.shape[1] * scale_xy, crop.shape[2] * scale_z), order=0, anti_aliasing=True)
        
        label = measure.label(crop_rescale)       
        cc = measure.regionprops(label)
        new_coords = cc[0]['coords']
        tracked_cells_df.iloc[idx, tracked_cells_df.columns.get_loc('vol_rescaled')] = len(new_coords)
   
        tmp[tmp > 0] = 0  # reset
        
        # import napari
        # with napari.gui_qt():
        #     viewer = napari.view_image(crop)
            
            




    
    """ 
        Also do density analysis of where new cells pop-up???
    
    """
    analyze = 1;
    
    ax_title_size = 16
    
    if analyze == 1:
        tracked_cells_df = tmp.copy()
        
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
            ax = plt.gca()
            plt.scatter(total_z, total_dists, s=5, marker='o');
            plt.scatter(new_z, new_dists, s=10, marker='o');
            plt.xlabel('depth (um)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num)) 
            rs = ax.spines["right"]; rs.set_visible(False)
            ts = ax.spines["top"]; ts.set_visible(False)
            plt.xlim(0, 500)
            plt.ylim(30, 200)
            

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
            ax = plt.gca()
            plt.scatter(total_z, total_dists, s=5, marker='o');            
            plt.scatter(term_z, term_dists, s=10, marker='o', color='k');
            plt.xlabel('depth (um)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=ax_title_size)
            rs = ax.spines["right"]; rs.set_visible(False)
            ts = ax.spines["top"]; ts.set_visible(False)
            plt.xlim(0, 500)
            plt.ylim(30, 200)
            
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
            ax = plt.gca()
            plt.scatter(total_z, total_vols, s=5, marker='o');            
            plt.scatter(term_z, term_vol, s=10, marker='o', color='k');
            plt.xlabel('depth (um)', fontsize=ax_title_size); plt.ylabel('size', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=ax_title_size)
            #plt.xlim(0, 350)
            rs = ax.spines["right"]; rs.set_visible(False)
            ts = ax.spines["top"]; ts.set_visible(False)
            
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
    
    
            """ Get new cells """
            for cur_cell in cells:
                
                dist_idx = np.where(all_series_num == cur_cell)
                cur_vol = len(all_coords[dist_idx][0])
                
    
                new_vol = np.concatenate((new_vol, [cur_vol]))
                
                ### compare it with all the other cells at the same depth that are NOT new
                cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
                
                
                new_z = np.concatenate((new_z, [cur_z]))
    
    
            plt.figure(neighbors * 100 + frame_num); 
            ax = plt.gca()
            plt.scatter(total_z, total_vols, s=5, marker='o');            
            plt.scatter(new_z, new_vol, s=10, marker='o');
            plt.xlabel('depth (um)', fontsize=ax_title_size); plt.ylabel('size', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=ax_title_size)
            #plt.xlim(0, 350)
            rs = ax.spines["right"]; rs.set_visible(False)
            ts = ax.spines["top"]; ts.set_visible(False)    
            plt.ylim(0, 4000)   
            
            plt.savefig(sav_dir + 'SIZE_term_' + str(neighbors * 100 + frame_num) + '.png')
            
            
            
            
            
            
        """ 
            Plot size decay
        """
            
        def get_sizes_and_z_cur_frame(tracked_cells_df, frame, use_scaled=0):      
            all_sizes_cur_frame = []
            all_z = []
            for cell_num in np.unique(tracked_cells_df.SERIES):
                
                frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
                track_coord = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].coords
                
                track_z = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z
                
                
                beginning_frame = np.min(frames_cur_cell)
                if beginning_frame == frame:   # skip during cuprizone treatment
                    cur_sizes = [] 
                    
                    if not use_scaled:
                        for cc in track_coord:
                                cur_sizes.append(len(cc))

                    else:
                        cur_sizes = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].vol_rescaled)
                    
                            
                            
                    all_sizes_cur_frame.append(cur_sizes)
                    
                    
                    cur_zs = []                
                    for cc in track_z:
                        cur_zs.append(cc)
                    all_z.append(cur_zs)
                    
            return all_sizes_cur_frame, all_z
            
        
        
        
        for frame in range(len(np.unique(tracked_cells_df.FRAME))):
            all_sizes_cur_frame, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame, use_scaled=1)
                    
            plt.figure();
            for idx in range(len(all_sizes_cur_frame)):
                
                size = all_sizes_cur_frame[idx]
                z = all_z[idx]
                if len(size) == len(np.unique(tracked_cells_df.FRAME)) - frame:
                    plt.plot(z, size, linewidth=1)
                    plt.ylim(0, 10000)   
                
                
                



        


                
        """ Plot scatters of each type:
                
                - control/baseline day 1 ==> frame 0
                        ***cuprizone ==> frame 4
                - 1 week after cupr
                - 2 weeks after cupr
                - 3 weeks after cupr 
            
            """
        import seaborn as sns
        
        plt.figure()
        
        ### (0) baseline cells
        use_scaled = 1
        
        
        all_sizes_frame_0, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame=0, use_scaled=use_scaled)
        first_frame_sizes = []
        first_z = []
        for idx, size in enumerate(all_sizes_frame_0):
            first_frame_sizes.append(size[0])
            first_z.append(all_z[idx][0])
            
        ### (1) sizes of all cells that are within 1 week old and also within 3 weeks of recovery
        
        for frame in range(4, 7):
            all_sizes_frame_0, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame=frame, use_scaled=use_scaled)
            first_frame_1_week = []; z_1_week = [];
            first_frame_2_week = []; z_2_week = [];
            first_frame_3_week = []; z_3_week = [];
            for idx, size in enumerate(all_sizes_frame_0):
                first_frame_1_week.append(size[0])
                z_1_week.append(all_z[idx][0])
            
                if len(size) > 1:
                    first_frame_2_week.append(size[1])
                    z_2_week.append(all_z[idx][1])
                    
                if len(size) > 2:
                    first_frame_3_week.append(size[2])            
                    z_3_week.append(all_z[idx][2])
 
        
        data = {'Baseline':first_frame_sizes, '1 week':first_frame_1_week, '2 week': first_frame_2_week, '3 week': first_frame_3_week}
        #df = pd.DataFrame(data=data)
        
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        
        ax = sns.violinplot(data=df)
        
        plt.ylabel('Cell size', fontsize=16)
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)   
        plt.savefig(sav_dir + 'Cell sizes changing.png')
    
        ### plot with z as well
        plt.figure();
        plt.scatter(first_z, first_frame_sizes, color='b')
        plt.scatter(z_1_week, first_frame_1_week, color='orange')
        plt.scatter(z_2_week, first_frame_2_week, color='green')
        plt.scatter(z_3_week, first_frame_3_week, color='red')
        
        
            
        
        """ SCALE Z-DIMENSION FOR APPROPRIATE VOLUME ANALYSIS???
            
        
        ???????????????????????????????????????????????????????????????????????????????????????????????????????????
                multiply each dimension by corresponding scale factor
                    then round and fill in all gaps to determine # of voxels!
        
        
        """
        
        
                
        """ Predict age based on size??? 
        
                what is probability that cell is 1 week old P(B) given that it is size X P(A) == P(B|A) == P(A and B) / P(A)
                
                
                P(A) == prob cell is ABOVE size X
                P(B) == prob cell 1 week old
                
                P(A and B) == prob cell is at least 1 week old AND above size X
                
                
        """
        """ DOUBLE CHECK THIS PROBABILITY CALCULATION!!!"""
        def find_prob_given_size(cur_frame):
            all_probs = []; all_sizes = []
            #for size_thresh in range(500, 4100, 100):
            for size_thresh in range(2000, 8000, 100):   ### FOR RESCALED VOLUMES
                all_cell_sizes = np.concatenate((first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week))
                num_A = 0
                total_num = len(all_cell_sizes)
                for sizes in all_cell_sizes:
                    
                    if sizes > size_thresh:
                        num_A += 1

                P_A = num_A/total_num;
        
                ### find P(B)
                num_1_week_old = len(first_frame_1_week)
                P_B = num_1_week_old/total_num
                                
                ### find P(A and B) *** NOT equal to P(A) * P(B) because are INDEPENDENT events!!!
                ### so only way to find is to count # that are BOTH 1 week young AND > size thresh / num total
                num_above_thresh_1_week = len(np.where(np.asarray(cur_frame) > size_thresh)[0])
                ### SHOULD BE DIVIDED BY TOTAL # OF first frame cells??? i.e. total number of cells, not occurences
                #len(np.unique(tracked_cells_df.SERIES))
                
                P_A_B = num_above_thresh_1_week/total_num;
                #P_A_B = num_above_thresh_1_week/len(np.unique(tracked_cells_df.SERIES));
                
                PB_A = P_A_B/P_A
                
                #print(PB_A)
                
                all_probs.append(PB_A)
                all_sizes.append(size_thresh)
                
            return all_probs, all_sizes
                
                
        all_probs, all_sizes = find_prob_given_size(first_frame_1_week)
        plt.figure(); plt.plot(all_sizes, all_probs)
        ax = plt.gca()
        plt.xlabel('Size threshold', fontsize=ax_title_size); plt.ylabel('Probability', fontsize=ax_title_size)
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)            
                    

        all_probs, all_sizes = find_prob_given_size(first_frame_2_week)
        plt.plot(all_sizes, all_probs)
        
        all_probs, all_sizes = find_prob_given_size(first_frame_3_week)
        plt.plot(all_sizes, all_probs)
        

        all_probs, all_sizes = find_prob_given_size(np.concatenate((first_frame_1_week, first_frame_2_week)))
        plt.plot(all_sizes, all_probs)
        
        all_probs, all_sizes = find_prob_given_size(np.concatenate((first_frame_1_week, first_frame_2_week, first_frame_3_week)))
        plt.plot(all_sizes, all_probs)

        
        plt.legend(['1 week old', '2 weeks old', '3 weeks old', '< 2 weeks old', '< 3 weeks old'])
        
        
        
        plt.savefig(sav_dir + 'probabilities.png')

                
        """ vs. cells in control condition???
        
        
        
        """
                
            
            
       
        
        
        """
            Re-plot stuff from tracker
        """
        tracker = check['tracker']
        #plot_tracker(tracker, sav_dir)
        
        plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32,
                        location='lower right')
        plt.xlabel('Epochs', fontsize=16); plt.ylabel('Jaccard', fontsize=16)
        plt.yticks(np.arange(0.12, 0.35, 0.04))
        plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
        plt.savefig(sav_dir + 'Jaccard.png')
        
           
        plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33,
                        location='upper right')
        
        plt.xlabel('Epochs', fontsize=16); plt.ylabel('Loss', fontsize=16)
        #plt.yticks(np.arange(0.12, 0.35, 0.04))
        plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
        
        plt.figure(33); plt.yscale('log'); plt.savefig(sav_dir + 'loss_per_epoch.png')          
                        

        plot_metric_fun([], tracker.plot_acc, class_name='', metric_name='accuracy', plot_num=29,
                        location='lower right')
        plt.xlabel('Epochs', fontsize=16); plt.ylabel('Accuracy', fontsize=16)
        #plt.yticks(np.arange(0.12, 0.35, 0.04))
        plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))       
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('orange')
        plt.figure(29); plt.savefig(sav_dir + 'Accuracy.png')
        print('Final accuracy: ' + str(tracker.plot_acc[-1]))
        
        
        plot_metric_fun(tracker.plot_sens_val, tracker.plot_sens, class_name='', metric_name='sensitivity', plot_num=30,
                        location='lower right')
        plt.xlabel('Epochs', fontsize=16); plt.ylabel('Sensitivity', fontsize=16)
        plt.yticks(np.arange(0.4, 1.05, 0.1))        
        plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('orange')
        plt.figure(30); plt.savefig(sav_dir + 'Sensitivity.png')
        print('Final sensitivity: ' + str(tracker.plot_sens[-1]))
        
              
        plot_metric_fun(tracker.plot_prec_val, tracker.plot_prec, class_name='', metric_name='precision', plot_num=31,
                        location='lower right')
        plt.xlabel('Epochs', fontsize=16); plt.ylabel('Precision', fontsize=16)
        #plt.yticks(np.arange(0.12, 0.35, 0.04))
        plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))               
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('orange')
        plt.figure(31); plt.savefig(sav_dir + 'Precision.png')
        print('Final precision: ' + str(tracker.plot_prec[-1]))
        
        
        
        """
            Test a good looking cell and introduce noise:
                    - degrees of warping
                    - degrees of noise
                    - degrees of intensity changes
                    
                    - random translations of the second image???
                    
            ALSO TRAIN NEURAL NETWORK WITH THESE PERTURBATIONS???
                - translate the second image randomly???
            
        
        """
            
            
        
        
        
        
        
        
    
    
    
    
    
    