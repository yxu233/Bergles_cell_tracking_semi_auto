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


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
s_path = './(1) Checkpoints_full_auto_no_spatialW/'
#s_path = './(2) Checkpoints_full_auto_spatialW/'

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

    """ Initialize matrix of cells """   
    input_name = examples[0]['input']            
    input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
    depth_total, empty, empty = input_im.shape
    
    input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
    width_tmp, height_tmp, depth_tmp = input_im.shape
    
    seg_name = examples[0]['seg']  
    cur_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
    cur_seg = cur_seg[0:lowest_z_depth, ...]
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
         if int(centroid[2]) + z_size/2 >= lowest_z_depth:
               continue
         
         coords = cell['coords']
         row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': 0, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
         tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
          
    
    """ Get truth from .csv as well """
    truth = 1
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'; scale = 0
         truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv';  scale = 1  # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv'
         #truth_name = '680_syGlass_10x.csv'                           
         
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
       
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[frame_num]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            next_input = np.moveaxis(next_input[0:lowest_z_depth, ...], 0, -1)
     
            seg_name = examples[frame_num]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg = np.moveaxis(next_seg[0:lowest_z_depth, ...], 0, -1)

            
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
           
            """ Iterate through all cells """
            iterator = 0;            
            for cell_idx in np.where(tracked_cells_df.visited == 0)[0]: 
                 
                 cell = tracked_cells_df.iloc[cell_idx]
                 
                 ### go to unvisited cells
                 x = cell.X; y = cell.Y; z = cell.Z;

                 ### SO DON'T VISIT AGAIN
                 tracked_cells_df.visited[cell_idx] = 1
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 if len(cell.coords) < 10:
                      continue;
                 
                 """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
                 if z + z_size/2 >= lowest_z_depth:
                      print('skip'); skipped += 1
                      continue
                 

                 """ Crop and prep data for CNN"""
                 batch_x, crop_next_seg, crop_seed, box_x_min, box_y_min, box_z_min = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
                                                                                                         next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
                                                                                                         height_tmp, width_tmp, depth_tmp)
                 ### Convert to Tensor
                 inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)

                 # forward pass to check validation
                 output_val = unet(inputs_val)

                 """ Convert back to cpu """                                      
                 output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                 seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
                 
                 iterator += 1

                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!"""
                 cc_seg_train, seg_train, crop_next_seg = select_one_from_excess(seg_train, crop_next_seg)
                 
                 """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                 """
                 if len(cc_seg_train) > 0:
                      next_coords = cc_seg_train[0].coords
                      next_coords = scale_coords_of_crop_to_full(next_coords, box_x_min , box_y_min, box_z_min)
                      
                      next_centroid = np.asarray(cc_seg_train[0].centroid)
                      next_centroid[0] = np.int(next_centroid[0] + box_x_min)   # SCALING the ep_center
                      next_centroid[1] = np.int(next_centroid[1] + box_y_min)
                      next_centroid[2] = np.int(next_centroid[2] + box_z_min)

                      ### add to matrix 
                      row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                      tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     
 
                      """ FIND DOUBLES EARLY TO CORRECT AS YOU GO """
                      if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250): ### if this place has already been visited in the past
                           print('double_linked'); double_linked += 1
                        
                           tracked_cells_df = sort_double_linked(tracked_cells_df, next_centroid, frame_num)
                           
                      """ set current one to be value 2 so in future will know has already been identified """
                      next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
                      
                      
                 """ Check if TP, TN, FP, FN """
                 if truth:
                     
                      TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude = parse_truth(truth_cur_im,  truth_array, truth_output_df, truth_next_im, 
                                                                                                              seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size,
                                                                                                              blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp)
                 


                 print('Testing cell: ' + str(iterator) + ' of total: ' + str(len(np.where(tracked_cells_df.visited == 0)[0]))) 



            """ associate remaining cells that are "new" cells and add them to list to check as well as the TRUTH tracker """
            tracked_cells_df, truth_output_df, truth_next_im, truth_array = associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size,
                                                                           truth=truth, truth_output_df=truth_output_df, truth_next_im=truth_next_im, truth_array=truth_array)

                             
                    
            
            """ Set next frame to be current frame """
            input_im = next_input
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 0
            truth_cur_im = truth_next_im
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
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
    #tracked_cells_df = pd.read_csv(sav_dir + 'tracked_cells_df_RAW.csv', sep=',')
    
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         


               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
               
               #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
               if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                   singles.append(cell_num)
                   tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
                   continue;
                        

   

    """  Save images in output """
    input_name = examples[0]['input']
    filename = input_name.split('/')[-1]
    filename = filename.split('.')[0:-1]
    filename = '.'.join(filename)
    
    for frame_num, im_dict in enumerate(examples):
         
         output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
         im = convert_matrix_to_multipage_tiff(output_frame)
         imsave(sav_dir + filename + '_' + str(frame_num) + '_output.tif', im)
         
         
         output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
         im = convert_matrix_to_multipage_tiff(output_frame)
         imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)


         """ Also save image with different colors for RED/YELLOW and GREEN"""
         
     ### (3) drop other columns
    tracked_cells_df = tracked_cells_df.drop(columns=['visited', 'coords'])
    
    
    ### and reorder columns
    cols =  ['SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z']
    tracked_cells_df = tracked_cells_df[cols]

    ### (4) save cleaned
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)               
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    """ Plot compare to truth """
            


    if truth:
        truth_array.to_csv(sav_dir + 'truth_array.csv', index=False)
        truth_output_df = truth_output_df.sort_values(by=['SERIES'])
        truth_output_df.to_csv(sav_dir + 'truth_output_df.csv', index=False)
        #truth_output_df = pd.read_csv(sav_dir + 'truth_output_df.csv', sep=',')            
        #truth_array = pd.read_csv(sav_dir + 'truth_array.csv', sep=',')        
            
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
                        
                        
    
                        
        plt.figure(); plt.plot(all_lengths)
        print(len(all_lengths))
        print(len(np.where(np.asarray(all_lengths) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths) < 0)[0]))
        #truth_output_df = truth_output_df.sort_values(by=['SERIES'])
    
        
    
        """ order the lists before plotting """
        stacked_lens = np.transpose(np.vstack([truth_lengths, output_lengths]))
        sort_lens = stacked_lens[stacked_lens[:, 1].argsort()[::-1]]
        
        truth_lengths = sort_lens[:, 0]
        output_lengths = sort_lens[:, 1]
        
        fig = plt.figure()
        axis = plt.gca()
        y_pos = np.arange(len(all_lengths))
        axis.barh(y_pos, truth_lengths, alpha = 0.5, color = 'b')
        axis.barh(y_pos, output_lengths, alpha = 0.5, color = 'g')
        
        
        plt.figure()
        plt.scatter(truth_lengths, output_lengths)
    
    
        import seaborn as sns
        #sns.set(style="whitegrid")
        #sns.set_color_codes("pastel")
        f, ax = plt.subplots()
        #ax.xaxis.set_ticks(np.arange(0, len(truth_lengths), 100))
        sns.scatterplot(y = truth_lengths, x= output_lengths, color = 'b', x_jitter=0.5, y_jitter = 0.5)
    
    
        ax.xaxis.set_ticks(np.arange(0, len(truth_lengths), 100))













    





        """ Load old .csv FROM MATLAB OUTPUT and plot it??? 
        
        
        
        
        
        """
        #MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO.csv'
        MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'
        #MATLAB_name = 'output.csv'
        
        
        MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
        
        
        all_cells_MATLAB = np.unique(MATLAB_auto_array.SERIES)
        all_cells_TRUTH = np.unique(truth_array.SERIES)
    
    
        all_lengths = []
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
                       
                            all_lengths.append(track_length_TRUTH - track_length_MATLAB)
                            truth_lengths.append(track_length_TRUTH)
                            MATLAB_lengths.append(track_length_MATLAB)   
                            
                            all_cell_nums.append(num_new_truth)
                       
                       
        plt.figure(); plt.plot(all_lengths)
        print(len(np.where(np.asarray(all_lengths) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths) < 0)[0]))         
                                       
        
        
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
    
    
    
    
    