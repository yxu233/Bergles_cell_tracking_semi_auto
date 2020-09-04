#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:24:27 2020

@author: user
"""


from skimage import measure
import numpy as np
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
from matlab_crop_function import *
from UNet_functions_PYTORCH import *


""" Crop and prep input to prepare for CNN 

"""
def prep_input_for_CNN(cell, input_im, next_input, cur_seg, next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp):

    ### (1) crop the current frame (all segmentations)
    crop_cur_seg, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(cur_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
   
    ### (2) create the target seed for the current frame and crop it out
    blank_im = np.zeros(np.shape(input_im))
    blank_im[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 1                 
    crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(blank_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    
    ### (3) set the target seed to be 50, and the other segmentations in the frame to be 10
    crop_cur_seg[crop_cur_seg > 0] = 10
    crop_cur_seg[crop_seed > 0] = 50                 
       
    ### (4) crop the raw input data for the current and next segmentation
    crop_im, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(input_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    crop_next_input, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(next_input, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
    
    ### (5) get segmentation for next frame
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
    
    ### NORMALIZE
    batch_x = normalize(batch_x, mean_arr, std_arr)

    return batch_x, crop_next_seg, crop_seed, box_x_min, box_y_min, box_z_min 



""" When there is more than one cell/object identified in next frame,
        need to associate the one that is BEST MATCHED with the original frame
"""
def select_one_from_excess(seg_train, crop_next_seg):
    label = measure.label(seg_train)
    cc_seg_train = measure.regionprops(label)
    if len(cc_seg_train) > 1:
         #doubles += 1
         print('multi objects in seg')
         
         
         ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
         """ Best one is one that takes up most of the area in the "next_seg" """
         #add = crop_next_seg + seg_train
         label = measure.label(crop_next_seg)
         cc_next = measure.regionprops(label)
         
         
         best = np.zeros(np.shape(crop_next_seg))
         
         all_ratios = []
         for multi_check in cc_seg_train:
              coords_m = multi_check['coords']
              crop_next_seg[coords_m[:, 0], coords_m[:, 1], coords_m[:, 2]]
              #print('out')
              for seg_check in cc_next:
                   coords_n = seg_check['coords']
                   if np.any((coords_m[:, None] == coords_n).all(-1).any(-1)):   ### overlapped
                        ratio = len(coords_m)/len(coords_n)
                        all_ratios.append(ratio)
                        #print('in')
            
               
         if len(all_ratios) > 0:
             best_coords = cc_seg_train[all_ratios.index(max(all_ratios))]['coords']
             best[best_coords[:, 0], best_coords[:, 1], best_coords[:, 2]] = 1
         seg_train = best
         label = measure.label(seg_train)
         cc_seg_train = measure.regionprops(label)      
         
         
    return cc_seg_train, seg_train, crop_next_seg




""" Find cell number that matches in next_frame (i.e. where more than 1 cell 
         from cur_frame points to a cell in next frame)
"""
def sort_double_linked(tracked_cells_df, next_centroid, frame_num):
    ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
    """ Find cell number that matches in next_frame (i.e. where more than 1 cell from cur_frame points to a cell in next frame)"""
    dup_series = []
    for idx_next in np.where(tracked_cells_df.FRAME == frame_num)[0]:
         
         if tracked_cells_df.X[idx_next] == next_centroid[0] and tracked_cells_df.Y[idx_next] == next_centroid[1] and tracked_cells_df.Z[idx_next] == next_centroid[2]:
              dup_series.append(tracked_cells_df.SERIES[idx_next])
              
              
    """ Get location of cell on previous frame corresponding to these SERIES numbers
              and find cell that is CLOSEST
    
    """
    all_dist = []
    for dup in dup_series:
         index = np.where((tracked_cells_df["SERIES"] == dup) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
         x_check = tracked_cells_df.X[index];
         y_check = tracked_cells_df.Y[index];
         z_check = tracked_cells_df.Z[index];
         
         sub = np.copy(next_centroid)
         sub[0] = (sub[0] - x_check) * 0.083
         sub[1] = (sub[1] - y_check) * 0.083
         sub[2] = (sub[2] - z_check) * 3
         
         dist = np.linalg.norm(sub)
         all_dist.append(dist)
         
    closest = all_dist.index(min(all_dist))
     
    """ drop everything else that is not close and set their series to be RED, set the closest to be YELLO """
    keep_series = dup_series[closest]
    tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == keep_series] = 'YELLOW'
     
     
    dup_series = np.delete(dup_series, np.where(np.asarray(dup_series) == keep_series)[0])
    for dup in dup_series:
          tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == dup] = 'RED'
          
          ### also delete the 2nd occurence of it
          tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(((tracked_cells_df["SERIES"] == dup) & (tracked_cells_df["FRAME"] == frame_num)))[0]])


    return tracked_cells_df







""" Parse the data and compare to ground truth for later """

def parse_truth(truth_cur_im, truth_array, truth_output_df, truth_next_im, seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size, blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp):
   
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
    
    
    """ REMOVE EVERYTHING IN CROP_NEXT_SEG THAT DOES NOT MATCH WITH SOMETHING CODY PUT UP, to prevent FPs of unknown checking"""
    # if nothing in the second frame
    value_cur_frame = np.unique(crop_truth_cur[crop_seed > 0])
    value_cur_frame = np.delete(value_cur_frame, np.where(value_cur_frame == 0)[0][0])  # DELETE zero
    
    values_next_frame = np.unique(crop_truth_next[crop_next_seg > 0])
    
    ### skip if no match on cur frame in truth
    if len(value_cur_frame) == 0:
         #not_registered += 1;  
         print('not_registered')
         return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude
    
    if not np.any(np.in1d(value_cur_frame, values_next_frame)):   ### if it does NOT exist on next frame               
         ### BUT IF EXISTS IN GENERAL ON 2nd frame, just not in the segmentation, then skip ==> is segmentation missed error
         values_next_frame_all = np.unique(crop_truth_next[crop_truth_next > 0])
         if np.any(np.in1d(value_cur_frame, values_next_frame_all)):
              #seg_error += 1;
              print('seg_error')
              list_exclude.append(value_cur_frame[0])
              return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude  # SKIP
         
         ### count blobs:
         if len(value_cur_frame) > 1:
              blobs += 1
              #blobs = 1
       
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
              FN += 1
              
              """ Missing a lot of little tiny ones that are dim
                        *** maybe at the end add these back in??? by just finding nearest unassociated ones???
              """
         else:
              
            ### (2) find out if seg_train has identified point with same index as previous frame
            values_next_frame = np.unique(crop_truth_next[seg_train > 0])
            values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == 0)[0][0])  # delete zeros
         
            if np.any(np.in1d(value_cur_frame, values_next_frame)):
                   TP += 1
                   values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == values_next_frame)[0][0])
              
            """ if this is first time here, then also add the ones from initial index """
            if frame_num == 1:
                 row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == 0)]
                 truth_output_df = truth_output_df.append(row)                                      
            row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == frame_num)]
            truth_output_df = truth_output_df.append(row) 
            
            # but if have more false positives
            if len(values_next_frame) > 0:
              #FP += len(values_next_frame)
              extras += len(values_next_frame)
    #plt.close('all')  
    
    
    return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude


""" Associate remainder as newly segmented cells """
def associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, truth=0, truth_output_df=0, truth_next_im=0, truth_array=0):
    bw_next_seg = np.copy(next_seg)
    bw_next_seg[bw_next_seg > 0] = 1
    
    labelled = measure.label(bw_next_seg)
    next_cc = measure.regionprops(labelled)
      
      
    ### add the cells from the first frame into "tracked_cells" matrix
    num_new = 0; num_new_truth = 0
    for idx, cell in enumerate(next_cc):
       coords = cell['coords']
       
       
       if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250):   ### 250 means already has been visited
            series = np.max(tracked_cells_df.SERIES) + 1        
            centroid = cell['centroid']
     
            """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
            if int(centroid[2]) + z_size/2 >= lowest_z_depth:
                  continue                        
            
            row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': frame_num, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
            tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
                                  
            """ Add to TRUTH as well """
            if truth:
                value_next_frame = np.max(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                if value_next_frame > 0:
                     row  = truth_array[(truth_array["SERIES"] == np.max(value_next_frame)) & (truth_array["FRAME"] == frame_num)]
                     if truth:
                         truth_output_df = truth_output_df.append(row) 
                     print('value_next_frame')
                     num_new_truth += 1
                 
            num_new += 1
            
       print('Checking cell: ' + str(idx) + ' of total: ' + str(len(next_cc)))
                        
    return tracked_cells_df, truth_output_df, truth_next_im, truth_array


