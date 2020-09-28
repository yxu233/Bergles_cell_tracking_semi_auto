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
def prep_input_for_CNN(cell, input_im, next_input, cur_seg, next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp, next_bool=1, retry=0):

    ### (1) crop the current frame (all segmentations)
    crop_cur_seg, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
   
    ### (2) create the target seed for the current frame and crop it out
    blank_im = np.zeros(np.shape(input_im))
    blank_im[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 1                 
    crop_seed, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(blank_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    
    ### (3) set the target seed to be 50, and the other segmentations in the frame to be 10
    crop_cur_seg[crop_cur_seg > 0] = 10
    crop_cur_seg[crop_seed > 0] = 50                 
       
    ### (4) crop the raw input data for the current and next segmentation
    crop_im, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(input_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    crop_next_input, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_input, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
    
    ### (5) get segmentation for next frame
    crop_next_seg, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    crop_next_seg_non_bin = np.copy(crop_next_seg)
    
    """ for testing doubles again at the end, eliminate all previously checked"""
    if retry:
        bw = crop_next_seg
        bw[bw > 0] = 1
        label = measure.label(bw)
        cc = measure.regionprops(label, intensity_image=crop_next_seg_non_bin)
        for obj in cc:
            coord = obj['coords']
            min_int = obj['min_intensity']
            if min_int == 250:  ### remove visited before cells
                bw[coord[:, 0], coord[:, 1], coord[:, 2]] = 0
                
        crop_next_seg = bw
        
        
    
    crop_next_seg[crop_next_seg > 0] = 10
  
    
    """ Get ready for inference """
    if next_bool:
        batch_x = np.zeros((4, ) + np.shape(crop_im))
        batch_x[0,...] = crop_im
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = crop_next_input
        batch_x[3,...] = crop_next_seg
        batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)

    else:
        batch_x = np.zeros((3, ) + np.shape(crop_im))
        batch_x[0,...] = crop_im
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = crop_next_input
        #batch_x[3,...] = crop_next_seg
        batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)

    
    ### NORMALIZE
    batch_x = normalize(batch_x, mean_arr, std_arr)

    return batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over



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
         all_lens = []
         for multi_check in cc_seg_train:
              coords_m = multi_check['coords']
              crop_next_seg[coords_m[:, 0], coords_m[:, 1], coords_m[:, 2]]
              #print('out')
              all_lens.append(len(coords_m))
              all_cur_r = [];
              for seg_check in cc_next:
                   coords_n = seg_check['coords']
                  
                   if np.any((coords_m[:, None] == coords_n).all(-1).any(-1)):   ### overlapped
                        ratio = len(coords_m)/len(coords_n)
                        all_cur_r.append(ratio)
                        
              if len(all_cur_r) > 0:
                   all_ratios.append(all_cur_r[np.argmin(np.abs(np.asarray(all_cur_r) - 1))])  ### OTHERWISE MIGHT MATCH EVEN MORE???
                             ### SHOULD ONLY APPEND RATIO OF LARGEST MATCHED
                        
         
               
         ### if there is ratio
         if len(all_ratios) > 0:
             best_coords = cc_seg_train[np.argmin(np.abs(np.asarray(all_ratios) - 1))]['coords']
             best[best_coords[:, 0], best_coords[:, 1], best_coords[:, 2]] = 1
             
         ### otherwise, if no cell matches between the 2 frames, pick the largest cell
         else:
             best_coords = cc_seg_train[np.argmax(all_lens)]['coords']
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
         
         #if tracked_cells_df.X[idx_next] == next_centroid[0] and tracked_cells_df.Y[idx_next] == next_centroid[1] and tracked_cells_df.Z[idx_next] == next_centroid[2]:
             
             
         if any((next_centroid == x).all() for x in tracked_cells_df.coords[idx_next]):
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
         sub[0] = (sub[0] - x_check)
         sub[1] = (sub[1] - y_check)
         sub[2] = (sub[2] - z_check)       ### SHOULD I SCALE Z???
         
         dist = np.linalg.norm(sub)
         all_dist.append(dist)
 
         

    """ also this might be empty??? """
    if len(all_dist) == 0:
        print('what? error, didnt find other cell')
        #zzz
        
    else:
         
        closest = all_dist.index(min(all_dist))
         
        """ drop everything else that is not close and set their series to be RED, set the closest to be YELLOW """
        keep_series = dup_series[closest]
        tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == keep_series] = 'YELLOW'
         
         
        dup_series = np.delete(dup_series, np.where(np.asarray(dup_series) == keep_series)[0])
    
        if len(dup_series) == 0:
            print('what? error, didnt find other cell')
            #zzz
        
        
        for dup in dup_series:
              tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == dup] = 'RED'
              
              ### also delete the 2nd occurence of it
              
              ### OR MAYBE SHOULD RE-CHECK TO SEE IF IT CAN ASSOCIATE TO OTHER CELL???
              
              
              #tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(((tracked_cells_df["SERIES"] == dup) & (tracked_cells_df["FRAME"] == frame_num)))[0]])


    return tracked_cells_df, dup_series







""" Parse the data and compare to ground truth for later """

def parse_truth(truth_cur_im, truth_array, truth_output_df, truth_next_im, seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size, blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp):
   
    crop_truth_cur, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(truth_cur_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
    crop_truth_next, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(truth_next_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      

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
def associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, min_size=0, truth=0, truth_output_df=0, truth_next_im=0, truth_array=0):
    bw_next_seg = np.copy(next_seg)
    bw_next_seg[bw_next_seg > 0] = 1
    
    labelled = measure.label(bw_next_seg)
    next_cc = measure.regionprops(labelled)
      
      
    ### add the cells from the first frame into "tracked_cells" matrix
    num_new = 0; num_new_truth = 0
    for idx, cell in enumerate(next_cc):
       coords = cell['coords']
       
       
       if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250) and len(coords) > min_size:   ### 250 means already has been visited
            series = np.max(tracked_cells_df.SERIES) + 1        
            centroid = cell['centroid']
     
            """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
            # if int(centroid[2]) + z_size/2 >= lowest_z_depth:
            #       continue                        
            
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





""" Given an image with segmentations for the next frame + coordinates of center of current frame
        find closest cell to associate current cell to within min_dist of 20        

"""
def associate_to_closest(tracked_cells_df, cc, seg_train, x, y, z, box_xyz, box_over, cur_idx, frame_num, width_tmp, height_tmp, depth_tmp, min_dist=20):

    all_dist = []
    for obj in cc:
        center = obj['centroid']
        
        center = np.asarray(center)
        center = scale_single_coord_to_full(center, box_xyz, box_over)
         
        dist = [x, y, z] - center
        dist = np.linalg.norm(dist)
        all_dist.append(dist)
         
   
    """ only keep if smallest is within 20 pixels """
    closest_obj = cc[np.argmin(all_dist)]
    closest_dist = np.min(all_dist)
   
    #print(closest_dist)
   
    next_coords = []; next_centroid = []; cell_next = [];
    if closest_dist <= min_dist:
        index_next = np.where((tracked_cells_df["SERIES"] == cur_idx) & (tracked_cells_df["FRAME"] == frame_num))[0]
        
        
        if len(index_next) > 0:        
            cell_next = tracked_cells_df.iloc[index_next[0]]
        
        next_coords = np.asarray(closest_obj['coords'])
        seg_train = np.zeros(np.shape(seg_train))
        seg_train[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 1
        
        next_coords = scale_coords_of_crop_to_full(next_coords, box_xyz, box_over)
   
   
        next_centroid = np.asarray(closest_obj['centroid'])
        next_centroid = scale_single_coord_to_full(next_centroid, box_xyz, box_over)
        
        """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
        next_coords = check_limits([next_coords], width_tmp, height_tmp, depth_tmp)[0]
        next_centroid = check_limits_single([next_centroid], width_tmp, height_tmp, depth_tmp)[0]       
        
            
    return cell_next, next_coords, seg_train, next_centroid, closest_dist



""" Predict next xyz given cells in current crop frame of arbitrary size       

"""
def predict_next_xyz(tracked_cells_df, x, y, z, crop_size, z_size, frame_num):
    cell_locs = np.where((tracked_cells_df.X >= x - crop_size/2) & (tracked_cells_df.X <= x + crop_size/2) & (tracked_cells_df.Y >= y - crop_size/2) & (tracked_cells_df.Y <= y + crop_size/2) & (tracked_cells_df.Z >= z - z_size/2) & (tracked_cells_df.Z <= z + z_size/2) & (tracked_cells_df.FRAME == frame_num - 1))
    cells_cur_crop = tracked_cells_df.iloc[cell_locs]  
    series_cur_crop = cells_cur_crop.SERIES
                    
    cur_centers = []; next_centers = []; num_tracked = 0;
    for series in series_cur_crop:
        
        locs_cur = np.where((tracked_cells_df.SERIES == series) & (tracked_cells_df.FRAME == frame_num - 1))[0]
        locs_next =  np.where((tracked_cells_df.SERIES == series) & (tracked_cells_df.FRAME == frame_num))[0]
        
        ### if is tracked cell (location == 2)
        if len(locs_cur)  == 1 and len(locs_next) == 1:
            cur_centroid = [tracked_cells_df.iloc[locs_cur[0]].X, tracked_cells_df.iloc[locs_cur[0]].Y, tracked_cells_df.iloc[locs_cur[0]].Z]
            next_centroid = [tracked_cells_df.iloc[locs_next[0]].X, tracked_cells_df.iloc[locs_next[0]].Y, tracked_cells_df.iloc[locs_next[0]].Z]
            
            cur_centers.append(cur_centroid)
            next_centers.append(next_centroid)
            
            num_tracked += 1
            
    if len(next_centers) > 0:
        all_vectors = (np.asarray(next_centers) - np.asarray(cur_centers))
        
        median_disp = [np.median(all_vectors[:, 0]), np.median(all_vectors[:, 1]), np.median(all_vectors[:, 2])]
   
        pred_x = int(x + median_disp[0])
        pred_y = int(y + median_disp[1])
        pred_z = int(z + median_disp[2])
        
    else:
        pred_x = x
        pred_y = y
        pred_z = z
        
        num_tracked = 0
    
    return pred_x, pred_y, pred_z, num_tracked




""" Use predictions to cleanup whatever candidates you wish to try """
def clean_with_predictions(tracked_cells_df, candidate_series, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=12):
    print('cleaning with predictions')
    
    deleted = 0; term_count = 0; new = 0; not_changed = 0;
    not_assoc = 0;
    to_drop = [];
    recheck_series = []   ### TO BE RECHECKED LATER
    for series in candidate_series:
        
        
        index = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
        cell = tracked_cells_df.iloc[index[0]]
        x = cell.X; y = cell.Y; z = cell.Z;
        pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size, z_size, frame_num)

        ### DEBUG: when debugging get next cell too and plot it
        #im = np.zeros(np.shape(next_seg))
        index_next = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num))[0]
        if len(index_next) > 0:
            cell_next = tracked_cells_df.iloc[index_next[0]]
            x_n = cell_next.X; y_n = cell_next.Y; z_n = cell_next.Z;

        else:
            cell_next = []
            
        # im = np.zeros(np.shape(next_seg))
        # im[x_n, y_n, z_n] = 1
        # im[pred_x, pred_y, pred_z] = 2
        # crop_seg_out, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)       
    
        # batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
        #                                                                                           next_seg, 0, 0, x, y, z, crop_size, z_size,
        #                                                                                           height_tmp, width_tmp, depth_tmp, next_bool=next_bool)   

        # inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
  
        # # forward pass to check validation
        # output_val = unet(inputs_val)
  
        # """ Convert back to cpu """                                      
        # output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
        # seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)


        crop_next_seg, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                    

        """ now associate cell to the object closest to the predicted location
        
                a few things can happen:
                        (1) no cell currently occupied, so just associate
                        (2) currently occupied, in which case, remove that association, and add that cell to the new list of cells to check
                        (3) no cell found nearby, in which case, set as terminated
        """            
        
        ### use predicted xyz only if num_tracked > 5:
        bw = crop_next_seg
        bw[bw > 0] = 1
        label = measure.label(crop_next_seg)
        cc = measure.regionprops(label)
        next_coords = []
        if len(cc) > 0:
        
            num_tracked = 0; scale = 0.25
            while num_tracked < 4 and scale <= 2:
                pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                
                scale += 0.25 
                 
            ### Try to associate with nearest cell in crop_next_seg        
            if num_tracked >= 4:
                empty, next_coords, seg_train, next_centroid, closest_dist = associate_to_closest(tracked_cells_df, cc, seg_train, pred_x, pred_y, pred_z, box_xyz, box_over, series, 
                                                                                                 frame_num, width_tmp, height_tmp, depth_tmp, min_dist=min_dist)       

        """ Change next coord only if something close was found
        """
        term_bool = 0
        if len(next_coords) > 0:   ### only add if not empty
            if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250):
                """ CASE #2: otherwise, find which cell is currently matched to this cell, then check what the prediction is for that new cell
                    and find which is closer 
                    
                    """
                    
                found = 0
                for series_check in np.unique(tracked_cells_df.SERIES):
                    index_cur_c = np.where((tracked_cells_df["SERIES"] == series_check) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]     
                    index_next_c = np.where((tracked_cells_df["SERIES"] == series_check) & (tracked_cells_df["FRAME"] == frame_num))[0]                        
                    if len(index_next_c) > 0:
                        
                        ### check if matched
                        cell_check = tracked_cells_df.iloc[index_next_c[0]]
                        x_c = cell_check.X; y_c = cell_check.Y; z_c = cell_check.Z;
                                
                        ### find if row matches row in next_coords
                        if len(np.where((next_coords == (x_c, y_c, z_c)).all(axis=1))[0]) > 0:
                            #print('matched')
                            found += 1
                                                        
                            """ Predict where this cell is going and see which is closer """
                            cell_check_cur = tracked_cells_df.iloc[index_cur_c[0]]
                            x_c_cur = cell_check_cur.X; y_c_cur = cell_check_cur.Y; z_c_cur = cell_check_cur.Z;                                    
                            
                
                            ### keep looping until have sufficient neighbor landmarks
                            num_tracked = 0; scale = 0.25
                            while num_tracked < 4 and scale <= 2:
                                pred_x_c, pred_y_c, pred_z_c, num_tracked = predict_next_xyz(tracked_cells_df, x_c_cur, y_c_cur, z_c_cur, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                                
                                scale += 0.25
                                #
                            
                            
                            
                            
                            dist_to_check = np.linalg.norm(np.asarray([pred_x_c, pred_y_c, pred_z_c]) - next_centroid)
                            dist_to_new = np.linalg.norm(np.asarray([pred_x, pred_y, pred_z]) - next_centroid)
                            
                            ### DELETE CELL
                            if dist_to_new <= dist_to_check:
                                #print('delete cell')
                                deleted += 1
                                to_drop.append(index_next_c[0])   ### drop NEXT FRAMES cell that was conflicting
                                
                                ### and append the cell to check later
                                recheck_series.append(series_check)

                                ### and add the new cell
                                ### ADD CELL
                                if len(cell_next) > 0:
                                    
                                    old_coords = cell_next.coords
                                    next_seg[old_coords[:, 0], old_coords[:, 1], old_coords[:, 2]] = 255;   ### RESET NEXT_SEG 
                                    
                                    cell_next.coords = next_coords
                                    cell_next.X = next_centroid[0]
                                    cell_next.Y = next_centroid[1]
                                    cell_next.Z = next_centroid[2]
                                else:
                                    row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                                    tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     
                
                                    """ Change next coord """
                                    next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;    
                                
                            else:
                                ### otherwise, leave current cell as empty
                                #print('not linked')
                                term_bool = 1;
                                term_count += 1
                                
                                next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 255;   ### RESET NEXT_SEG 
                     
                                
                if found == 0:
                    not_assoc += 1
                    print(not_assoc)


            """ CASE #1: if next_seg does NOT contain 250, then just associate to current cell"""
            #next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;      

            if not np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250) or not found:
                
                ### ADD CELL
                if len(cell_next) > 0:
                    old_coords = cell_next.coords
                    next_seg[old_coords[:, 0], old_coords[:, 1], old_coords[:, 2]] = 255;   ### RESET NEXT_SEG 
                    
                    
                    cell_next.coords = next_coords
                    cell_next.X = next_centroid[0]
                    cell_next.Y = next_centroid[1]
                    cell_next.Z = next_centroid[2]
                else:
                    row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                    tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     

                    """ Change next coord """
                    next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;                      
                    
                #print('new cell')
                new += 1

                        
                     
                                
        if len(next_coords) == 0 or term_bool:
            """   CASE #3: none matched, set as eliminated and remove cell_next """
            if len(index_next) > 0:
                to_drop.append(index_next[0])
                deleted += 1
            
            else:
                not_changed += 1
                
        else:
            print('what')
            #zzz
                

    """ drop everything that has .coords = [] """
    tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[to_drop])
    print('new associations: ' + str(new) + '\ndeleted_old_tracks: ' + str(deleted) + '\nterminated: ' + str(term_count) + '\nnot changed: ' + str(not_changed))
    
    
    return tracked_cells_df, recheck_series, next_seg







""" Given dataframe, find predicted distances of ALL

        can be used to find possible errors
        
        ***also used to plot show distribution of how accurate Cody's traces are to predicting each other!!!
"""

def check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh = 10):
    print('checking distances')
    all_dist = []; check_series = []; dist_check = []; num_checked = 0;
    for series in np.unique(tracked_cells_df.SERIES):
        index = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
        if len(index) > 0:
            
            num_checked += 1
            
            cell = tracked_cells_df.iloc[index[0]]
             
            ### go to unvisited cells
            x = cell.X; y = cell.Y; z = cell.Z;           
            
            
            ### keep looping until have sufficient neighbor landmarks
            num_tracked = 0; scale = 0.25
            while num_tracked < 4 and scale <= 2:
                pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                
                scale += 0.25
            
            if num_tracked >= 4:
                index_next = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num))[0]
                if len(index_next) > 0:
                    cell_next = tracked_cells_df.iloc[index_next[0]]    
    
                    x_n = cell_next.X; y_n = cell_next.Y; z_n = cell_next.Z; 
                    
                    dist = np.linalg.norm(np.asarray([x_n, y_n, z_n]) - np.asarray([pred_x, pred_y, pred_z]))
                    all_dist.append(dist)
                    #print(dist)
                    
                    if dist > dist_error_thresh:
                        check_series.append(series)
                        dist_check.append(dist)                       
    # len(np.where(np.asarray(all_dist) > 4)[0])
    # len(all_dist) 
    
    return tracked_cells_df, all_dist, dist_check, check_series