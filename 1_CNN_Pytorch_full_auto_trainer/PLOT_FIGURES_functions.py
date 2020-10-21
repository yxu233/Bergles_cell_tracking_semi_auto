#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:05:26 2020

@author: user
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt


import pandas as pd
import scipy.stats as sp
import seaborn as sns

""" Plot """
def plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_', depth_lim_lower=0, depth_lim_upper=100000, only_one_plot=0,ax_title_size=18, leg_size=14):
    new_cells_per_frame =  np.zeros(np.max(tracked_cells_df.FRAME) + 1)
    terminated_cells_per_frame = np.zeros(np.max(tracked_cells_df.FRAME) + 1)
    num_total_cells_per_frame = np.zeros(np.max(tracked_cells_df.FRAME) + 1)
    
    
    num_baseline = np.zeros(np.max(tracked_cells_df.FRAME) + 1)
    num_new = np.zeros(np.max(tracked_cells_df.FRAME) + 1)
    
    for cell_num in np.unique(tracked_cells_df.SERIES):
        
        frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
        
        
        z_cur = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z
        
        
        if np.min(z_cur) <= depth_lim_lower or np.max(z_cur) >= depth_lim_upper:
            #print('skip')
            continue;
            
            
        beginning_frame = np.min(frames_cur_cell)
        if beginning_frame > 0:   # skip the first frame
            new_cells_per_frame[beginning_frame] += 1

                    
        term_frame = np.max(frames_cur_cell)
        if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
            terminated_cells_per_frame[term_frame] += 1
        
        for num in frames_cur_cell:
            num_total_cells_per_frame[num] += 1    
            
        """ To match Cody's plots!!! """
        ### get # of baseline cells at every timeseries
        if beginning_frame == 0:
            for frame in frames_cur_cell:
                num_baseline[frame] += 1
                
        else:
            for frame in frames_cur_cell:
                num_new[frame] += 1
            
        
    #y_pos = np.arange(np.min(tracked_cells_df.FRAME), np.max(tracked_cells_df.FRAME))) + 1)
    
    y_pos = np.arange(0, np.max(tracked_cells_df.FRAME) + 1)
    if not only_one_plot:
        
        plt.figure(); plt.bar(y_pos, new_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'new cells per frame'
        #plt.title(name);
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# new cells', fontsize=ax_title_size)
        # ax.set_xticklabels(x_ticks, rotation=0, fontsize=12)
        # ax.set_yticklabels(y_ticks, rotation=0, fontsize=12)
        plt.tight_layout()
        plt.savefig(sav_dir + add_name + name + '.png')
    
        plt.figure(); plt.bar(y_pos, terminated_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'terminated cells per frame'
        #plt.title(name)
        
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# terminated cells', fontsize=ax_title_size)
        plt.tight_layout()
        plt.savefig(sav_dir + add_name + name + '.png')
    
        
        plt.figure(); plt.bar(y_pos, num_total_cells_per_frame, color='k')
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        name = 'number cells per frame'
        #plt.title(name)
        plt.tight_layout()
        plt.xlabel('time frame', fontsize=16); plt.ylabel('# cells', fontsize=ax_title_size)
        plt.tight_layout()
        plt.savefig(sav_dir + add_name + name + '.png')
    


    """ Normalize to proportions like Cody did
    
    """
    # new_cells_per_frame
    # terminated_cells_per_frame
    # num_total_cells_per_frame
    
    
    """ USE CUMULATIVE SUM OF NEW CELLS """   
    baseline = num_baseline[0]
    
    norm_tots = num_baseline/baseline
    norm_new = num_new/baseline

    width = 0.35       # the width of the bars: can also be len(x) sequence
    plt.figure()
    p1 = plt.bar(y_pos, norm_tots, yerr=0, color='k')
    p2 = plt.bar(y_pos, norm_new, bottom=norm_tots, yerr=0, color='g')
    
    line = np.arange(-5, len(y_pos) + 5, 1)
    plt.plot(line, np.ones(len(line)), 'r--', linewidth=2, markersize=10)
    
    plt.ylabel('Proportion of cells', fontsize=ax_title_size)
    plt.xlabel('weeks', fontsize=16); 
    plt.xticks(np.arange(0, len(y_pos), 1))
    plt.xlim(-1, len(y_pos))
    plt.ylim(0, 1.4)
    plt.yticks(np.arange(0, 1.4, 0.2))
    #plt.legend((p1[0], p2[0]), ('Baseline', 'New cells'), fontsize=leg_size)
    
    
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
    name = 'normalized recovery'
    plt.tight_layout()
    plt.savefig(sav_dir + add_name + name + '.png')
    
    
    return norm_tots, norm_new
    






def get_sizes_and_z_from_cell_list(tracked_cells_df, cells, frame_num, scale_xy, scale_z, neighbors, density, scaled_vol=1):
   new_metric = []
   new_z = []
   total_metric = []
   total_z = []
   
   ### get list of all cell locations on current frame
   all_x = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].X
   all_y =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Y
   all_z =  tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].Z
   
   all_series_num = tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].SERIES
   
   if not scaled_vol:
       all_coords = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].coords)

   else:
       all_vols = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.FRAME == frame_num)].vol_rescaled)

   
   all_centroids = [np.asarray(all_x) * scale_xy, np.asarray(all_y) * scale_xy, np.asarray(all_z) * scale_z]
   all_centroids = np.transpose(all_centroids)
          
   
   if density:
       nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(all_centroids)
       distances, indices = nbrs.kneighbors(all_centroids)        
              
       """ Get all distances to see distribution comparing DEPTH and density/distance """
       for obj in distances:
           cur_dist = obj[1:-1]
           mean = np.mean(cur_dist)
           #total_dists.append(cur_dist)
           total_metric = np.concatenate((total_metric, [mean]))           
       total_z = np.concatenate((total_z, all_centroids[:, -1]))
       
       """ Go cell by cell through NEW cells only """
       for cur_cell in cells:
           
           dist_idx = np.where(all_series_num == cur_cell)
           cur_dists = distances[dist_idx][0][1:-1]
           mean_dist = np.mean(cur_dists)
           new_metric = np.concatenate((new_metric, [mean_dist]))
           
           ### compare it with all the other cells at the same depth that are NOT new
           cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
           new_z = np.concatenate((new_z, [cur_z]))    


   elif not density:
       """ Get all distances to see distribution comparing DEPTH and density/distance """
       if not scaled_vol:
           for obj in all_coords:
               cur_vol = len(obj)
               total_metric = np.concatenate((total_metric, [cur_vol]))
       else:
           total_metric = np.concatenate((total_metric, all_vols))

            
       total_z = np.concatenate((total_z, np.asarray(all_z) * scale_z))            


       """ Get terminated cells """
       for cur_cell in cells:
            
            dist_idx = np.where(all_series_num == cur_cell)
            
            if not scaled_vol:
                cur_vol = len(all_coords[dist_idx][0])
                new_metric = np.concatenate((new_metric, [cur_vol]))
            else:
                cur_vol = all_vols[dist_idx]
                new_metric = np.concatenate((new_metric, cur_vol))
            
            ### compare it with all the other cells at the same depth that are NOT new
            cur_z = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cur_cell)].Z)[0] * scale_z
            new_z = np.concatenate((new_z, [cur_z]))

                
   return total_metric, total_z, new_metric, new_z   
      
        
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







""" FOR POOLING """
""" Plot pooled normalized proportions """
def balance_find_mean_std_sem(list_arrs):
    row_lengths = []
    for row in list_arrs:
        row_lengths.append(len(row))
    max_length = max(row_lengths)  
    
    
    balanced_rows = []
    for row in list_arrs:       
        while len(row) < max_length:      
             row = np.concatenate((row, [np.nan]))       
        balanced_rows.append(row)        
    balanced_array = np.array(balanced_rows)          
            
    
    ### find mean and std of sizes
    size_arr = np.stack(balanced_array)
    mean = np.nanmean(size_arr, axis=0)
    std = np.nanstd(size_arr, axis=0)
    
    ### find sem
    all_n = []
    for col_idx in range(len(balanced_array[0, :])):
        all_n.append(len(np.where(~np.isnan(balanced_array[:,col_idx]))[0]))
        
    sem = std/np.sqrt(all_n)
    
    return mean, std, sem


def plot_pooled_trends(all_norm_tots, all_norm_new, ax_title_size, sav_dir, add_name='OUTPUT_', max_week=-1):

    # ### sort list of lists so can be converted to array for mean and std
    mean_tots, std_tots, sem_tots = balance_find_mean_std_sem(all_norm_tots)
    
    mean_tots = mean_tots[:max_week]
    std_tots = std_tots[:max_week]
    sem_tots = sem_tots[:max_week]
    
    
    width = 0.35       # the width of the bars: can also be len(x) sequence
    plt.figure()
    p1 = plt.bar(np.arange(len(mean_tots)), mean_tots, yerr=0, color='k')
    plt.errorbar(np.arange(len(mean_tots)), mean_tots, yerr=sem_tots, color='r', fmt='none', capsize=1, capthick=1)
    
    """ Plot new green on top """
    mean, std, sem = balance_find_mean_std_sem(all_norm_new)
    
    mean = mean[:max_week]
    std = std[:max_week]
    sem = sem[:max_week]

    p2 = plt.bar(np.arange(len(mean)), mean, bottom=mean_tots, yerr=0, color='g')
    plt.errorbar(np.arange(len(mean)), mean + mean_tots, yerr=sem, color='r', fmt='none', capsize=1, capthick=1)
    
    
    line = np.arange(-5, len(mean) + 5, 1)
    plt.plot(line, np.ones(len(line)) * 0.5, 'r--', linewidth=2, markersize=10)
    
    plt.ylabel('Proportion of cells', fontsize=ax_title_size)
    plt.xlabel('weeks', fontsize=16); 
    plt.xticks(np.arange(0, len(mean), 1))
    plt.xlim(-1, len(mean))
    plt.ylim(0, 1.5)
    plt.yticks(np.arange(0, 1.5, 0.2))
    #plt.legend((p1[0], p2[0]), ('Baseline', 'New cells'), fontsize=leg_size)
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
    name = 'normalized recovery'
    plt.tight_layout()
    
    name = 'POOLED_normalized_recovery'
   
    plt.savefig(sav_dir + add_name + name + '.png')
        


""" CONVERT TO MESH"""
def get_all_points_on_line(p_lim, p_0, z_x, z_0, stable_val, b):
    all_z = []
    
    all_x = np.arange(p_0, p_lim)
    all_stable = np.full(len(all_x), stable_val)
    for x in range(p_lim - p_0):
        z = ((z_x - z_0)/(p_lim - p_0)) * x + b
        all_z.append(z)
    all_z = np.asarray(all_z)
    
    return all_z, all_stable, all_x
   
def points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy):
    ### connect first origin with upper right corner
    all_z, all_stable, all_x = get_all_points_on_line(p_lim=x_lim, p_0=x_0, z_x=z_x, z_0=z_0, stable_val=y_0, b=z_0)
    
    line_1 = np.transpose(np.asarray([all_x, all_stable, all_z]))


    ### then connect bottom left with bottom right corner
    all_z, all_stable, all_x = get_all_points_on_line(p_lim=x_lim, p_0=x_0, z_x=z_xy, z_0=z_y, stable_val=y_lim, b=z_y)
    line_2 = np.transpose(np.asarray([all_x, all_stable, all_z]))
    
    
    ### NOW link points from line 1 to line 2
    all_lines = []
    #all_lines.append(line_1)
    for idx, coords in enumerate(line_1):
        x_0 = coords[0]
        y_0 = int(coords[1])
        z_0 = coords[2]
        
        coords_2 = line_2[idx]
        x_1 = coords_2[0]
        y_1 = int(coords_2[1])
        z_1 = coords_2[2]
        
        all_z, all_stable, all_y = get_all_points_on_line(p_lim=y_1, p_0=y_0, z_x=z_1, z_0=z_0, stable_val=x_0, b=z_0)
        line = np.transpose(np.asarray([all_stable, all_y, all_z]))       
        
        all_lines.append(line)
        
    #all_lines.append(line_2)
        
    all_lines = np.vstack(all_lines)
    
    
    
    mesh = np.zeros([x_lim, y_lim])
    for point in all_lines:
        mesh[int(point[0]), int(point[1])] = point[2]
    ### DEBUG
    #plt.figure(); plt.imshow(mesh)
    
    return mesh




#correct if the population S.D. is expected to be equal for the two groups.

""" 
    Calculate effect size 
    
"""     
from numpy import std, mean, sqrt
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.nanmean(x) - np.nanmean(y)) / sqrt(((nx-1)*np.nanstd(x, ddof=1) ** 2 + (ny-1)*np.nanstd(y, ddof=1) ** 2) / dof)



""" 
    Plot size decay for each frame STARTING from recovery then do paired t-test to compare
    subsequent frames and their sizes
"""     
def plot_size_decay_in_recovery(tracked_cells_df, sav_dir, start_end_NEW_cell=[3, 8], min_survive_frames=3, use_scaled=1, y_lim=10000, last_plot_week=0, ax_title_size=16, x_label='Weeks after recovery', intial_week=1, figsize=(6, 5)):
    
    start_frame = start_end_NEW_cell[0]
    end_frame = start_end_NEW_cell[1]
    
    plt.figure(figsize=figsize);
    list_arrs = []
    for frame in np.unique(tracked_cells_df.FRAME):

        if frame >= start_frame and frame <= end_frame:
            all_sizes_cur_frame, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame, use_scaled=use_scaled)
            for idx in range(len(all_sizes_cur_frame)):
                
                sizes_cur_cell = all_sizes_cur_frame[idx]
                if len(sizes_cur_cell) >= min_survive_frames:   ### MUST BE ALIVE FOR AT LEAST 4 frames
                    from random import uniform
                    t = uniform(0.,2.)
                    col = (t/2.0, t/2.0, t/2.0)   # different gray colors
                    
                    plt.plot(np.arange(intial_week, len(sizes_cur_cell[:last_plot_week]) + intial_week), sizes_cur_cell[:last_plot_week], linewidth=0.5, color=col, alpha=0.2)
                    plt.ylim(intial_week - 1, y_lim)
                    plt.tight_layout()
                    list_arrs.append(list(sizes_cur_cell[:last_plot_week]))
                    
    
    ### sort list of lists so can be converted to array for mean and std
    row_lengths = []
    for row in list_arrs:
        row_lengths.append(len(row))
    max_length = max(row_lengths)  
    
    for row in list_arrs:       
        while len(row) < max_length:      
            row.append(np.nan)       
    balanced_array = np.array(list_arrs)          
            
    
    ### find mean and std of sizes
    size_arr = np.stack(balanced_array)
    mean = np.nanmean(size_arr, axis=0)
    std = np.nanstd(size_arr, axis=0)
    
    ### find sem
    all_n = []
    for col_idx in range(len(balanced_array[0, :])):
        all_n.append(len(np.where(~np.isnan(balanced_array[:,col_idx]))[0]))
        
    sem = std/np.sqrt(all_n)
    
    ### plot
    plt.plot(np.arange(intial_week, len(mean) + intial_week), mean, color='r', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(intial_week, len(mean) + intial_week), mean, color='r', marker='o', s=30)
    plt.errorbar(np.arange(intial_week, len(mean) + intial_week), mean, yerr=sem, color='r', fmt='none', capsize=10, capthick=1)

    plt.xlabel(x_label, fontsize=ax_title_size)
    plt.ylabel('Cell size ($\u03bcm^3$)', fontsize=ax_title_size)
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    from matplotlib.ticker import MaxNLocator   ### force integer tick sizes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(sav_dir + x_label + '_cell sizes changing statistics.png')
    
    """ STATISTICS """     
    ### PAIRED 2-tailed, assume equal variances
    for col in range(len(size_arr[0]) - 1):
        week_1 = size_arr[:, col]
        week_2 = size_arr[:, col + 1]
        t_test = sp.stats.ttest_rel(week_1, week_2, nan_policy='omit')
        print('p-value for week ' +  str(col) + ' vs. ' + str(col + 1) + ' of recovery for size: ' + str(t_test.pvalue))
        
        
        effect_size = cohen_d(week_1, week_2)
        print('Effect size for week ' +  str(col) + ' vs. ' + str(col + 1) + ' of recovery for size: ' + str(effect_size))
        

    return mean, sem, size_arr





""" Plot scatters of each type:
        - control/baseline day 1 ==> frame 0
                ***cuprizone ==> frame 4
        - 1 week after cupr
        - 2 weeks after cupr
        - 3 weeks after cupr 
    
    """        
def plot_size_scatters_by_recovery(tracked_cells_df, sav_dir, start_frame=3, end_frame=8, min_survive_frames=3, use_scaled=1, y_lim=10000, ax_title_size=16):
    ### (0) baseline cells
    
    all_sizes_frame_0, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame=0, use_scaled=use_scaled)
    first_frame_sizes = []
    first_z = []
    for idx, size in enumerate(all_sizes_frame_0):
        first_frame_sizes.append(size[0])
        first_z.append(all_z[idx][0])
        
    ### (1) sizes of all cells that are within 1 week old and also within 3 weeks of recovery
    #for frame in range(4, 7):
        
    ### if skipped week 2:
    first_frame_1_week = []; z_1_week = [];
    first_frame_2_week = []; z_2_week = [];
    first_frame_3_week = []; z_3_week = [];
    
    for frame in range(start_frame, end_frame):
        
        all_sizes_frame_0, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame=frame, use_scaled=use_scaled)


        for idx, size in enumerate(all_sizes_frame_0):
      
            if len(size) >= min_survive_frames:
                first_frame_1_week.append(size[0])
                z_1_week.append(all_z[idx][0])
            
            
                """ IS THIS DOUBLE COUNTING??? """
            
                if len(size) > 1:
                    first_frame_2_week.append(size[1])
                    z_2_week.append(all_z[idx][1])
                    
                if len(size) > 2:
                    first_frame_3_week.append(size[2])            
                    z_3_week.append(all_z[idx][2])
         
    return first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week
    
        
    
""" PROBABILITY CALCULATIONS given certain size"""
def find_prob_given_size(cur_frame, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week, thresh_range=[0, 8000, 100]):
    all_probs = []; all_sizes = []
    #for size_thresh in range(500, 4100, 100):
    for size_thresh in range(thresh_range[0], thresh_range[1], thresh_range[2]):   ### FOR RESCALED VOLUMES
        all_cell_sizes = np.concatenate((first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week))
        num_A = 0
        total_num = len(all_cell_sizes)
        for sizes in all_cell_sizes:
            
            #print(size_thresh)
            if sizes > size_thresh:
                num_A += 1
                
        if num_A > 0:

            P_A = num_A/total_num;
    
            ### find P(B)
            # num_1_week_old = len(first_frame_1_week)
            # P_B = num_1_week_old/total_num
                            
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



""" Predict age based on size??? 

        what is probability that cell is 1 week old P(B) given that it is size X P(A) == P(B|A) == P(A and B) / P(A)
        P(A) == prob cell is ABOVE size X
        P(B) == prob cell 1 week old
        P(A and B) == prob cell is at least 1 week old AND above size X
"""
""" DOUBLE CHECK THIS PROBABILITY CALCULATION!!!"""        
def probability_curves(sav_dir, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week, thresh_range, ax_title_size, leg_size, figsize=(6, 5)):

    all_probs, all_sizes = find_prob_given_size(first_frame_1_week, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week,
                                                thresh_range=thresh_range)
    plt.figure(figsize=figsize); plt.plot(all_sizes, all_probs)
    ax = plt.gca()
    plt.xlabel('Size threshold ($\u03bcm^3$)', fontsize=ax_title_size); plt.ylabel('Probability', fontsize=ax_title_size)
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)            
                

    all_probs, all_sizes = find_prob_given_size(first_frame_2_week, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week,
                                                thresh_range=thresh_range)
    plt.plot(all_sizes, all_probs)
    
    all_probs, all_sizes = find_prob_given_size(first_frame_3_week, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week,
                                                thresh_range=thresh_range)
    plt.plot(all_sizes, all_probs)
    
    all_probs, all_sizes = find_prob_given_size(np.concatenate((first_frame_1_week, first_frame_2_week)), first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week,
                                                thresh_range=thresh_range)
    plt.plot(all_sizes, all_probs)
    
    all_probs, all_sizes = find_prob_given_size(np.concatenate((first_frame_1_week, first_frame_2_week, first_frame_3_week)), first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week,
                                                thresh_range=thresh_range)
    plt.plot(all_sizes, all_probs)
    plt.legend(['1 week old', '2 weeks old', '3 weeks old', '< 2 weeks old', '< 3 weeks old'], fontsize=leg_size)
    plt.tight_layout()
    plt.savefig(sav_dir + 'probabilities.png')


""" Plot density and volume of cells by depth per frame """

def plot_density_and_volume(tracked_cells_df, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, ax_title_size, leg_size, scaled_vol=1, name = '', figsize=(6,5), plot=1):
    plt.close('all')
    all_total_dists = []
    all_total_vols = []
    all_total_z = []
    all_new_dists = [] 
    all_term_dists = []
    all_new_vol = [] 
    all_term_vol = [] 
    all_new_z = [] 
    all_term_z = []
    
    for frame_num, cells in enumerate(new_cells_per_frame):
        
        if len(np.where(tracked_cells_df.FRAME == frame_num)[0]) == 0:
             continue
         
        total_dists, total_z, new_dists, new_z  = get_sizes_and_z_from_cell_list(tracked_cells_df, cells, frame_num, scale_xy, scale_z, neighbors, density=1, scaled_vol=scaled_vol)
        
        if plot:
            plt.figure(neighbors + frame_num, figsize=figsize); 
            ax = plt.gca()
            plt.scatter(total_z, total_dists, s=5, marker='o', color='k');
            plt.scatter(new_z, new_dists, s=8, marker='o', color='limegreen');
            plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=14) 
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            plt.xlim(0, 400); plt.ylim(30, 200) 
            ax.legend(['stable', 'new'], fontsize=leg_size, loc='upper right')
            plt.tight_layout()
            plt.savefig(sav_dir + name + '_DENSITY_new' + str(neighbors + frame_num) + '.png')     
            
        all_total_dists.append(total_dists)
        all_total_z.append(total_z)
        
        all_new_dists.append(new_dists)
        all_new_z.append(new_z)
        
        
    """ Plt density of TERMINATED cells vs. depth """
    for frame_num, cells in enumerate(terminated_cells_per_frame):
        if len(np.where(tracked_cells_df.FRAME == frame_num)[0]) == 0:
             continue
         
        
        total_dists, total_z, term_dists, term_z  = get_sizes_and_z_from_cell_list(tracked_cells_df, cells, frame_num, scale_xy, scale_z, neighbors, density=1, scaled_vol=scaled_vol)
        
        if plot:
            plt.figure(neighbors * 10 + frame_num, figsize=figsize); 
            ax = plt.gca()
            plt.scatter(total_z, total_dists, s=5, marker='o', color='k');
            plt.scatter(term_z, term_dists, s=8, marker='o', color='r');
            plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=14) 
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            plt.xlim(0, 400); plt.ylim(30, 200)
            ax.legend(['stable', 'dying'], fontsize=leg_size, loc='upper right')
            plt.tight_layout()
            plt.savefig(sav_dir +  name + '_DENSITY_term' + str(neighbors + frame_num) + '.png')              
    
        all_term_dists.append(term_dists)
        all_term_z.append(term_z)


    """ Plt VOLUME of NEW cells vs. depth """
    for frame_num, cells in enumerate(new_cells_per_frame):

        if len(np.where(tracked_cells_df.FRAME == frame_num)[0]) == 0:
             continue
         
        total_vols, total_z, new_vols, new_z  = get_sizes_and_z_from_cell_list(tracked_cells_df, cells, frame_num, scale_xy, scale_z, neighbors, density=0, scaled_vol=scaled_vol)
        
        if plot:
            plt.figure(neighbors * 100 + frame_num, figsize=figsize); 
            ax = plt.gca()
            plt.scatter(total_z, total_vols, s=5, marker='o', color='k');
            plt.scatter(new_z, new_vols, s=8, marker='o', color='limegreen');
            plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('volume ($um^3$)', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=14) 
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            plt.xlim(0, 400); plt.ylim(30, 10000)
            ax.legend(['stable', 'new'], fontsize=leg_size, loc='upper right')            
            plt.tight_layout()
            plt.savefig(sav_dir +  name + '_VOLUME_new' + str(neighbors + frame_num) + '.png')                
        
        all_total_vols.append(total_vols)
        all_new_vol.append(new_vols) 
           
    
    """ Plt VOLUME of TERMINATED cells vs. depth """
    for frame_num, cells in enumerate(terminated_cells_per_frame):

        if len(np.where(tracked_cells_df.FRAME == frame_num)[0]) == 0:
             continue
       
        total_vols, total_z, term_vols, term_z  = get_sizes_and_z_from_cell_list(tracked_cells_df, cells, frame_num, scale_xy, scale_z, neighbors, density=0, scaled_vol=scaled_vol)
 
        if plot:   
            plt.figure(neighbors * 1000 + frame_num, figsize=figsize); 
            ax = plt.gca()
            plt.scatter(total_z, total_vols, s=5, marker='o', color='k');
            plt.scatter(term_z, term_vols, s=8, marker='o', color='r');
            plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('volume ($um^3$)', fontsize=ax_title_size)
            plt.title('frame num: ' + str(frame_num), fontsize=14) 
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            ax.legend(['stable', 'dying'], fontsize=leg_size, loc='upper right')
            plt.xlim(0, 400); plt.ylim(30, 8000)
            plt.tight_layout()
            plt.savefig(sav_dir +  name + '_VOLUME_term' + str(neighbors + frame_num) + '.png')              
            
        all_term_vol.append(term_vols)

    return all_total_dists, all_total_vols, all_total_z, all_new_dists, all_term_dists, all_new_vol, all_term_vol, all_new_z, all_term_z
            


def plot_DENSITY_VOLUME_GRAPHS(all_total_dists, all_total_vols, all_total_z, all_new_dists, all_term_dists, all_new_vol, all_term_vol, all_new_z, all_term_z, sav_dir, neighbors, ax_title_size, leg_size, name = '', figsize=(6,5)):
    plt.close('all')
    
    """ Plot density of NEW cells """
    for idx, total_dists in enumerate(all_total_dists):
        
        total_z = all_total_z[idx]
        new_z = all_new_z[idx]
        
        new_dists = all_new_dists[idx]
        
        plt.figure(neighbors + idx, figsize=figsize); 
        ax = plt.gca()
        plt.scatter(total_z, total_dists, s=5, marker='o', color='k');
        plt.scatter(new_z, new_dists, s=8, marker='o', color='limegreen');
        plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
        plt.title('frame num: ' + str(idx), fontsize=14) 
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        plt.xlim(0, 400); plt.ylim(30, 200) 
        ax.legend(['stable', 'new'], fontsize=leg_size, loc='upper right')
        plt.tight_layout()
        plt.savefig(sav_dir + name + '_DENSITY_new_' + str(neighbors + idx) + '.png')     

        
        
    """ Plt density of TERMINATED cells vs. depth """
    for idx, total_dists in enumerate(all_total_dists):
       
        total_z = all_total_z[idx]
        term_z = all_term_z[idx]
        
        term_dists = all_term_dists[idx]       
       
        plt.figure(neighbors * 10 + idx, figsize=figsize); 
        ax = plt.gca()
        plt.scatter(total_z, total_dists, s=5, marker='o', color='k');
        plt.scatter(term_z, term_dists, s=8, marker='o', color='r');
        plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('sparsity', fontsize=ax_title_size)
        plt.title('frame num: ' + str(idx), fontsize=14) 
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        plt.xlim(0, 400); plt.ylim(30, 200)
        ax.legend(['stable', 'dying'], fontsize=leg_size, loc='upper right')
        plt.tight_layout()
        plt.savefig(sav_dir +  name + '_DENSITY_term_' + str(neighbors + idx) + '.png')              

  

    """ Plt VOLUME of NEW cells vs. depth """
    for idx, total_vols in enumerate(all_total_vols):
 
        total_z = all_total_z[idx]
        new_z = all_new_z[idx]
        
        new_vols = all_new_vol[idx]
        
        
        plt.figure(neighbors * 100 + idx, figsize=figsize); 
        ax = plt.gca()
        plt.scatter(total_z, total_vols, s=5, marker='o', color='k');
        plt.scatter(new_z, new_vols, s=8, marker='o', color='limegreen');
        plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('volume ($um^3$)', fontsize=ax_title_size)
        plt.title('frame num: ' + str(idx), fontsize=14) 
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        plt.xlim(0, 400); plt.ylim(30, 10000)
        ax.legend(['stable', 'new'], fontsize=leg_size, loc='upper right')            
        plt.tight_layout()
        plt.savefig(sav_dir +  name + '_VOLUME_new_' + str(neighbors + idx) + '.png')                
    

    """ Plt VOLUME of TERMINATED cells vs. depth """
    for idx, total_vols in enumerate(all_total_vols):

        total_z = all_total_z[idx]
        term_z = all_term_z[idx]
        
        term_vols = all_term_vol[idx]  
        
        
        plt.figure(neighbors * 1000 + idx, figsize=figsize); 
        ax = plt.gca()
        plt.scatter(total_z, total_vols, s=5, marker='o', color='k');
        plt.scatter(term_z, term_vols, s=8, marker='o', color='r');
        plt.xlabel('depth (\u03bcm)', fontsize=ax_title_size); plt.ylabel('volume ($um^3$)', fontsize=ax_title_size)
        plt.title('frame num: ' + str(idx), fontsize=14) 
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        ax.legend(['stable', 'dying'], fontsize=leg_size, loc='upper right')
        plt.xlim(0, 400); plt.ylim(30, 8000)
        plt.tight_layout()
        plt.savefig(sav_dir +  name + '_VOLUME_term_' + str(neighbors + idx) + '.png')              



    return all_total_dists, all_total_z, all_new_dists, all_term_dists, all_new_vol, all_term_vol, all_new_z, all_term_z
