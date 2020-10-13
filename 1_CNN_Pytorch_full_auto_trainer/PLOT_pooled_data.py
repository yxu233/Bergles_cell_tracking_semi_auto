#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:26:05 2020

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:47:35 2020

@author: user
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

from matlab_crop_function import *
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
from functions_cell_track_auto import *
from skimage.transform import rescale, resize, downscale_local_mean


from PLOT_FIGURES_functions import *


import pandas as pd
import scipy.stats as sp
import seaborn as sns


lowest_z_depth = 180;
crop_size = 160
z_size = 32
num_truth_class = 2
min_size = 10
both = 0


MATLAB = 0

scale_xy = 0.83
scale_z = 3
truth = 1


exclude_side_px = 10

min_size = 100;
upper_thresh = 800;

""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"


all_norm_tots = []; all_norm_new = [];
all_norm_t32 = []; all_norm_n32 = [];
all_norm_t65 = []; all_norm_n65 = [];
all_norm_t99 = []; all_norm_n99 = [];
all_norm_t132 = []; all_norm_n132 = [];
all_norm_t165 = []; all_norm_n165 = [];


all_tracked_cells_df = [];


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

### TRACK WHICH FOLDERS TO POOL
folder_pools = np.zeros(len(list_folder))
    
for fold_idx, input_path in enumerate(list_folder):
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_3'


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
    
    
    
    """Load in uncleaned array  """
    tracked_cells_df = pd.read_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         
               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
        
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
        elif np.any(X_cur_cell > width_tmp - exclude_side_px) or np.any(X_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            num_edges += 1
            
        
        elif np.any(Y_cur_cell > height_tmp - exclude_side_px) or np.any(Y_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
            num_edges += 1
                

    """ Also remove by min_size """
    num_small = 0; real_saved = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
                
        idx = np.where(tracked_cells_df.SERIES == cell_num)
        all_lengths = []
        small_bool = 0;
        for iter_idx, cell_obj in enumerate(tracked_cells_df.iloc[idx].coords):
            if len(cell_obj) < min_size:  
                
                small_bool = 1
            
            ### exception, spare if large cell at any point
            if len(cell_obj) > upper_thresh:  
                small_bool = 0
                real_saved += 1
                break;
        
        if small_bool:
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])   ### DROPS ENTIRE CELL SERIES
            num_small += 1

                
    """  Save images in output """
    input_name = examples[0]['input']
    filename = input_name.split('/')[-1]
    filename = filename.split('.')[0:-1]
    filename = '.'.join(filename)
    
    for frame_num, im_dict in enumerate(examples):
         
            output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
            im = convert_matrix_to_multipage_tiff(output_frame)
            imsave(sav_dir + filename + '_' + str(frame_num) + '_output_CLEANED.tif', im)
         
         
          # output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
          # im = convert_matrix_to_multipage_tiff(output_frame)
          # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)


            # output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=0)
            # im = convert_matrix_to_multipage_tiff(output_frame)
            # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_TERMINATED.tif', im)


            output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=1)
            im = convert_matrix_to_multipage_tiff(output_frame)
            imsave(sav_dir + filename + '_' + str(frame_num) + '_output_NEW.tif', im)
         
            
         
            
    """ SCALE CELL COORDS to true volume 
    """
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
        
    """ Create copy """
    all_tracked_cells_df.append(tracked_cells_df)

            
                
    """ If filename contains 235 or 246, then must +1 to timeframes after baseline, because week 2 skipped """
    if '235' in foldername or '264' in foldername:
        
        for idx in range(len(tracked_cells_df)):
            
            cell = tracked_cells_df.iloc[idx]
            if cell.FRAME >= 1:
                new_val = cell.FRAME + 1
                cell.FRAME = new_val
                tracked_cells_df.iloc[idx] = cell
            

    """ Normalize to 0 um using vertices """
    
    if '235' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 0; z_y = 27 - 2  ### upper right corner, y=0
        z_x = 0; z_xy = 31 - 2
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)

    elif '037' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 18 - 0; z_y = 18 - 0   ### upper right corner, y=0
        z_x = 2 - 0; z_xy = 7 - 0
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    

    elif '030' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 16 + 2; z_y = 30 + 2   ### upper right corner, y=0
        z_x = 6 + 2; z_xy = 26 + 2

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)  
        
        folder_pools[fold_idx] = 1;   ### TO POOL LATER

    elif '033' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 12 - 0; z_y = 21 - 0   ### upper right corner, y=0
        z_x = 5 - 0; z_xy = 16 - 0

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    

        folder_pools[fold_idx] = 1;   ### TO POOL LATER

    elif '097' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 10 + 1; z_y = 19 + 1   ### upper right corner, y=0
        z_x = 2 + 1; z_xy = 10 + 1

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)   
        
        folder_pools[fold_idx] = 2;   ### TO POOL LATER

    elif '099' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 0 + 3; z_y = 15 + 3   ### upper right corner, y=0
        z_x = 0 + 3; z_xy = 15 + 3

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)  
        
        folder_pools[fold_idx] = 2;   ### TO POOL LATER
        
        
    """ Scale Z to mesh """
    #m = np.max(input_im, axis=-1)
    print('scaling cell Z to mesh')
    for idx in range(len(tracked_cells_df)):
        
        cell = tracked_cells_df.iloc[idx]
    
        x = cell.X
        y = cell.Y
        z = cell.Z
        
        
        ### ENSURE ORDER OF XY IS CORRECT HERE!!!
        scale = mesh[int(y), int(x)]
    
        new_z = z - scale
        
        ### DEBUG
        #m[int(y), int(x)] = 0
        cell.Z = new_z
    
        tracked_cells_df.iloc[idx] = cell
    
    #plt.figure(); plt.imshow(m)        
                
    """ Set globally """
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    ax_title_size = 18
    leg_size = 14

    """ plot timeframes """
    norm_tots_ALL, norm_new_ALL = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_', depth_lim_lower=0, depth_lim_upper=120, ax_title_size=ax_title_size, leg_size=leg_size)
    
    """ 
        Also split by depths
    """
    norm_tots_32, norm_new_32 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_0-32', depth_lim_lower=0, depth_lim_upper=32, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_65, norm_new_65 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_33-65', depth_lim_lower=33, depth_lim_upper=65, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_99, norm_new_99 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_66-99', depth_lim_lower=66, depth_lim_upper=99, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_132, norm_new_132 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_100-132', depth_lim_lower=100, depth_lim_upper=132, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_165, norm_new_165 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_133-165', depth_lim_lower=133, depth_lim_upper=165, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    

    all_norm_tots.append(norm_tots_ALL); all_norm_new.append(norm_new_ALL);
    all_norm_t32.append(norm_tots_32) ; all_norm_n32.append(norm_new_32);
    all_norm_t65.append(norm_tots_65); all_norm_n65.append(norm_new_65);
    all_norm_t99.append(norm_tots_99); all_norm_n99.append(norm_new_99);
    all_norm_t132.append(norm_tots_132); all_norm_n132.append(norm_new_132);
    all_norm_t165.append(norm_tots_165); all_norm_n165.append(norm_new_165);


    
    """ Per-timeseries analysis """
    """ 
        Also do density analysis of where new cells pop-up???
    """        
    analyze = 1;    
    if analyze == 1:
        neighbors = 10
        
        new_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)) + 1)]
        terminated_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)) + 1)]
        for cell_num in np.unique(tracked_cells_df.SERIES):
            
            frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
            
            beginning_frame = np.min(frames_cur_cell)
            if beginning_frame > 0:   # skip the first frame
                new_cells_per_frame[beginning_frame].append(cell_num)
                        
            term_frame = np.max(frames_cur_cell)
            if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
                terminated_cells_per_frame[term_frame].append(cell_num)
            
    
        ### loop through each frame and all the new cells and find "i.... i + n" nearest neighbors        
        """ Plt density of NEW cells vs. depth """
        scaled_vol = 1
        plot_density_and_volume(tracked_cells_df, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, ax_title_size, leg_size, scaled_vol=scaled_vol)

       
        """ 
            Plot size decay for each frame STARTING from recovery
        """     
        plt.close('all'); 
        plot_size_decay_in_recovery(tracked_cells_df, sav_dir, start_frame=3, end_frame=8, min_survive_frames=3, use_scaled=0, y_lim=8000, ax_title_size=ax_title_size)
    


        """ Plot scatters of each type:
                - control/baseline day 1 ==> frame 0
                        ***cuprizone ==> frame 4
                - 1 week after cupr
                - 2 weeks after cupr
                - 3 weeks after cupr 
            
            """
        first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week = plot_size_scatters_by_recovery(tracked_cells_df, sav_dir, start_frame=3, end_frame=8, min_survive_frames=3, use_scaled=1, y_lim=10000, ax_title_size=ax_title_size)

        
        
        """ Predict age based on size??? 
        
                what is probability that cell is 1 week old P(B) given that it is size X P(A) == P(B|A) == P(A and B) / P(A)
                P(A) == prob cell is ABOVE size X
                P(B) == prob cell 1 week old
                P(A and B) == prob cell is at least 1 week old AND above size X
        """
        """ DOUBLE CHECK THIS PROBABILITY CALCULATION!!!"""
        
        upper_r = 8000
        lower_r = 0
        step = 100
        thresh_range = [lower_r, upper_r, step]
        probability_curves(sav_dir, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week, thresh_range, ax_title_size, leg_size)





""" Also pool things from WITHIN the same experiment """
folder_pools




zzz

max_week = -1
plot_pooled_trends(all_norm_tots, all_norm_new, ax_title_size, sav_dir, add_name='OUTPUT_overall_')
plot_pooled_trends(all_norm_t32, all_norm_n32, ax_title_size, sav_dir, add_name='OUTPUT_overall_0-32', max_week=max_week)
plot_pooled_trends(all_norm_t65, all_norm_n65, ax_title_size, sav_dir, add_name='OUTPUT_overall_32-65', max_week=max_week)
plot_pooled_trends(all_norm_t99, all_norm_n99, ax_title_size, sav_dir, add_name='OUTPUT_overall_65_99', max_week=max_week)
plot_pooled_trends(all_norm_t132, all_norm_n132, ax_title_size, sav_dir, add_name='OUTPUT_overall_99-132', max_week=max_week)
plot_pooled_trends(all_norm_t165, all_norm_n165, ax_title_size, sav_dir, add_name='OUTPUT_overall_132-165', max_week=max_week)



""" Pool all tracked_cells_df and add np.max() so series numbers don't overlap! """
pooled_tracks = all_tracked_cells_df[0]
for idx, tracks in enumerate(all_tracked_cells_df):
    
    max_series = np.max(pooled_tracks.SERIES)
    
    
    if idx > 0:
    
        tracks.SERIES = tracks.SERIES + max_series    
        pooled_tracks = pd.concat([pooled_tracks, tracks])
    
    
tracked_cells_df = pooled_tracks


""" 
    Also do density analysis of where new cells pop-up???

"""        
analyze = 1;    
if analyze == 1:
    neighbors = 10
    
    new_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)) + 1)]
    terminated_cells_per_frame = [[] for _ in range(len(np.unique(tracked_cells_df.FRAME)) + 1)]
    for cell_num in np.unique(tracked_cells_df.SERIES):
        
        frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
        
        beginning_frame = np.min(frames_cur_cell)
        if beginning_frame > 0:   # skip the first frame
            new_cells_per_frame[beginning_frame].append(cell_num)
                    
        term_frame = np.max(frames_cur_cell)
        if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
            terminated_cells_per_frame[term_frame].append(cell_num)
        

    ### loop through each frame and all the new cells and find "i.... i + n" nearest neighbors        
    """ Plt density of NEW cells vs. depth """
    scaled_vol = 1
    plot_density_and_volume(tracked_cells_df, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, ax_title_size, leg_size, scaled_vol=scaled_vol)

        
    """ 
        Plot size decay for each frame STARTING from recovery
    """     
    plt.close('all'); 
    plot_size_decay_in_recovery(tracked_cells_df, sav_dir, start_frame=3, end_frame=8, min_survive_frames=3, use_scaled=0, y_lim=8000, ax_title_size=ax_title_size)


    """ Plot scatters of each type:
            - control/baseline day 1 ==> frame 0
                    ***cuprizone ==> frame 4
            - 1 week after cupr
            - 2 weeks after cupr
            - 3 weeks after cupr 
        
        """
    first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week = plot_size_scatters_by_recovery(tracked_cells_df, sav_dir, start_frame=3, end_frame=8, min_survive_frames=3, use_scaled=1, y_lim=10000, ax_title_size=ax_title_size)

    
    
    """ Predict age based on size??? 
    
            what is probability that cell is 1 week old P(B) given that it is size X P(A) == P(B|A) == P(A and B) / P(A)
            P(A) == prob cell is ABOVE size X
            P(B) == prob cell 1 week old
            P(A and B) == prob cell is at least 1 week old AND above size X
    """
    """ DOUBLE CHECK THIS PROBABILITY CALCULATION!!!"""
    
    upper_r = 8000
    lower_r = 0
    step = 100
    thresh_range = [lower_r, upper_r, step]
    
    
    """    ### SHOULD BE COMPARED TO CELLS ON LATER TIMEFRAMES???? IN TERMS OF SIZE???
    ### because baseline may have some new cells as well!!!
    
                ***compared with cells/themselves that have been around for > 8 weeks???
                
                
                
                ****RETROSPECTIVELY CAN SEE IF NEW CELLS ARE INJURED FASTER AT EARLIER TIMEPOINTS by cuprizone???
                
                
                ***plot size vs. death time
                
                
                    ==> example in 030!!!
                
                
    
    """    
    
    probability_curves(sav_dir, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week, thresh_range, ax_title_size, leg_size)


            
    """ vs. cells in control condition???
    
    
    
    """
            
        
        
   
    
    
    """
        Density (histogram of cells) by depth on frame 0 (baseline) 
    """
    plt.rcParams['figure.figsize'] = [12.0, 3.0]
    #plt.rcParams['figure.dpi'] = 140        
    all_z_frame_0 = tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([0])].Z
    
    plt.figure();
    plt.hist(all_z_frame_0 * scale_z, bins=50, color='gray')
    plt.xlabel('Depth (\u03bcm)', fontsize=ax_title_size)
    plt.ylabel('Number of cells', fontsize=ax_title_size)
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    plt.tight_layout()        
    plt.savefig(sav_dir + 'density along depth.png')
    
    mpl.rcParams['figure.figsize'] = [8.0, 6.0]   ### restores default size
    

        
        
        
        
        
    