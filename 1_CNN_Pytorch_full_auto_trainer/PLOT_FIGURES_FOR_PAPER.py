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
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762'


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
    
    
    """ Get truth from .csv as well """
    truth = 1
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'; scale = 0
         #truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv';  scale = 1  # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv';    # gets hazy at the end
         truth_name = '680_syGlass_10x.csv'     

         #truth_name = '235_syGlass_10x.csv';    ### CUPRIZONE

         # truth_name = '264_syGlass_10x.csv'          
         
         # truth_name = 'a5151201-r097_syGlass_20x.csv';
         
         # truth_name = 'a2151201-r037_syGlass_20x.csv';
         

         truth_name = 'a4151201-r033_syGlass_20x.csv';

         #truth_name = 'MOBPF_190106w_5_cuprBZA_10x.tif - T=0_650_syGlass_10x.csv'   # well registered and clean window except for single frame
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})    
    
    
    
    
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
      
                        
      
        
      
        
      
        
      
        
    """ Set plot size and DPI """
    #plt.rcParams['figure.figsize'] = [6.0, 4.0]
    #plt.rcParams['figure.dpi'] = 140

    
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
    
    if MATLAB:
        #MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO_MATLAB.csv'
        #MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'; ### CUPRIZONE
        #MATLAB_name = 'output.csv'
        MATLAB_name = '680_MATLAB_output.csv'  
        MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
        plot_timeframes(MATLAB_auto_array, sav_dir, add_name='MATLAB_', depth_lim_lower=0, depth_lim_upper=120, ax_title_size=ax_title_size, leg_size=leg_size)
        


    if truth:
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_', depth_lim_lower=0, depth_lim_upper=120, ax_title_size=ax_title_size, leg_size=leg_size)
        
        ### by every 100 um
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_0-32', depth_lim_lower=0, depth_lim_upper=32, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_33-65', depth_lim_lower=33, depth_lim_upper=65, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_66-99', depth_lim_lower=66, depth_lim_upper=99, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_100-132', depth_lim_lower=100, depth_lim_upper=132, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
        plot_timeframes(truth_array, sav_dir, add_name='TRUTH_133-165', depth_lim_lower=133, depth_lim_upper=165, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
        

    """ Also plot rate of loss??? """
    """ Show that truth predicitions are accurate! """
    if truth:
         distances = []
         for frame_num in range(len(examples)):
        
            empty, all_dist, dist_check, check_series = check_predicted_distances(truth_array, frame_num, crop_size, z_size, dist_error_thresh=10)
                
            if frame_num == 1:
                plt.figure(); plt.hist(all_dist, color='k')
                ax = plt.gca()
                plt.xlabel('distance of prediction to truth (px)', fontsize=ax_title_size); plt.ylabel('number of cells', fontsize=ax_title_size)
                rs = ax.spines["right"]; rs.set_visible(False)
                ts = ax.spines["top"]; ts.set_visible(False)

                plt.tight_layout()
                
                plt.savefig(sav_dir + 'prediction_accuracy_truth.png')
                #break;
            distances = np.concatenate((distances, np.asarray(all_dist)))

         num_above_10 =  len(np.where(distances > 10)[0])
         num_total = len(distances)
         print('% cells above 10 pixels: ' + str((num_above_10/num_total)  * 100))
         
         """ 
             ^^^ADD THIS NUMBER TO PAPER???
         
         """


    """ Plot compare to truth """
    if MATLAB:

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
        ax = plt.gca(); rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        ax.margins(x=0); ax.margins(y=0.02)
                
        
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
        plt.xlabel("proportion of tracks", fontsize=ax_title_size)
        plt.ylabel("track difference (# frames)", fontsize=ax_title_size)
        
        ax.legend(['CNN tracker', 'Heuristic'], fontsize=leg_size, loc='upper left')
        plt.tight_layout()
        plt.savefig(sav_dir + 'plot.png')

        
        """ Parse the old array: """
        print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
        MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
        
        """ Plot errors  """
        fig = plt.figure(); ax = plt.gca()
        
        errs = [perc_errs, perc_errs_MATLAB]
        errs_over_2 = [perc_errs_over_2, perc_errs_over_2_MATLAB]
        
        X = np.arange(len(errs))
        ax.bar(X + 0.00, errs, color = 'k', width = 0.25)
        ax.bar(X + 0.25, errs_over_2, color = 'g', width = 0.25)

        ind = np.arange(len(errs))
        width = 0.25
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('CNN tracker', 'Heuristic'))
        ax.legend(['All errors', 'Errors > 1 frame'], fontsize=leg_size)

        #plt.xlabel("proportion of tracks", fontsize=14)
        plt.ylabel("% cells tracked with errors", fontsize=ax_title_size)
        plt.yticks(np.arange(0, max(errs)+1, 5))
        rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
        plt.tight_layout()
        plt.savefig(sav_dir + 'cell_tracking_errors' + '.png')






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
        
        # import napari
        # with napari.gui_qt():
        #     viewer = napari.view_image(crop)
            
    
    """ 
        Also do density analysis of where new cells pop-up???
    
    """        
    analyze = 1;    
    if analyze == 1:
        neighbors = 10
        
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
        """ Plt density of NEW cells vs. depth """
        scaled_vol = 1
        plot_density_and_volume(tracked_cells_df, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, ax_title_size, leg_size, scaled_vol=scaled_vol)
    
            
        """ 
            Plot size decay for each frame STARTING from recovery
        """     
        # plt.close('all')
        # for frame in range(len(np.unique(tracked_cells_df.FRAME))):
        #     all_sizes_cur_frame, all_z = get_sizes_and_z_cur_frame(tracked_cells_df, frame, use_scaled=1)
                    
        #     plt.figure();
        #     for idx in range(len(all_sizes_cur_frame)):
                
        #         size = all_sizes_cur_frame[idx]
        #         z = all_z[idx]
        #         if len(size) == len(np.unique(tracked_cells_df.FRAME)) - frame:
        #             plt.plot(z, size, linewidth=1)
        #             plt.ylim(0, 10000)   
        #             plt.tight_layout()
                
         
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
                    
        """ vs. cells in control condition???
        
        
        
        """
                
            
            
       
        
        
        """
            Density (histogram of cells) by depth on frame 0 (baseline) 
        """
        plt.rcParams['figure.figsize'] = [12.0, 2.0]
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
        

        """
            Test a good looking cell and introduce noise:
                    - degrees of warping
                    - degrees of noise
                    - degrees of intensity changes
                    
                    - random translations of the second image???
                    
            ALSO TRAIN NEURAL NETWORK WITH THESE PERTURBATIONS???
                - translate the second image randomly???
            
        
        """
            
        
        
        
        
        """ 
        
            Also get fluorescence values??? ==> probably correlate strongly with size??? 
        
        
        """
        
        
            
        
        
        
        
        
        
    