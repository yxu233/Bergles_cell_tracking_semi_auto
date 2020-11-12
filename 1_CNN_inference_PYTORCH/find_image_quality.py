#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:45:32 2020

@author: user
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

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


""" Import network """
#unet = UNet_online()

"""  Network Begins: """
#s_path = './Checkpoints_for_GITHUB/'
#s_path = './(4) Checkpoints_PYTORCH_5x5_256x64_no_CONVTRANSP_matched_no_DILATION_COMPLEX/'
#s_path = './(12) Checkpoints_TITAN_5x5_256x64_NO_transforms_AdamW_spatial/'
#s_path = './(18) Checkpoints_TITAN_NO_transforms_AdamW_batch_norm_SPATIAL/' 
#s_path = './(16) Checkpoints_TITAN_YES_transforms_AdamW_SLOWER_switchable_BN/'

s_path = './(19) Checkpoints_TITAN_NO_transforms_AdamW_batch_norm_CLEAN_DATA/'


#s_path = './(20) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA/'

s_path = './(21) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA_LARGE_NETWORK/'




""" AMOUNT OF EDGE TO ELIMINATE 


    scaling???
"""


import argparse
from pyimq import filters, script_options, utils, myimage
def get_quality_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
                    "image quality ranking software"
    )

    parser.add_argument(
        "--file",
        help="Defines a path to the image files",
        default=None
    )
    parser.add_argument(
        "--working-directory",
        dest="working_directory",
        help="Defines the location of the working directory",
        default="/home/sami/Pictures/Quality"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "directory", "analyze", "plot"],
        action="append",
        help="The argument containing the functionality of the main program"
             "You can concatenate actions by defining multiple modes in a"
             "single command, e.g. --mode=directory --mode=analyze"
    )
    # Parameters for controlling the way plot functionality works.
    parser.add_argument(
        "--result",
        default="average",
        choices=["average", "fskew", "ientropy", "fentropy", "fstd",
                 "fkurtosis", "fpw", "fmean", "icv", "meanbin"],
        help="Tell how you want the results to be calculated."
    )
    parser.add_argument(
        "--npics",
        type=int,
        default=9,
        help="Define how many images are shown in the plots"
    )

    parser = filters.get_common_options(parser)
    parser = myimage.get_options(parser)
    return parser.parse_args(arguments)




overlap_percent = 0.5
input_size = 256
depth = 64
num_truth_class = 2


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
    sav_dir = input_path + '/' + foldername + '_output_PYTORCH_RETRAINED_105834'

    """ For testing ILASTIK images """
    # images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


    images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_single_channel.tif','_truth.tif'), ilastik=i.replace('_single_channel.tif','_single_Object Predictions_.tiff')) for i in images]


    # images = glob.glob(os.path.join(input_path,'*_RAW_REGISTERED.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED.tif','_TRUTH_REGISTERED.tif'), ilastik=i.replace('_RAW_REGISTERED.tif','_single_Object Predictions_.tiff')) for i in images]


    # images = glob.glob(os.path.join(input_path,'*_RAW_REGISTERED_substack_1_110.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED_substack_1_110.tif','_TRUTH_REGISTERED_substack_1_11_m_ilastik.tif'), ilastik=i.replace('_RAW_REGISTERED.tif','_single_Object Predictions_.tiff')) for i in images]

     

     
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
    for i in range(len(examples)):
         
    
        
         """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
         with torch.set_grad_enabled(False):  # saves GPU RAM            
            input_name = examples[i]['input']            
            input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')

    
            from skimage import filters

    
            """ Find BRISQUE quality """
            plt.figure()
            append_mean_SNR = []
            for val in range(10, 240, 10):
                #val = 100
                all_SNR = []
                for depth in range(0, len(input_im) - 33, 33):
                
                    first_slices= input_im[depth:depth + 33, ...]
                    max_first = plot_max(first_slices, ax=0, plot=0)
                    im = Image.fromarray(np.uint8(max_first))
                    
                    xres = 0.83
                    yres = 0.83
                    
                    signal = np.mean(np.where(max_first > val))
                    noise = np.std(np.where(max_first < val))
                    
                    SNR = 10 * math.log10(signal/noise)
                    
                    all_SNR.append(SNR)
                    
                plt.plot(all_SNR)
                append_mean_SNR.append(all_SNR)
                   
            mean_SNR = np.nanmean(append_mean_SNR, axis=0)
            #save_snr = mean_SNR
            
            
            zzz
            
            
            """ Set globally """
            plt.rc('xtick',labelsize=16)
            plt.rc('ytick',labelsize=16)
            ax_title_size = 18
            leg_size = 16
            plt.rcParams['figure.dpi'] = 300
            
            """ Stop here and run again with low SNR"""
            plt.figure(figsize=(5,4));
            x_axis = [100, 200, 300, 400]
            plt.plot(x_axis, save_snr[0:len(x_axis)])
            plt.plot(x_axis, mean_SNR[0:len(x_axis)])
            
            plt.ylim([0, 4])
            
            ax = plt.gca()


            ax.legend(['optimal quality', 'degraded quality'], frameon=False, fontsize=leg_size, loc='upper right')

            #plt.xlabel("proportion of tracks", fontsize=14)
            plt.xlabel('Depth of max projection', fontsize=ax_title_size)
            plt.ylabel("SNR", fontsize=ax_title_size)
            #plt.yticks(np.arange(0, max(errs)+1, 5))
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            plt.tight_layout()
            plt.savefig(sav_dir + 'SNR_comparison' + '.png')                
            zzz
            #zzz
  
    
  
    