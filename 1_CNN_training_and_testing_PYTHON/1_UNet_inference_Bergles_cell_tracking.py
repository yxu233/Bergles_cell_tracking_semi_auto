# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
"""

import tensorflow as tf
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
from UNet import *
from UNet_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

from csbdeep.internals import predict
from tifffile import *
import tkinter
from tkinter import filedialog


""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


"""  Network Begins: """
#s_path = './Checkpoints_for_GITHUB/'
s_path = './Checkpoints_new_training_BEST/'

overlap_percent = 0.5

""" TO LOAD OLD CHECKPOINT """
sess = tf.InteractiveSession()
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'*.meta'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = last_file.split('check_')[0] + 'check_' +  checkpoint 
    
saver = tf.train.import_meta_graph(checkpoint + '.meta')
saver.restore(sess, checkpoint)


graph = tf.get_default_graph()
x_3D = graph.get_tensor_by_name("3D_x:0")
shape = x_3D.get_shape().as_list()
input_size = shape[2]
depth = shape[1]

crop_size = int(input_size/2)
z_size = depth

y_3D_ = graph.get_tensor_by_name('3D_CorrectLabel:0')
y_shape = y_3D_.get_shape().as_list()
num_truth_class = y_shape[-1]
#weight_matrix_3D = graph.get_tensor_by_name('weighted_labels:0')
softMaxed = graph.get_tensor_by_name('Softmaxed:0')
training = graph.get_tensor_by_name('training:0')


print("Input size is: " + str(input_size))
print("Input depth is: " + str(depth))
# """ load mean and std """  
input_path = './normalize/'
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= input_path,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
    print('Do you want to select another folder? (y/n)')
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_analytic_results'

    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


    #images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    #images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    #examples = [dict(input=i,truth=i.replace('_single_channel.tif','_truth.tif'), ilastik=i.replace('_single_channel.tif','_single_Object Predictions_.tiff')) for i in images]


    #images = glob.glob(os.path.join(input_path,'*_RAW_REGISTERED.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    #images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    #examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED.tif','_TRUTH_REGISTERED.tif'), ilastik=i.replace('_RAW_REGISTERED.tif','_single_Object Predictions_.tiff')) for i in images]
        
                
        
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
            
            input_name = examples[i]['input']            
            input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
   
            """ Analyze each block with offset in all directions """
            
            # Display the image
            #max_im = plot_max(input_im, ax=0)
            
            print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
            plot_max(input_im)
            segmentation = UNet_inference_by_subparts(input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                                                      mean_arr=mean_arr, std_arr=std_arr, num_truth_class=num_truth_class,
                                                      x_3D=x_3D, y_3D_=y_3D_, training=training, softMaxed=softMaxed, skip_top=1)
           
            segmentation[segmentation > 0] = 255
            plot_max(segmentation)
            filename = input_name.split('\\')[-1]
            filename = filename.split('.')[0:-1]
            filename = '.'.join(filename)
            

            segmentation = np.asarray(segmentation, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            segmentation[segmentation > 0] = 1
            
            input_im = np.asarray(input_im, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im)
            
            
            """ Load in truth data for comparison!!! sens + precision """
            # truth_name = examples[i]['truth']
            
            # truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
            # truth_im[truth_im > 0] = 1                                   
            
            # truth_im_cleaned = clean_edges(truth_im, extra_z=1, extra_xy=3, skip_top=1)
                                             
            # TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(segmentation, truth_im_cleaned, size_limit=5)

                  
            # plot_max(truth_im_cleaned)
            # plot_max(cleaned_seg)
            

            # if TP + FN == 0: TP;
            # else: sensitivity = TP/(TP + FN);     # PPV
                   
            # if TP + FP == 0: TP;
            # else: precision = TP/(TP + FP);     # precision

            # print(filename)
            # print(str(sensitivity))
            # print(str(precision))
            
            # truth_im_cleaned = np.asarray(truth_im_cleaned, np.uint8)
            # imsave(sav_dir + filename + '_' + str(int(i)) +'_truth_im_cleaned.tif', truth_im_cleaned)            
            
            """ Compare with ilastik () if you want to """
            # ilastik_compare = 0
            # if ilastik_compare:
                 
            #      """ Load in truth data for comparison!!! sens + precision """
            #      ilastik_name = examples[i]['ilastik']

            #      ilastik_im = open_image_sequence_to_3D(ilastik_name, width_max='default', height_max='default', depth='default')
            #      ilastik_im[ilastik_im > 0] = 1                                   
                 

            #      ilastik_im_cleaned = clean_edges(ilastik_im, depth_im, w=width, h=height, extra_z=1, extra_xy=3)
                                                   
            #      TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(ilastik_im, truth_im_cleaned, size_limit=10)
                                    
                               
            #      ilastik_im_cleaned = np.asarray(ilastik_im_cleaned, np.uint8)
            #      imsave(sav_dir + filename + '_' + str(int(i)) +'_ilastik_cleaned.tif', ilastik_im_cleaned)
                 
                      
            #      plot_max(ilastik_im_cleaned)
                 
            #      if TP + FN == 0: TP;
            #      else: sensitivity = TP/(TP + FN);     # PPV
                        
            #      if TP + FP == 0: TP;
            #      else: precision = TP/(TP + FP);     # precision
        
            #      print(str(sensitivity))
            #      print(str(precision))                                         