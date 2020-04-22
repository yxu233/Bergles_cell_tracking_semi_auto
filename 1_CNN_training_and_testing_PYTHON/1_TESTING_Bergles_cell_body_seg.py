# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:46:29 2020

@author: tiger
"""

from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL numexpr!!!
 
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
import pickle as pickle
import os
import zipfile
import scipy
import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from data_functions_3D import *
#from split_im_to_patches import *
from UNet import *
from UNet_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


from csbdeep.internals import predict


import tkinter
from tkinter import filedialog
import os
    
truth = 0

# Initialize everything with specific random seeds for repeatability
tf.reset_default_graph() 
tf.set_random_seed(1); np.random.seed(1)


def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im[:, :])
     

def find_TP_FP_FN_from_seg(segmentation, truth_im):
     seg = segmentation      
     true = truth_im
     
     #true = truth_im[:, :, :, 1]
     #seg = seg_train[-1, :, :, :]
     #plot_max(seg)
     #plot_max(true)
     
     
     coloc = seg + true
     bw_coloc = coloc > 0
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     
     true_positive = np.zeros(np.shape(coloc))
     TP_count = 0;
     FN_count = 0;
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          if max_val > 1:
               TP_count += 1
               #for obj_idx in range(len(coords)):
               #     true_positive[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(bw_coloc)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count

             
             
     if pad_z is not 0:  output_im = output_im[pad_z:-pad_z, :, :];
     if pad_x is not 0:  output_im = output_im[:, pad_x:-pad_x, :];
     if pad_y is not 0:  output_im = output_im[:, :, pad_y:-pad_y];
     
     
     #plot_max(output_im)
     #output = np.asarray(output_im * 255, dtype=np.uint8) 
     output = output_im
     return output                        
            

""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

"""  Network Begins: """
s_path = './Checkpoints/'

# """ load mean and std """  
# mean_arr = load_pkl('', 'mean_val_VERIFIED.pkl')
# std_arr = load_pkl('', 'std_val_VERIFIED.pkl')
               
# """ SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""
# input_size = 64
# depth = 16   # ***OR can be 160
# num_truth_class = 1 # 1 for reconstruction
# multiclass = 0

# tf_size = input_size

# """ original == 60 * 320 * 320, now == 2100 * 150 * 150    so about 7.5 x larger image """
# x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name='3D_x') 
# y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name='3D_CorrectLabel')
# #weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name = 'weighted_labels')
# training = tf.placeholder(tf.bool, name='training')

# """ Creates network and cost function"""
# depth_filter = 5
# height_filter = 5
# width_filter = 5
# kernel_size = [depth_filter, height_filter, width_filter]
# y_3D, y_b_3D, L1, L2, L3, L8, L9, L9_conv, L10, L11, logits_3D, softMaxed = create_network_3D_PERCEPTUAL_LOSS_128(x_3D, y_3D_, kernel_size, training, num_truth_class)
# accuracy, jaccard, train_step, loss = costOptm_MSSSIM_MAE(y_3D, y_b_3D, logits_3D, train_rate=1e-5, epsilon=1e-8, optimizer='adam', loss_function='MAE')












resize_bool = 0

input_size = 128
depth = 32   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0

tf_size = input_size

""" original == 60 * 320 * 320, now == 2100 * 150 * 150    so about 7.5 x larger image """
x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name='3D_x') 
y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name='3D_CorrectLabel')
weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name = 'weighted_labels')
training = tf.placeholder(tf.bool, name='training')

""" Creates network and cost function"""
depth_filter = 5
height_filter = 5
width_filter = 5
kernel_size = [depth_filter, height_filter, width_filter]
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=True, optimizer='adam', multiclass=0)







sess = tf.InteractiveSession()

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
#num_check = int(num_check[0])   
#checkpoint = 'check_36400' 
saver = tf.train.Saver()
saver.restore(sess, s_path + checkpoint)
    

#root = tkinter.Tk()
#sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
#                                        title='Please select saving directory')
#sav_dir = sav_dir + '/'

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
    
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_analytic_results'
 
    """ Load filenames from zip """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED.tif','_TRUTH_REGISTERED.tif')) for i in images]
    
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
            #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
            
            input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
           
            """ NEED TO CONVERT TO np.uint8 if the original input is np.uint16!!!"""
            # NORMALIZED BECAUSE IMAGE IS uint16 ==> do same when actually running images!!!
            #input_im = np.asarray(Image.open(input_name))
            if input_im.dtype == 'uint16':
                input_im = np.asarray(input_im, dtype=np.float32)
                input_im = cv.normalize(input_im, 0, 255, cv.NORM_MINMAX)
                input_im = input_im * 255
                
            input_im = np.asarray(input_im, dtype= np.float32)
            
                    
            size_whole = input_im.shape[0]
            
            size = int(size_whole) # 4775 and 6157 for the newest one
            if resize_bool:
                size = int((size * im_scale) / 0.45) # 4775 and 6157 for the newest one
                input_im = resize_adaptive(Image.fromarray(input_im), size, method=Image.BICUBIC)
                input_im = np.asarray(input_im, dtype=np.float32)

                
            """ NORMALIZE PROPERLY HERE: """
            from csbdeep.utils import utils
            norm_im = utils.normalize(input_im, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32)
            
            
            
            
            """ Analyze each block with offset in all directions """
            
            quad_size = input_size
            quad_depth = depth
            im_size = np.shape(input_im);
            height = im_size[1];  width = im_size[2]; depth_im = im_size[0];
              
            num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);
             

            quad_idx = 1;
            
            overlap_percent = 0.10
            
            larger_input_im = np.zeros([depth_im + quad_depth, height + quad_size, width + quad_size])
            larger_input_im[0:depth_im, 0:height, 0:width] = input_im
            plot_max(larger_input_im)

            
            segmentation = np.zeros([depth_im + quad_depth, height + quad_size, width + quad_size])
            total_blocks = 0;
            for x in range(1, width + round(quad_size/2), round(quad_size - quad_size * overlap_percent)):
                 for y in range(1, height + round(quad_size/2), round(quad_size - quad_size * overlap_percent)):
                      for z in range(1, depth_im + round(quad_depth/2), round(quad_depth - quad_depth * overlap_percent)):
                          batch_x = []; batch_y = [];
                          
                          if z + quad_depth > depth_im or x + quad_size > width or y + quad_size > height:
                               continue
                          
                          quad_intensity = larger_input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size];
                          #rectangle('Position',[x y quad_size quad_size], 'Edgecolor', 'y');
                          
                          #quad_segmentations = truth_1(x:x + quad_size, y:y + quad_size, z:z + quad_depth);
                          
                          ## Rescale this so XYZ are equal distances!!!
                          #quad_intensity = imresize3(quad_intensity, [quad_size, quad_size, quad_size], 'method', 'nearest');
                          
                          #resized_seg = imresize3(quad_segmentations, [quad_size, quad_size, quad_size], 'method', 'nearest');
                          
                          
                          """ Analyze """
                          """ set inputs and truth """
                          quad_intensity = np.expand_dims(quad_intensity, axis=-1)
                          batch_x.append(quad_intensity)
                          batch_y.append(np.zeros([depth, input_size, input_size, num_truth_class]))
                    
                          """ Feed into training loop """
                          feed_dict = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:batch_y} 
                          
                          output_tile = softMaxed.eval(feed_dict=feed_dict)
                          seg_train = np.argmax(output_tile, axis=-1)
                          
                          
                          
                          """ ADD IN THE NEW SEG??? or just let it overlap??? """
                          
                          segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = seg_train
                          total_blocks += 1
                          
            plot_max(segmentation)
            filename = input_name.split('\\')[-1]
            filename = filename.split('.')[0:-1]
            filename = '.'.join(filename)
          
            imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_larger_input_im.tif', larger_input_im)
            
            
            """ Load in truth data for comparison!!! sens + precision """
            truth_name = examples[i]['truth']
            #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
            
            truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
            truth_im[truth_im > 0] = 1                                   
            
            
            truth_im_large = np.zeros(np.shape(larger_input_im))
            truth_im_large[0:depth_im, 0:height, 0:width] = truth_im 
            
            plot_max(truth_im_large)
            
            TP, FN, FP = find_TP_FP_FN_from_seg(segmentation, truth_im_large)
            
            if TP + FN == 0: TP;
            else: sensitivity = TP/(TP + FN);     # PPV
                   
            if TP + FP == 0: TP;
            else: precision = TP/(TP + FP);     # precision
   
            print(str(sensitivity))
            print(str(precision))

