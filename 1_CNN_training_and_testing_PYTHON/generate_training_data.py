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

def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im[:, :])
     

""" removes detections on the very edges of the image """
def clean_edges(im, depth, w, h, extra_z=1, extra_xy=5):
     labelled = measure.label(im)
     cc_coloc = measure.regionprops(labelled)
    
     cleaned_im = np.zeros(np.shape(im))
     for obj in cc_coloc:
         #max_val = obj['max_intensity']
         coords = obj['coords']
         
         bool_edge = 0
         for c in coords:
              if (c[0] <= 0 + extra_z or c[0] >= depth - extra_z):
                   #print('badz')
                   bool_edge = 1
                   break;
              if (c[1] <= 0 + extra_xy or c[1] >= w - extra_xy):
                   #print('badx')
                   bool_edge = 1
                   break;                                       
              if (c[2] <= 0 + extra_xy or c[2] >= h - extra_xy):
                   #print('bady')
                   bool_edge = 1
                   break;                                        
                   
                   
    
         if not bool_edge:
              #print('good')
              for obj_idx in range(len(coords)):
                   cleaned_im[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1

     return cleaned_im                     
            


resize_bool = 0

input_size = 128
depth = 32   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0


input_size = 256
depth = 64   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0

tf_size = input_size


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
    images = glob.glob(os.path.join(input_path,'*_RAW_REGISTERED_substack_1_110.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED_substack_1_110.tif','_TRUTH_REGISTERED_substack_1_110.tif'), ilastik=i.replace('_RAW_REGISTERED_substack_1_110.tif','_Object_Predictions.tiff')) for i in images]

    
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

            truth_name = examples[i]['truth']
            truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
            
            """ NEED TO CONVERT TO np.uint8 if the original input is np.uint16!!!"""
            # NORMALIZED BECAUSE IMAGE IS uint16 ==> do same when actually running images!!!
            #input_im = np.asarray(Image.open(input_name))
            if input_im.dtype == 'uint16':
                input_im = np.asarray(input_im, dtype=np.float32)
                input_im = cv.normalize(input_im, 0, 255, cv.NORM_MINMAX)
                input_im = input_im * 255
                
            input_im = np.asarray(input_im, dtype= np.float32)
            
   
            """ Analyze each block with offset in all directions """
            quad_size = input_size
            quad_depth = depth
            im_size = np.shape(input_im);
            width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
              
            num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);
             

            quad_idx = 1;
            
            overlap_percent = 0.40
           
            plot_max(input_im)
           
            segmentation = np.zeros([depth_im, width, height])
            input_im_check = np.zeros(np.shape(input_im))
            total_blocks = 0;
            for x in range(1, width + quad_size, round(quad_size - quad_size * overlap_percent)):
                 if x + quad_size > width:
                      #print('x hit boundary')
                      difference = (x + quad_size) - width
                      x = x - difference
                           
                 for y in range(1, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                      
                      if y + quad_size > height:
                           #print('y hit boundary')
                           difference = (y + quad_size) - height
                           y = y - difference
                      
                      for z in range(1, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                          batch_x = []; batch_y = [];
                          
                          if z + quad_depth > depth_im:
                               #print('z hit boundary')
                               difference = (z + quad_depth) - depth_im
                               z = z - difference
                          
                          
                          quad_intensity = input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                          quad_truth = truth_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                          
                          """ Clean segmentation by removing objects on the edge """
                          #cleaned_seg = clean_edges(seg_train[0], quad_depth, w=quad_size, h=quad_size, extra_z=1, extra_xy=3)
                          #cleaned_seg = seg_train
                          
                          #plot_max(quad_intensity)
                          #plot_max(quad_truth)
                          

                          """ Save block """                          
                          filename = input_name.split('\\')[-1]
                          filename = filename.split('.')[0:-1]
                          filename = '.'.join(filename)
                          
                          
                          filename = filename.split('RAW_REGISTERED')[0]
               
                           
                          #segmentation = np.asarray(segmentation, np.uint8)
                          #segmentation[segmentation > 0] = 255
                          imsave(sav_dir + filename + '_' + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_INPUT.tif', np.uint8(quad_intensity))
                          #segmentation[segmentation > 0] = 1
                           
                          #input_im = np.asarray(input_im, np.uint8)
                          quad_truth[quad_truth > 0] = 255
                          imsave(sav_dir + filename + '_' + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_TRUTH.tif', np.uint8(quad_truth))
                          
                         
                           
            


