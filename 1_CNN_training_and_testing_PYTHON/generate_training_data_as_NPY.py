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

input_size = 256
depth = 64   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0


# input_size = 256
# depth = 64   # ***OR can be 160
# num_truth_class = 1 + 1 # for reconstruction
# multiclass = 0

# tf_size = input_size


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
    
    input_batch = []; truth_batch = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    
    
    # Create a new HDF5 file
    import h5py
    file = h5py.File(sav_dir + "test_real_yea.h5", "w")
    empty = 1

    total_samples = 0
    for i in range(len(examples)):
            
            input_name = examples[i]['input']
            input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')

            truth_name = examples[i]['truth']
            truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
                      
   
            """ Analyze each block with offset in all directions """
            quad_size = input_size
            quad_depth = depth
            im_size = np.shape(input_im);
            width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
              
            num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);
             

            quad_idx = 1;
            
            overlap_percent = 0.40
           
            #plot_max(input_im)
           
            segmentation = np.zeros([depth_im, width, height])
            input_im_check = np.zeros(np.shape(input_im))
            total_blocks = 0;
            
            all_xyz = []
            for x in range(1, width + quad_size, round(quad_size - quad_size * overlap_percent)):
                 if x + quad_size > width:
                      #print('x hit boundary')
                      difference = (x + quad_size) - width
                      x = x - difference
                           
                      
                 if total_samples >= 500:
                      file.close()
                      break   
                      
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


                          """ Check if repeated """
                          skip = 0
                          for coord in all_xyz:
                               #print(coord)
                               
                               if coord == [x,y,z]:
                                    skip = 1
                                    #print(coord)
                                    #print([x, y, z])
                                    #print('skip')
                                    break                      
                               
                          if skip:
                               continue
                               
                          all_xyz.append([x, y, z])  
                           
                          #segmentation = np.asarray(segmentation, np.uint8)
                          #segmentation[segmentation > 0] = 255
                          imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_INPUT.tif', np.uint8(quad_intensity))
                          # #segmentation[segmentation > 0] = 1
                           
                          # #input_im = np.asarray(input_im, np.uint8)
                          quad_truth[quad_truth > 0] = 255
                          imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_TRUTH.tif', np.uint8(quad_truth))
                          
                          
                          """ Batch for later """
                          #input_batch.append(quad_intensity)
                          
                          #truth_batch.append(quad_truth)
 


                          # initialize dataset in the file
                          quad_intensity = np.expand_dims(quad_intensity, axis=0)
                          quad_truth = np.expand_dims(quad_truth, axis=0)
                          
                          
                          quad_intensity = np.asarray(quad_intensity, np.uint8)
                          quad_truth = np.asarray(quad_truth, np.uint8)
                          
                                                   
                          
                          """ TO SAVE MORE TIME, DO MORE-PREPROCESSING HERE... find mean and std as well??? """
                          
                          
                          if empty:                         
                               dataset = file.create_dataset(
                                  "image", data=quad_intensity, maxshape=(600 ,depth ,input_size, input_size), chunks=(1, depth, input_size, input_size)
                               )
                               truth_set = file.create_dataset(
                                  "truth", data=quad_truth, maxshape=(600, depth,input_size, input_size), chunks=(1, depth, input_size, input_size), 
                               )
                               empty = 0
                          else:                        
                               ## append
                               file['image'].resize((file["image"].shape[0] + quad_intensity.shape[0]), axis = 0)
                               file["image"][-quad_intensity.shape[0]:] = quad_intensity
                        

                               file['truth'].resize((file["truth"].shape[0] + quad_truth.shape[0]), axis = 0)
                               file["truth"][-quad_truth.shape[0]:] = quad_truth                         

                          total_samples += 1
                          print(total_samples)
     
            if total_samples >= 500:
                    file.close()
                    break   



file.close()


""" open hdf5 file """
# filename = './' + "test_real_yea.h5"   # from SSD drive


sav_dir = 'E:/7) Bergles lab data/RemyelinationData/Training_data_substack/Training_data_substack_analytic_results/'
filename = sav_dir + "test_real_yea.h5" # from HDD
f = h5py.File(filename, 'r')

# keys = list(f.keys())

# import time

# start = time.perf_counter()
# input_key = keys[0]
# truth_key = keys[1]

# input_batch = np.asarray(f[input_key], dtype=np.float32)
# truth_batch = np.asarray(f[truth_key], dtype=np.float32)


# stop = time.perf_counter()

# diff = stop - start
# print(diff)


""" Load one at a time """
num_iter = 250
keys = list(f.keys())


input_key = keys[0]
truth_key = keys[1]


import time
start = time.perf_counter()
input_batch = []
truth_batch = []
for i in range(0, num_iter):

      input_batch.append(f[input_key][i])
      truth_batch.append(f[truth_key][i])
     
input_batch = np.float32(input_batch);     
truth_batch = np.float32(truth_batch);

stop = time.perf_counter()

diff = stop - start
print(diff)


f.close()








""" Index once to improve speed??? """
# keys = list(f.keys())


# input_key = keys[0]
# truth_key = keys[1]


# import time
# start = time.perf_counter()


# num_epoch = 1000
# for i in range(0,10000 , num_epoch):
#      input_batch = []
#      truth_batch = []
#      #for idx in range(i, i + num_epoch):

#      input_batch.append(f[input_key][i:i+num_epoch])
#      truth_batch.append(f[truth_key][i:i+num_epoch])
     
#      print(i)
     
# #input_batch = np.float32(input_batch);     
# #truth_batch = np.float32(truth_batch);


# stop = time.perf_counter()

# diff = stop - start
# print(diff)






""" Speed testing for disk loaded training vs. RAM training """
s_path = sav_dir


input_size = 256
depth = 64   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0


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


""" Trained at 1e-5 until 67000 epochs"""
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=True, optimizer='adam', multiclass=0)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)


""" If no old checkpoint then starts fresh FROM PRE-TRAINED NETWORK """
if not onlyfiles_check:  
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 
    plot_cost = []; plot_cost_val = []; 
    plot_jacc = []; plot_jacc_val = [];
    
    
    plot_sens = []; plot_sens_val = [];
    plot_prec = []; plot_prec_val = [];
    
    num_check= 0;
    
    



""" Prints out all variables in current graph """
tf.trainable_variables()


# Required to initialize all
batch_size = 1; 
save_epoch = 1000;
plot_every = 10;
epochs = num_check;

batch_x = []; batch_y = []; batch_weighted = [];

bool_plot_sens = 0


# 2000 steps ==> 311.15984899999967 seconds

def from_RAM(X, Y, cross_entropy, jaccard, batch_size, train_step, depth, input_size, num_truth_class, num_iter):
 
      for i in range(num_iter):
                 
             batch_x = []; batch_y = []; batch_weighted = [];
             
             """ Load input image """
             input_im = X[i]
             #input_im = np.expand_dims(input_im, axis=-1)
     
             """ Load truth image """  
             truth_im = Y[i]
             #truth_im = np.expand_dims(truth_im, axis=-1)
      
      

             # truth_im[truth_im > 0] = 1
             
             # background = np.zeros(np.shape(truth_im))
             # background[truth_im == 0] = 1
             
             # truth_full = np.zeros([depth, input_size, input_size, num_truth_class])
             # truth_full[:, :, :, 1] = truth_im[:, :, :, 0]
             # truth_full[:, :, :, 0] = background[:, :, :, 0]
             
             # truth_im = truth_full
             
             
             
             # """ create spatial weight """
             # #sp_weighted_labels = spatial_weight(truth_im[:, :, :, 1],edgeFalloff=10,background=0.01,approximate=True)
     
             # """ Create a matrix of weighted labels """
             # weighted_labels = np.copy(truth_im)
             # #weighted_labels[:, :, :, 1] = sp_weighted_labels
             
                 
             # """ maybe remove normalization??? """
             # # input_im_save = np.copy(input_im)
             # # input_im = normalize_im(input_im, mean_arr, std_arr) 
     
             # """ set inputs and truth """
             # batch_x.append(input_im)
             # batch_y.append(truth_im)
             # batch_weighted.append(np.ones(np.shape(weighted_labels)))
             
             
             # """ Feed into training loop """
             # if len(batch_x) == batch_size:
                
             #    feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:0, weight_matrix_3D:batch_weighted}
                                      
             #    #train_step.run(feed_dict=feed_dict_TRAIN)
     
             #    batch_x = []; batch_y = []; batch_weighted = [];         
             #    print('Trained: %d' %(i))
                
      
             #    """ Training loss"""
             #    loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
             #    plot_cost.append(loss_t);                 
     
             #    """ Training loss"""
             #    jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN);
             #    plot_jacc.append(jacc_t);        

import time

start = time.perf_counter()
from_RAM(input_batch, truth_batch, cross_entropy, jaccard, batch_size, train_step,depth, input_size, num_truth_class, num_iter=num_iter)
stop = time.perf_counter()

diff = stop - start
print(diff)

         


          



# with 2000 steps ==> 432.1105502999999 seconds

def from_DISK(examples, cross_entropy, jaccard, batch_size, train_step, depth, input_size, num_truth_class, num_iter):
     
      for i in range(num_iter):
                 
             batch_x = []; batch_y = []; batch_weighted = [];

             input_name = examples[i]['input']
             input_im = open_image_sequence_to_3D(input_name, width_max=input_size, height_max=input_size, depth=depth)


             truth_name = examples[i]['truth']
             truth_im = open_image_sequence_to_3D(truth_name, width_max=input_size, height_max=input_size, depth=depth)     
             
             
             # """ Load input image """
             # #input_im = X[i]
             # input_im = np.expand_dims(input_im, axis=-1)
     
             # """ Load truth image """  
             # #truth_im = Y[i]
             # truth_im = np.expand_dims(truth_im, axis=-1)
             # truth_im[truth_im > 0] = 1
             
             # background = np.zeros(np.shape(truth_im))
             # background[truth_im == 0] = 1
             
             # truth_full = np.zeros([depth, input_size, input_size, num_truth_class])
             # truth_full[:, :, :, 1] = truth_im[:, :, :, 0]
             # truth_full[:, :, :, 0] = background[:, :, :, 0]
             
             # truth_im = truth_full
             

             # """ create spatial weight """
             # #sp_weighted_labels = spatial_weight(truth_im[:, :, :, 1],edgeFalloff=10,background=0.01,approximate=True)
     
             # """ Create a matrix of weighted labels """
             # weighted_labels = np.copy(truth_im)
             # #weighted_labels[:, :, :, 1] = sp_weighted_labels
             
                 
             # """ maybe remove normalization??? """
             # # input_im_save = np.copy(input_im)
             # # input_im = normalize_im(input_im, mean_arr, std_arr) 
     
             # """ set inputs and truth """
             # batch_x.append(input_im)
             # batch_y.append(truth_im)
             # batch_weighted.append(np.ones(np.shape(weighted_labels)))
             
             
             # """ Feed into training loop """
             # if len(batch_x) == batch_size:
                
             #    feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:0, weight_matrix_3D:batch_weighted}
                                      
             #    #train_step.run(feed_dict=feed_dict_TRAIN)
     
             #    batch_x = []; batch_y = []; batch_weighted = [];         
             #    print('Trained: %d' %(i))
                
      
             #    """ Training loss"""
             #    loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
             #    plot_cost.append(loss_t);                 
     
             #    """ Training loss"""
             #    jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN);
             #    plot_jacc.append(jacc_t);                 
                                                     

input_batch = []
truth_batch = []

input_path = 'E:/7) Bergles lab data/RemyelinationData/Training_data_substack/Training_data_substack_analytic_results/'
images = glob.glob(os.path.join(input_path,'*_INPUT.tif'))

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
examples = [dict(input=i,truth=i.replace('_INPUT','_TRUTH')) for i in images]


import time
start = time.perf_counter()
from_DISK(examples, cross_entropy, jaccard, batch_size, train_step,depth, input_size, num_truth_class, num_iter=num_iter)
stop = time.perf_counter()

diff = stop - start
print(diff)


