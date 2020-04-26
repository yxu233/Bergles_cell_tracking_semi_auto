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








0.9351851851851852
0.9181818181818182
0.8153846153846154
0.9814814814814815
0.8775933609958506
0.831041257367387



0.9351851851851852
0.9439252336448598
0.8153846153846154
0.9464285714285714
0.8858921161825726
0.9405286343612335




0.8046272493573264  0.87
0.828042328042328  0.83

0.9665809768637532
0.6596491228070176



MOBPF_190105w_1_cuprBZA_10x_T=0_single_channel_RAW_REGISTERED
0.8795518207282913
0.8373333333333334
MOBPF_190105w_1_cuprBZA_10x_T=4_single_channel_aRECOVERY_RAW_REGISTERED
0.8873239436619719
0.6472602739726028
MOBPF_190105w_1_cuprBZA_10x_T=8_single_channel_RAW_REGISTERED
0.8679245283018868
0.8070175438596491
MOBPF_190106w_5_cuprBZA_10x.tif - T=0_single_RAW_REGISTERED
0.922879177377892
0.8692493946731235
MOBPF_190106w_5_cuprBZA_10x.tif - T=2_single_RAW_REGISTERED
0.9189944134078212
0.8308080808080808
MOBPF_190106w_5_cuprBZA_10x.tif - T=6_single_RAW_REGISTERED
0.9315789473684211
0.8119266055045872


Run with ILASTIK
MOBPF_190105w_1_cuprBZA_10x_T=0_single
0.8046272493573264
0.828042328042328
0.9665809768637532
0.6596491228070176
MOBPF_190105w_1_cuprBZA_10x_T=4_single
0.7907949790794979
0.6494845360824743
0.9121338912133892
0.4678111587982833
MOBPF_190105w_1_cuprBZA_10x_T=8_single
0.8476190476190476
0.7876106194690266
0.8238095238095238
0.6577946768060836
MOBPF_190106w_5_cuprBZA_10x.tif - T=0_single
0.9149484536082474
0.8637469586374696
0.9845360824742269
0.583206106870229
MOBPF_190106w_5_cuprBZA_10x.tif - T=2_single
0.9103641456582633
0.8207070707070707
0.9719887955182073
0.5447409733124019
MOBPF_190106w_5_cuprBZA_10x.tif - T=6_single
0.9263157894736842
0.7892376681614349
0.9368421052631579
0.42280285035629456





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

from tifffile import *
    
truth = 0

# Initialize everything with specific random seeds for repeatability
tf.reset_default_graph() 
tf.set_random_seed(1); np.random.seed(1)


def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im[:, :])
     

def find_TP_FP_FN_from_seg(segmentation, truth_im, size_limit=0):
     seg = segmentation      
     true = truth_im
     
     #true = truth_im[:, :, :, 1]
     #seg = seg_train[-1, :, :, :]
     #plot_max(seg)
     #plot_max(true)
     
     
     """ Also remove tiny objects from Truth due to error in cropping """
     #bw_coloc = coloc > 0
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled)
     
     cleaned_truth = np.zeros(np.shape(true))
     for obj in cc_coloc:
          #max_val = obj['max_intensity']
          coords = obj['coords']
          
          # can also skip by size limit          
          if len(coords) > 10:
               for obj_idx in range(len(coords)):
                    cleaned_truth[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
    
     
     
     
     """ Find matched """
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
          
          # can also skip by size limit          
          if max_val > 1 and len(coords) > size_limit:
               TP_count += 1
               #for obj_idx in range(len(coords)):
               #     true_positive[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(seg)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     cleaned_seg = np.zeros(np.shape(seg))
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
     
          # can also skip by size limit
          if  len(coords) < size_limit:
               continue;
          else:
               for obj_idx in range(len(coords)):
                    cleaned_seg[coords[obj_idx, 0], coords[obj_idx, 1], coords[obj_idx, 2]] = 1
          
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count, cleaned_truth, cleaned_seg

             
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
            

""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

"""  Network Begins: """
s_path = './Checkpoints_new_training/'

 
resize_bool = 0

input_size = 256
depth = 64   # ***OR can be 160
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
    


# """ load mean and std """  
input_path = './normalize/'
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')

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
    sav_dir = input_path + '/' + foldername + '_analytic_results_2'
 
    """ Load filenames from .tif """
    # images = glob.glob(os.path.join(input_path,'*_RAW_REGISTERED.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED.tif','_TRUTH_REGISTERED.tif'), ilastik=i.replace('_RAW_REGISTERED.tif','_Object_Predictions.tiff')) for i in images]



    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_single_channel.tif','_truth.tif'), ilastik=i.replace('_single_channel.tif','_single_Object Predictions_.tiff')) for i in images]
    
    
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
            
            
            """ Take out last few slices so can properly assess sens + precision """
            #input_im = input_im[1:110, :, :]
            
            #""" NORMALIZE PROPERLY HERE: """
            #from csbdeep.utils import utils
            #input_im = utils.normalize(input_im, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32)
                                       
            
               
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

   
            """ Analyze each block with offset in all directions """
            
            quad_size = input_size
            quad_depth = depth
            im_size = np.shape(input_im);
            width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
              
            num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);
             

            quad_idx = 1;
            
            overlap_percent = 0.25
            
            #larger_input_im = np.zeros([depth_im + quad_depth + round(quad_depth/2), height + quad_size + round(quad_size/2), width + quad_size + round(quad_size/2)])
            #larger_input_im[0:depth_im, 0:height, 0:width] = input_im
            plot_max(input_im)
            

            
            segmentation = np.zeros([depth_im, width, height])
            input_im_check = np.zeros(np.shape(input_im))
            total_blocks = 0;
            all_xyz = []
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
                          
                          #if z + quad_depth > depth_im or x + quad_size > width or y + quad_size > height:
                          #     continue
                          
                          if z + quad_depth > depth_im:
                               #print('z hit boundary')
                               difference = (z + quad_depth) - depth_im
                               z = z - difference
                          
                              
                          """ Check if repeated """
                          skip = 0
                          for coord in all_xyz:
                               if coord == [x,y,z]:
                                    skip = 1
                                    break                      
                               
                          if skip:
                               continue
                               
                          all_xyz.append([x, y, z])                            
                          
                          quad_intensity = input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size];
                          
                          
                          input_im_check[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = quad_intensity

                          """ NORMALIZE PROPERLY HERE: """
                          #from csbdeep.utils import utils
                          #quad_intensity = utils.normalize(quad_intensity, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32)
                           
                          quad_intensity = normalize_im(quad_intensity, mean_arr, std_arr) 
                                       
                                                          
                          """ Analyze """
                          """ set inputs and truth """
                          quad_intensity = np.expand_dims(quad_intensity, axis=-1)
                          batch_x.append(quad_intensity)
                          batch_y.append(np.zeros([depth, input_size, input_size, num_truth_class]))
                    
                          """ Feed into training loop """
                          feed_dict = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:batch_y} 
                          
                          output_tile = softMaxed.eval(feed_dict=feed_dict)
                          seg_train = np.argmax(output_tile, axis=-1)





                          """ Clean segmentation by removing objects on the edge """
                          cleaned_seg = clean_edges(seg_train[0], quad_depth, w=quad_size, h=quad_size, extra_z=1, extra_xy=3)
                                                    
                          
                          
                          """ ADD IN THE NEW SEG??? or just let it overlap??? """                         
                          #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg
                                                  
                          segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
 
                          total_blocks += 1
                          
                          
                          
                          """ For debug """
                          
                          #plot_max(seg_train[0], ax=0)
                          #plt.pause(3)
                          
                          
            plot_max(segmentation)
            filename = input_name.split('\\')[-1]
            filename = filename.split('.')[0:-1]
            filename = '.'.join(filename)
            
            
            
            
            segmentation = np.asarray(segmentation, np.uint8)
            #segmentation[segmentation > 0] = 255
            imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            segmentation[segmentation > 0] = 1
            
            input_im = np.asarray(input_im, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im)
            
            
            """ Load in truth data for comparison!!! sens + precision """
            truth_name = examples[i]['truth']
            #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
            
            truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
            truth_im[truth_im > 0] = 1                                   
            
            
            #truth_im_large = np.zeros(np.shape(larger_input_im))
            #truth_im_large[0:depth_im, 0:height, 0:width] = truth_im 

            truth_im_cleaned = clean_edges(truth_im, depth_im, w=width, h=height, extra_z=1, extra_xy=3)
                                              
            TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(segmentation, truth_im_cleaned, size_limit=5)

                  
            plot_max(truth_im_cleaned)
            plot_max(cleaned_seg)
            

            if TP + FN == 0: TP;
            else: sensitivity = TP/(TP + FN);     # PPV
                   
            if TP + FP == 0: TP;
            else: precision = TP/(TP + FP);     # precision
   
            print(filename)
            print(str(sensitivity))
            print(str(precision))
            
            truth_im_cleaned = np.asarray(truth_im_cleaned, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_truth_im_cleaned.tif', truth_im_cleaned)            
            
            
            #zzz
            """ Compare with ilastik () if you want to """

            ilastik_compare = 0
            if ilastik_compare:
                 
                 """ Load in truth data for comparison!!! sens + precision """
                 ilastik_name = examples[i]['ilastik']
                 #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
                 
                 ilastik_im = open_image_sequence_to_3D(ilastik_name, width_max='default', height_max='default', depth='default')
                 ilastik_im[ilastik_im > 0] = 1                                   
                 
                 
                 #truth_im_large = np.zeros(np.shape(larger_input_im))
                 #truth_im_large[0:depth_im, 0:height, 0:width] = truth_im 
     
                 ilastik_im_cleaned = clean_edges(ilastik_im, depth_im, w=width, h=height, extra_z=1, extra_xy=3)
                                                   
                 TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(ilastik_im, truth_im_cleaned, size_limit=10)
                                    
                               
                 ilastik_im_cleaned = np.asarray(ilastik_im_cleaned, np.uint8)
                 imsave(sav_dir + filename + '_' + str(int(i)) +'_ilastik_cleaned.tif', ilastik_im_cleaned)
                 
                      
                 plot_max(ilastik_im_cleaned)
                 
                 if TP + FN == 0: TP;
                 else: sensitivity = TP/(TP + FN);     # PPV
                        
                 if TP + FP == 0: TP;
                 else: precision = TP/(TP + FP);     # precision
        
                 print(str(sensitivity))
                 print(str(precision))                                         