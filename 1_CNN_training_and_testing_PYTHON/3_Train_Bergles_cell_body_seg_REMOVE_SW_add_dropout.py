# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger





***DO YOU HAVE TO SCALE MAE values to be same order of magnitude as MSSIM??? maybe not... b/c amount of movement of MSSSIM is still on same order of magnitude...?

Next tests:
     9) deeper even!!!
     10) double convolution layers???
     11) try different weighting of MAE/MSSIM
     
     
     start watershed segmentation
     learn to use csbdeep's tiling functions
     quantify shrinkage
      
     
     
     
     
     
STUFF TO ADD:
    - save x, y and x_val, y_val
    - add transforms
    - switch away from csbdeep?
    - boiler plate code
     

"""

""" ALLOWS print out of results on compute canada """
#from keras import backend as K
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))

import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt



""" Libraries to load """
import tensorflow as tf
import cv2 as cv2
from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from natsort import natsort_keygen, ns
#from skimage import measure
import pickle as pickle
import os

from random import randint

from plot_functions import *
from data_functions import *
from data_functions_3D import *
#from post_process_functions import *
from UNet import *
from UNet_3D import *
import glob, os
#from tifffile import imsave

from random import randint


# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);


""" Create training data """

from csbdeep import data
import numpy as np
import matplotlib.pyplot as plt
import datetime

from csbdeep import io
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order




# patch_size = (64, 256, 256)     # ALL MUST BE DIVISIBLE BY 4

# n_patches_per_image = 25    # ideally should be 15,000
# #""" Load and generate training data """
# raw_data = data.RawData.from_folder(basepath='D:/From Tiger/Tiger temp/cell_tracking_training/1_first_transfer_training_data/', source_dirs=['Raw'], 
#                                     target_dir='Truth', axes='CZYX', pattern='*.tif*')

# X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
#                                      patch_filter=None, shuffle=True)

# #io.save_training_data('training_data_CONFOCAL_PINHOLE', X, Y, XY_axes)
# io.save_training_data('training_data_Bergles_cell_tracking', X, Y, XY_axes)


# """ Save training data splits """
# np.save("X_train", X)
# np.save("X_val", X_val)
# np.save("Y_train", Y)
# np.save("Y_val", Y_val)



def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im)
     

# patch_size = (16, 64, 64)     # ALL MUST BE DIVISIBLE BY 4
# n_patches_per_image = 8000    # ideally should be 15,000
# #raw_data = data.RawData.from_folder(basepath='./Training data - only 63x/', source_dirs=['Train-Medium', 'Train-Bad'], 
# #                                    target_dir='Truth', axes='CZYX', pattern='*.tif*')

# raw_data = data.RawData.from_folder(basepath='/lustre04/scratch/yxu233/Training data - only 63x worse Z Huganir/', source_dirs=['Train-medium', 'Train-bad'],
#                                     target_dir='Truth', axes='CZYX', pattern='*.tif*')

# X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
#                                       shuffle=True)
#                                         # CAN TURN BACK ON PATCH_FILTER ==> to reduce background patches
# io.save_training_data('training_data_HUGANIR', X, Y, XY_axes)


def find_TP_FP_FN_from_im(feed_dict, truth_im, softMaxed):
     #feed_dict = feed_dict_TRAIN
     output_train = softMaxed.eval(feed_dict=feed_dict)
     seg_train = np.argmax(output_train, axis = -1)             
        
     true = truth_im[:, :, :, 1]
     seg = seg_train[-1, :, :, :]
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

""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

""" Start training """
from csbdeep.io import load_training_data
# (X,Y), (X_val,Y_val), axes = load_training_data('training_data_Bergles_cell_tracking_FIXED_with.npz', validation_split=0.1)


name = ''
X = np.load(name + "X_train.npy")
X_val = np.load(name + "X_val.npy")
Y = np.load(name +  "Y_train.npy")
Y_val = np.load(name + "Y_val.npy")

all_idx_low_sens = np.load(name + 'all_idx_low_sens.npy')


"""  Network Begins: """
s_path = './Checkpoints/'

s_path = './Checkpoints_remove_spatial_weighting_at_67000/'


s_path = './(3) Checkpoints_add_dropout/'



# """ load mean and std """  
# mean_arr = load_pkl('', 'mean_val_VERIFIED.pkl')
# std_arr = load_pkl('', 'std_val_VERIFIED.pkl')
               
""" SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""
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
#y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
#accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
#                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=True, optimizer='adam', multiclass=0)



y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class, dropout=1, drop_rate = 0.8)
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
    
    
    #plot_MSSIM = []; plot_MSSIM_val = [];
    #plot_MAE = []; plot_MAE_val = [];
    #plot_MSSIM_single = []; plot_loss_single = [];
    num_check= 0;
    
    #if multiclass:    
    #     for i in range(num_truth_class - 1):
    #          plot_MSSIM.append([])
    #         plot_MSSIM_val.append([])
    
else:   
    """ Find last checkpoint """       
    last_file = onlyfiles_check[-1]
    split = last_file.split('check_')[-1]
    num_check = split.split('.')
    checkpoint = num_check[0]
    checkpoint = 'check_' + checkpoint
    num_check = int(num_check[0])
    
    #checkpoint = 'check_36400' 
    saver.restore(sess, s_path + checkpoint)
    
    # Getting back the objects:
    with open(s_path + 'loss_global.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_cost = loaded[0]
        plot_cost = plot_cost[0:8000]
        #plot_cost = plot_cost      
    
    # Getting back the objects:
    with open(s_path + 'loss_global_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_cost_val = loaded[0]  
        plot_cost_val = plot_cost_val[0:8000]
        #plot_cost_val = plot_cost_val


    # Getting back the objects:
    with open(s_path + 'jacc_t.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_jacc = loaded[0]
        plot_jacc = plot_jacc[0:8000]
        #plot_jacc = plot_jacc      
    
    # Getting back the objects:
    with open(s_path + 'jacc_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_jacc_val = loaded[0]  
        plot_jacc_val = plot_jacc_val[0:8000]
        #plot_jacc_val = plot_jacc_val


    # Getting back the objects:
    with open(s_path + 'sens_t.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_sens = loaded[0]
        plot_sens = plot_sens[0:8000]
        #plot_sens = plot_sens
        
        
    # Getting back the objects:
    with open(s_path + 'sens_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_sens_val = loaded[0]  
        plot_sens_val = plot_sens_val[0:8000]
        #plot_sens_val = plot_sens_val


    # Getting back the objects:
    with open(s_path + 'prec_t.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_prec = loaded[0]
        plot_prec = plot_prec[0:8000]
        #plot_prec = plot_prec      
    
    # Getting back the objects:
    with open(s_path + 'prec_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_prec_val = loaded[0]  
        plot_prec_val = plot_prec_val[0:8000]
        #plot_prec_val = plot_prec_val




""" Prints out all variables in current graph """
tf.trainable_variables()


# Required to initialize all
batch_size = 1; 
save_epoch = 1000;
plot_every = 10;
epochs = num_check;

batch_x = []; batch_y = []; batch_weighted = [];

bool_plot_sens = 0

for P in range(8888888888888888888888888):
    #np.random.shuffle(X); np.random.shuffle(Y);
    S = np.arange(X.shape[0])
    np.random.shuffle(S);
    X = X[S]; Y = Y[S];
    
    L = np.arange(X_val.shape[0])
    np.random.shuffle(L);
    X_val = X_val[L]; Y_val = Y_val[L]    
    #np.random.shuffle(X_val); np.random.shuffle(Y_val);
    for i in range(len(X)):
             
        
        
        """ Train with BAD sensitivity samples 30% of the time ==> added at 103000"""
        # idx = i
        # rand = randint(1, 10)
        # if rand <= 3:  # run the bad samples for training 30% of the time
        #     print('yea boi')
        #     np.random.shuffle(all_idx_low_sens)
        #     idx = all_idx_low_sens[0]
            
        
        
        """ Load input image """
        input_im = X[i]

        """ Load truth image """  
        truth_im = Y[i]
        truth_im[truth_im > 0] = 1
        
        background = np.zeros(np.shape(truth_im))
        background[truth_im == 0] = 1
        
        truth_full = np.zeros([depth, input_size, input_size, num_truth_class])
        truth_full[:, :, :, 1] = truth_im[:, :, :, 0]
        truth_full[:, :, :, 0] = background[:, :, :, 0]
        
        truth_im = truth_full
        
        
        
        """ create spatial weight """
        #sp_weighted_labels = spatial_weight(truth_im[:, :, :, 1],edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_im)
        #weighted_labels[:, :, :, 1] = sp_weighted_labels
        
            
        """ maybe remove normalization??? """
        # input_im_save = np.copy(input_im)
        # input_im = normalize_im(input_im, mean_arr, std_arr) 

        """ set inputs and truth """
        batch_x.append(input_im)
        batch_y.append(truth_im)
        batch_weighted.append(np.ones(np.shape(weighted_labels)))
        
        """ Feed into training loop """
        if len(batch_x) == batch_size:
           
           feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:batch_weighted}
                                 
           train_step.run(feed_dict=feed_dict_TRAIN)

           batch_x = []; batch_y = []; batch_weighted = [];
           epochs = epochs + 1           
           print('Trained: %d' %(epochs))
           
           
           if epochs % plot_every == 0:
               
              """ Load validation """
              #plt.close(2)
              #plt.close(18)
              #plt.close(19)
              #plt.close(21)
              batch_x_val = []
              batch_y_val = []
              batch_weighted_val = []
              for batch_i in range(len(X_val)):
            
                  # select random validation image:
                  rand_idx = randint(0, len(X_val)- 1)
                     
                  """ Load input image """
                  input_im_val = X_val[rand_idx]
            
  
                  """ maybe remove normalization??? """
                  #input_im_val = normalize_im(input_im_val, mean_arr, std_arr) 
            
                  """ Load truth image """                  
                  truth_im_val = Y_val[rand_idx]
                  truth_im_val[truth_im_val > 0] = 1
                  
                  background = np.zeros(np.shape(truth_im_val))
                  background[truth_im_val == 0] = 1
                  
                  truth_full = np.zeros([depth, input_size, input_size, num_truth_class])
                  truth_full[:, :, :, 1] = truth_im_val[:, :, :, 0]
                  truth_full[:, :, :, 0] = background[:, :, :, 0]     
                  
                  truth_im_val = truth_full
     
                  """ create spatial weight """
                  #sp_weighted_labels = spatial_weight(truth_im_val[:, :, :, 1],edgeFalloff=10,background=0.01,approximate=True)
          
                  """ Create a matrix of weighted labels """
                  weighted_labels_val = np.copy(truth_im_val)
                  #weighted_labels_val[:, :, :, 1] = sp_weighted_labels
                          
                  """ set inputs and truth """
                  batch_x_val.append(input_im_val)
                  batch_y_val.append(truth_im_val)
                  batch_weighted_val.append(np.ones(np.shape(weighted_labels_val)))
                  
                  if len(batch_x_val) == batch_size:
                      break             
              feed_dict_CROSSVAL = {x_3D:batch_x_val, y_3D_:batch_y_val, training:0, weight_matrix_3D:batch_weighted_val}      
              
 
              """ Training loss"""
              loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
              plot_cost.append(loss_t);                 
                             
              """ loss """
              loss_val = cross_entropy.eval(feed_dict=feed_dict_CROSSVAL)
              plot_cost_val.append(loss_val)


              """ Training loss"""
              jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN);
              plot_jacc.append(jacc_t);                 
                             
              """ loss """
              jacc_val = jaccard.eval(feed_dict=feed_dict_CROSSVAL)
              plot_jacc_val.append(jacc_val)


          
          
              """ Calculate sensitivity + precision as other metrics """
              if jacc_val > 0.1:
                   bool_plot_sens = 1
                   
              if bool_plot_sens:
                   
                   TP, FN, FP = find_TP_FP_FN_from_im(feed_dict_TRAIN, truth_im, softMaxed)
                   
                   if TP + FN == 0: TP;
                   else: sensitivity = TP/(TP + FN); plot_sens.append(sensitivity);    # PPV
                   
                   if TP + FP == 0: TP;
                   else: precision = TP/(TP + FP);  plot_prec.append(precision)    # precision
   
                    

                   
                   TP, FN, FP = find_TP_FP_FN_from_im(feed_dict_CROSSVAL, truth_im_val, softMaxed)
                   if TP + FN == 0: TP;
                   else: sensitivity = TP/(TP + FN); plot_sens_val.append(sensitivity);    # PPV
                   
                   if TP + FP == 0: TP;
                   else: precision = TP/(TP + FP); plot_prec_val.append(precision)    # precision                 
     


                   """ Plot sens + precision """
                   plot_metric_fun(plot_sens, plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
                   plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                   
                   plot_metric_fun(plot_prec, plot_prec_val, class_name='', metric_name='precision', plot_num=31)
                   plt.figure(31); plt.savefig(s_path + 'Precision.png')
              
              if not multiclass:
                   
                   plot_cost_fun(plot_cost, plot_cost_val)
                   plot_jaccard_fun(plot_jacc, plot_jacc_val, class_name=' jaccard')
                   
                   
                   plt.figure(18); plt.savefig(s_path + 'global_loss.png')
                   plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
                   plt.figure(21); plt.savefig(s_path + 'Jaccard.png')
                   plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                   
                   """ Also plot MAE """
                   #plot_jaccard_fun(plot_MAE, plot_MAE_val, class_name=' MAE')
                   #plt.figure(21); plt.savefig(s_path + 'MAE.png')
                                  
                           
              plot_depth = 8
              if epochs > 500:
                  if epochs % 100 == 0:
                       plot_trainer_3D_HUGANIR(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                                       s_path, epochs, plot_depth=plot_depth, multiclass=multiclass)

              elif epochs % plot_every == 0:
                       plot_trainer_3D_HUGANIR(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                                       s_path, epochs, plot_depth=plot_depth, multiclass=multiclass)
                                                     
           """ To save (every x epochs) """
           if epochs % save_epoch == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_' +  str(epochs)
              save_path = saver.save(sess, save_name)
               
              """ Saving the objects """
              save_pkl(plot_cost, s_path, 'loss_global.pkl')
              save_pkl(plot_cost_val, s_path, 'loss_global_val.pkl')
              save_pkl(plot_jacc, s_path, 'jacc_t.pkl')
              save_pkl(plot_jacc_val, s_path, 'jacc_val.pkl')
              
             
              save_pkl(plot_sens, s_path, 'sens_t.pkl')
              save_pkl(plot_sens_val, s_path, 'sens_val.pkl')
              
              save_pkl(plot_prec, s_path, 'prec_t.pkl')
              save_pkl(plot_prec_val, s_path, 'prec_val.pkl')
              
              
              #save_pkl(plot_MAE, s_path, 'MAE.pkl')
              #save_pkl(plot_MAE_val, s_path, 'MAE_val.pkl')
              
              #save_pkl(validation_counter, s_path, 'val_counter.pkl')
              #save_pkl(input_counter, s_path, 'input_counter.pkl')   
              #save_pkl(plot_MSSIM_single, s_path, 'MSSIM_single.pkl')                                
              #save_pkl(plot_loss_single, s_path, 'loss_single.pkl')  
              
              
              """Getting back the objects"""
#              plot_cost = load_pkl(s_path, 'loss_global.pkl')
#              plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
#              plot_MSSIM = load_pkl(s_path, 'MSSIM.pkl')
#              plot_MSSIM_val = load_pkl(s_path, 'MSSIM_val.pkl')
#              plot_MSSIM_single = load_pkl(s_path, 'MSSIM_single.pkl')                                
#              plot_loss_single = load_pkl(s_path, 'loss_single.pkl')
              
              
              