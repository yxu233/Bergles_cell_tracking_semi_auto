# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger

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
import pickle as pickle
import os

from random import randint

from plot_functions import *
from data_functions import *
from data_functions_3D import *
from UNet import *
from UNet_3D import *
import glob, os

from random import randint


# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);


""" Create training data """
import numpy as np
import matplotlib.pyplot as plt
import datetime

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


from sklearn.model_selection import train_test_split
import time
import bcolz

def get_train_and_val_from_bcolz(input_path, test_size = 0.1, start_idx=0, end_idx=-1):

    X = bcolz.open(input_path + 'input_im', mode='r')
    Y = bcolz.open(input_path + 'truth_im', mode='r')
    

    start = time.perf_counter()
    
    input_batch = X[start_idx:end_idx]
    truth_batch = Y[start_idx:end_idx]
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
    
 
    start = time.perf_counter()
    
    
    input_batch.astype(np.float32)
    truth_batch.astype(np.float32)
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
        

    start = time.perf_counter()
    
    X_train, X_valid, y_train, y_valid = train_test_split(input_batch, truth_batch, test_size=test_size, random_state=2018)
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
    return X_train, X_valid, y_train, y_valid


""" shuffles training data """
def shuffle_data(X_train, X_valid, y_train, y_valid):   
    idx_train = np.arange(X_train.shape[0])
    np.random.shuffle(idx_train);

    idx_valid = np.arange(X_valid.shape[0])
    np.random.shuffle(idx_valid);

    return idx_train, idx_valid
        


""""  Network Begins: """
s_path = './Checkpoints/'
input_path = '../Train_tracking_data/Train_tracking_data_analytic_results_2/'

            
""" SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""
input_size = 256
depth = 64   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0


""" original == 60 * 320 * 320, now == 2100 * 150 * 150    so about 7.5 x larger image """
x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name='3D_x') 
y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name='3D_CorrectLabel')
#weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name = 'weighted_labels')
weight_matrix_3D = []
training = tf.placeholder(tf.bool, name='training')

""" Creates network and cost function"""
kernel_size = [5, 5, 5]
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=False, optimizer='adam', multiclass=0)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

if num_truth_class > 2:
     multiclass = 1
     

""" If no old checkpoint then starts fresh FROM PRE-TRAINED NETWORK """
if not onlyfiles_check:  
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 
    plot_cost = []; plot_cost_val = []; 
    plot_jacc = []; plot_jacc_val = [];
    
    
    plot_sens = []; plot_sens_val = [];
    plot_prec = []; plot_prec_val = [];
    
    num_check= 0;

    
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
    
    plot_cost = np.load(s_path + 'plot_cost.npy')
    plot_cost_val = np.load(s_path + 'plot_cost_val.npy')
    
    plot_jacc = np.load(s_path + 'plot_jacc.npy')
    plot_jacc_val = np.load(s_path + 'plot_jacc_val.npy')
    
    plot_sens = np.load(s_path + 'plot_sens.npy')
    plot_sens_val = np.load(s_path + 'plot_sens_val.npy')
    
    plot_prec = np.load(s_path + 'plot_prec.npy')
    plot_prec_val = np.load(s_path + 'plot_prec_val.npy')
    

""" Prints out all variables in current graph """
tf.trainable_variables()

# Required to initialize all
batch_size = 2; 
save_epoch = 1000;
plot_every = 100;
epochs = num_check;

batch_x = []; batch_y = [];
weights = [];



""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*_INPUT.tif'))

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
examples = [dict(input=i,truth=i.replace('_INPUT','_TRUTH')) for i in images]

for P in range(8000000000000000000000):
    counter = 0
    for i in range(len(examples)):

              if counter == 0:
                start = time.perf_counter()
              if counter == 100:
                stop = time.perf_counter()
                diff = stop - start
                print(diff)
                zzz    
                 
              
              
              input_name = examples[i]['input']
              input_im = open_image_sequence_to_3D(input_name, width_max=input_size, height_max=input_size, depth=depth)


              truth_name = examples[i]['truth']
              truth_im = open_image_sequence_to_3D(truth_name, width_max=input_size, height_max=input_size, depth=depth)     
             
             
              # """ Load input image """
              # #input_im = X[i]
              input_im = np.expand_dims(input_im, axis=-1)
     
              # """ Load truth image """  
              # #truth_im = Y[i]
              # truth_im = np.expand_dims(truth_im, axis=-1)
              # truth_im[truth_im > 0] = 1
             
              # background = np.zeros(np.shape(truth_im))
              # background[truth_im == 0] = 1
             
              truth_im = np.zeros([depth, input_size, input_size, num_truth_class])
              # truth_full[:, :, :, 1] = truth_im[:, :, :, 0]
              # truth_full[:, :, :, 0] = background[:, :, :, 0]
             
              # truth_im = truth_full
             

              """ create spatial weight """
              #sp_weighted_labels = spatial_weight(truth_im[:, :, :, 1],edgeFalloff=10,background=0.01,approximate=True)
     
              """ Create a matrix of weighted labels """
              #weighted_labels = np.copy(truth_im)
              #weighted_labels[:, :, :, 1] = sp_weighted_labels
             
                 
              """ maybe remove normalization??? """
              # input_im_save = np.copy(input_im)
              # input_im = normalize_im(input_im, mean_arr, std_arr) 
     
              """ set inputs and truth """
              batch_x.append(input_im)
              batch_y.append(truth_im)
              #batch_weighted.append(np.ones(np.shape(weighted_labels)))
              
              
              """ Feed into training loop """
              if len(batch_x) == batch_size:                  
                 feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:0}
                                      
                 train_step.run(feed_dict=feed_dict_TRAIN)
     
                 batch_x = []; batch_y = []; batch_weighted = [];         
                 print('Trained: %d' %(counter))
                
      
                 """ Training loss"""
                 loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
                 plot_cost.append(loss_t);                 
     
                 """ Training loss"""
                 jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN);
                 plot_jacc.append(jacc_t);       


                 batch_x = []; batch_y = []; batch_weighted = [];
               
                 counter += 1

              
              