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
matplotlib.use('Agg')
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

def get_train_and_val_from_bcolz(X, Y, input_path, test_size = 0.1, start_idx=0, end_idx=-1, convert_float=1):

    #X = bcolz.open(input_path + 'input_im', mode='r')
    #Y = bcolz.open(input_path + 'truth_im', mode='r')
    

    start = time.perf_counter()
    
    input_batch = X[start_idx:end_idx]
    truth_batch = Y[start_idx:end_idx]
    
    stop = time.perf_counter()
    acc_speed = stop - start
    print(acc_speed)
    
 
    start = time.perf_counter()
        
    X_train, X_valid, y_train, y_valid = train_test_split(input_batch, truth_batch, test_size=test_size, random_state=2018)
        
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
            
    
    start = time.perf_counter()    
    
    if convert_float:
        X_train = np.asarray(X_train, np.float32)
        X_valid = np.asarray(X_valid, np.float32)
        y_train = np.asarray(y_train, np.float32)
        y_valid = np.asarray(y_valid, np.float32)
    
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
    return X_train, X_valid, y_train, y_valid, acc_speed





def get_train_and_val_from_bcolz_by_idx(X, Y, input_path, idx_train, idx_valid=0, start_idx=0, end_idx=-1, convert_float=1):


    #X = bcolz.open(input_path + 'input_im', mode='r')
    #Y = bcolz.open(input_path + 'truth_im', mode='r')
    

    start = time.perf_counter()
    
    input_batch = []
    truth_batch = []
    sub_batch_idx = idx_train[start_idx:end_idx]
    for i in range(len(sub_batch_idx)):
        input_batch.append(X[sub_batch_idx[i]])
        truth_batch.append(Y[sub_batch_idx[i]])


    #input_batch = X[sub_batch_idx]
    #truth_batch = Y[sub_batch_idx]
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
    
 
    start = time.perf_counter()
    
    
    if convert_float:    
        input_batch = np.asarray(input_batch, np.float32)
        truth_batch = np.asarray(truth_batch, np.float32)
    
    stop = time.perf_counter()
    diff = stop - start
    print(diff)
        

    #start = time.perf_counter()
    
    #X_train, X_valid, y_train, y_valid = train_test_split(input_batch, truth_batch, test_size=test_size, random_state=2018)
    
    #stop = time.perf_counter()
    #diff = stop - start
    #print(diff)
    input_batch = np.expand_dims(input_batch, axis=-1)
    #truth_batch = np.expand_dims(truth_batch, axis=-1)
    return input_batch, truth_batch



""" shuffles training data """
def shuffle_data(X_train, X_valid, y_train, y_valid):   
    idx_train = np.arange(X_train.shape[0])
    np.random.shuffle(idx_train);

    idx_valid = np.arange(X_valid.shape[0])
    np.random.shuffle(idx_valid);

    return idx_train, idx_valid
        


""""  Network Begins: """
s_path = './Checkpoints_ILASTIK_matched/'

#s_path = './Checkpoints_ILASTIK_matched/'


#s_path = './2) Checkpoints_ILASTIK_matched/'

#s_path = './1) Checkpoints_no_ILASTIK_matched/'

input_path = '../Train_tracking_data/Train_tracking_data_analytic_results_2/'

input_path = 'C:/Users/Huganir Lab/Documents/GitHub/Bergles-lab/Training_on_C/'

input_path = 'C:/Users/Huganir Lab/Documents/GitHub/Bergles-lab/Training_on_C_ILASTIK_matched/'


#input_path = '/lustre04/scratch/yxu233/Training_on_C_ILASTIK_matched/'


            
""" SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""
input_size = 256
depth = 64   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0


""" original == 60 * 320 * 320, now == 2100 * 150 * 150    so about 7.5 x larger image """
x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 1], name='3D_x') 
y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name='3D_CorrectLabel')
#weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name = 'weighted_labels')


# x_3D = tf.placeholder('float32', shape=[None, None, None, None, None], name='3D_x') 

# depth = tf.shape(inputs_)[-1]
# with tf.control_dependencies([
#         tf.Assert(
#             tf.logical_or(tf.equal(depth, 3), tf.equal(depth, 1)), [depth])
# ]):
#     inputs = tf.cond(
#         tf.equal(tf.shape(inputs_)[-1], 3), lambda: inputs_,
#         lambda: tf.image.grayscale_to_rgb(inputs_))

# y_3D_ = tf.placeholder('float32', shape=[None, None, None, None, None], name='3D_CorrectLabel')
# weight_matrix_3D = []
# training = tf.placeholder(tf.bool, name='training')

""" Creates network and cost function"""
kernel_size = [5, 5, 5]

#y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
#accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
#                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=False, optimizer='adam', multiclass=0)



""" Switched to 1e-6 at 150,000"""
# y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
# accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
#                                                                                        train_rate=1e-6, epsilon = 1e-8, weight_mat=False, optimizer='adam', multiclass=0)




""" Switched to 1e-5 + dropout at 170,000"""
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class, dropout=1, drop_rate = 0.8)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits, weight_matrix_3D,
                                                                                       train_rate=1e-5, epsilon = 1e-8, weight_mat=False, optimizer='adam', multiclass=0)




 
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
    
    plot_cost = list(np.load(s_path + 'plot_cost.npy'))
    plot_cost_val = list(np.load(s_path + 'plot_cost_val.npy'))
    
    plot_jacc = list(np.load(s_path + 'plot_jacc.npy'))
    plot_jacc_val = list(np.load(s_path + 'plot_jacc_val.npy'))
    
    plot_sens = list(np.load(s_path + 'plot_sens.npy'))
    plot_sens_val = list(np.load(s_path + 'plot_sens_val.npy'))
    
    plot_prec = list(np.load(s_path + 'plot_prec.npy'))
    plot_prec_val = list(np.load(s_path + 'plot_prec_val.npy'))
    

""" Prints out all variables in current graph """
tf.trainable_variables()

# Required to initialize all
batch_size = 2; save_iter = 10000;
plot_every = 100; iterations = num_check;

batch_x = []; batch_y = []; batch_weighted = [];

bool_plot_sens = 0

test_size = 0.1




""" Load training data """
""" *** ADD OPTION TO ONLY READ IN PARTS OF DATA FOR SAKE OF RAM ***"""

    
    
print('loading data')
load_by_batch_per_epoch = 1; batch_LARGE_size = 1000;

convert_float = 0


# """ load mean and std """  
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')

acc_speed_all = []



X = bcolz.open(input_path + 'input_im', mode='r')
Y = bcolz.open(input_path + 'truth_im', mode='r')
if not load_by_batch_per_epoch:
    #zzz
    X_train, X_valid, y_train, y_valid = get_train_and_val_from_bcolz(X, Y, input_path, test_size = test_size, convert_float=convert_float)
    #zzz

    num_X_train = len(X_train)
else:
    counter = list(range(len(X)))
    idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
    num_train_samples = len(idx_train)    
    
    # load validation once into RAM
    #X_valid, y_valid = get_train_and_val_from_bcolz_by_idx(X, Y, input_path, idx_valid)
    

""" Start training loop """
while(True):  
    if not load_by_batch_per_epoch:
      idx_train_shuff, idx_valid_shuff = shuffle_data(X_train, X_valid, y_train, y_valid)
      
      
    else:
      np.random.shuffle(idx_train)
      np.random.shuffle(idx_valid)

        
    for i in range(0, len(X), batch_size):
        #start = time.perf_counter()
        if load_by_batch_per_epoch and i % batch_LARGE_size == 0:
            print('getting new batch')
            X_train = []; X_valid = []; y_train = []; y_valid = [];  # RELEASE from RAM???
            X_train, X_valid, y_train, y_valid, acc_speed = get_train_and_val_from_bcolz(X, Y, input_path, test_size = test_size, start_idx=i, end_idx=i + batch_LARGE_size,convert_float=convert_float)
            #X_train, y_train = get_train_and_val_from_bcolz_by_idx(X, Y, input_path, idx_train, start_idx=i, end_idx=i + batch_LARGE_size)
            num_X_train = len(X_train)
            print('length of training at i')
            print(i)
            print(num_X_train)
            acc_speed_all.append(acc_speed)
            idx_train_shuff, idx_valid_shuff = shuffle_data(X_train, X_valid, y_train, y_valid)
            
            
           
        
        if iterations == 0:
            start = time.perf_counter()
        if iterations == 1000:
            stop = time.perf_counter(); diff = stop - start; print(diff)
            
            
        """ Load data """
        batch_x = X_train[idx_train_shuff[i % (num_X_train-batch_size):i % (num_X_train-batch_size) + batch_size]]
        batch_y = y_train[idx_train_shuff[i % (num_X_train-batch_size):i % (num_X_train-batch_size) + batch_size]]
        if batch_x.shape[0] < 2:
            zzz
            
        
        if not convert_float:
            batch_x = np.asarray(batch_x, dtype=np.float32)
            batch_y = np.asarray(batch_y, dtype=np.float32)
            
            

        """ normalization """
        batch_x = normalize_im(batch_x, mean_arr, std_arr) 

        feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:1}
                         
        """ Training: """        
        train_step.run(feed_dict=feed_dict_TRAIN)

        iterations = iterations + 1           
        print('Trained: %d' %(iterations))
           
           
        if iterations % plot_every == 0:

               
              """ Load validation """
              rand = randint(0, len(idx_valid_shuff) - batch_size)   
              batch_x_val = X_valid[idx_valid_shuff[rand:rand+batch_size]]; batch_y_val = y_valid[idx_valid_shuff[rand:rand+batch_size]]
              
              """ normalization """
              batch_x_val = normalize_im(batch_x_val, mean_arr, std_arr) 

              if batch_x_val.shape[0] < 2:
                    zzz


              if not convert_float:
                 batch_x = np.asarray(batch_x, dtype=np.float32)
                 batch_y = np.asarray(batch_y, dtype=np.float32) 

           
              feed_dict_CROSSVAL = {x_3D:batch_x_val, y_3D_:batch_y_val, training:0}      
              
 
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

                   start = time.perf_counter()

                   """ For training data """
                   TP, FN, FP = find_TP_FP_FN_from_im(feed_dict_TRAIN, batch_y[-1], softMaxed)
                   
                   if TP + FN == 0: TP;
                   else: sensitivity = TP/(TP + FN); plot_sens.append(sensitivity);    # PPV
                   
                   if TP + FP == 0: TP;
                   else: precision = TP/(TP + FP);  plot_prec.append(precision)    # precision
                   
                   """ For validation """
                   TP, FN, FP = find_TP_FP_FN_from_im(feed_dict_CROSSVAL, batch_y_val[-1], softMaxed)
                   if TP + FN == 0: TP;
                   else: sensitivity = TP/(TP + FN); plot_sens_val.append(sensitivity);    # PPV
                   
                   if TP + FP == 0: TP;
                   else: precision = TP/(TP + FP); plot_prec_val.append(precision)    # precision                 
     
                   stop = time.perf_counter()
                   diff = stop - start
                   print(diff)

                   """ Plot sens + precision + jaccard + loss """
                   plot_metric_fun(plot_sens, plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
                   plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                   
                   plot_metric_fun(plot_prec, plot_prec_val, class_name='', metric_name='precision', plot_num=31)
                   plt.figure(31); plt.savefig(s_path + 'Precision.png')


              plot_metric_fun(plot_jacc, plot_jacc_val, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
              
                   
              plot_cost_fun(plot_cost, plot_cost_val)                   
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
              plt.close('all')
                            
              plot_depth = 8
              if iterations > 500:
                  if iterations % 1000 == 0:
                       plot_trainer_3D_HUGANIR(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, batch_x[-1], batch_x_val[-1], batch_y[-1], batch_y_val[-1],
                                       s_path, iterations, plot_depth=plot_depth, multiclass=multiclass)

              elif iterations % plot_every == 0:
                       plot_trainer_3D_HUGANIR(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, batch_x[-1], batch_x_val[-1], batch_y[-1], batch_y_val[-1],
                                       s_path, iterations, plot_depth=plot_depth, multiclass=multiclass)
                       
                                                     
        """ To save (every x iterations) """
        #stop = time.perf_counter(); diff = stop - start; print(diff)
        if iterations % save_iter == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_' +  str(iterations)
              save_path = saver.save(sess, save_name)
               
              """ Saving the objects """
              np.save(s_path + 'plot_cost.npy', plot_cost)
              np.save(s_path + 'plot_cost_val.npy', plot_cost_val)
              
              np.save(s_path + 'plot_jacc.npy', plot_jacc)
              np.save(s_path + 'plot_jacc_val.npy', plot_jacc_val)
              
              np.save(s_path + 'plot_sens.npy', plot_sens)
              np.save(s_path + 'plot_sens_val.npy', plot_sens_val)
              
              np.save(s_path + 'plot_prec.npy', plot_prec)
              np.save(s_path + 'plot_prec_val.npy', plot_prec_val)
              

              
              