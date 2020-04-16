# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

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
#import tensorflow as tf
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

#from UnetTrainingFxn_0v6_actualFunctions import *
from random import randint


images = glob.glob('./Checkpoints_HUGANIR_perceptual_loss_training/**/*MAE.pkl', recursive=True)
#images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
""" FOR CLAHE """
examples = [dict(MAE=i,MAE_val=i.replace('MAE','MAE_val'), 
                 loss=i.replace('MAE','loss_global'), loss_val=i.replace('MAE', 'loss_global_val'),
                 MSSIM = i.replace('MAE', 'MSSIM'), MSSIM_val = i.replace('MAE', 'MSSIM_val')
                 ) for i in images]

     
counter = list(range(len(examples)))  # create a counter, so can randomize it
input_counter = counter



def load_pickle(file):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
      loaded = pickle.load(f)
      obj_loaded = loaded[0]
      return obj_loaded
 
     
avg_window_size = 20


list_to_skip = []
#list_to_skip = [8]  # to plot all trials
#list_to_skip = [0,1,5,7,8,11,12,13,14,15]  # to plot different filters
#list_to_skip = [0,1,2,3,4,6,7,8,11]  # to plot different filters

#list_to_skip = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]  # to skip all large image trainings

for idx in input_counter:   
    """ Skip """
    if idx in list_to_skip:
       continue;         
    
    
    """ plot MAE """     
    MAE = examples[idx]['MAE']
    name = MAE.split('\\')[1]
    MAE = load_pickle(MAE)
    
    avg_jacc = moving_average(MAE, n=avg_window_size).tolist()
    plt.figure(1); plt.plot(avg_jacc, alpha=0.6, label='MAE ' + name)
         
    MAE_val = examples[idx]['MAE_val']
    MAE_val = load_pickle(MAE_val)
    
    avg_jacc = moving_average(MAE_val, n=avg_window_size).tolist()
    plt.figure(3); plt.plot(avg_jacc, alpha=0.6, label='MAE_val ' + name)


    """ plot Loss ***NOTE: not necessarily comparable!!! """
    loss = examples[idx]['loss']
    loss = load_pickle(loss)
    loss = np.asarray(loss); 
    if (loss < 0).any(): loss = (loss * -1).tolist()
    avg_jacc = moving_average(loss, n=avg_window_size).tolist()
    plt.figure(5); plt.plot(avg_jacc, alpha=0.6, label='loss ' + name)
    
    loss_val = examples[idx]['loss_val']
    loss_val = load_pickle(loss_val)
    loss_val = np.asarray(loss_val); 
    if (loss_val < 0).any(): loss_val = (loss_val * -1).tolist()
    avg_jacc = moving_average(loss_val, n=avg_window_size).tolist()
    plt.figure(6); plt.plot(avg_jacc, alpha=0.6, label='loss_val ' + name)
    
    
    """ plot MSSSSIM """
    MSSIM = examples[idx]['MSSIM']
    MSSIM = load_pickle(MSSIM)
    MSSIM = np.asarray(MSSIM); 
    if (MSSIM < 0).any(): MSSIM = (MSSIM * -1).tolist()
    avg_jacc = moving_average(MSSIM, n=avg_window_size).tolist()
    plt.figure(7); plt.plot(avg_jacc, alpha=0.6, label='MSSIM ' + name)
    
    MSSIM_val = examples[idx]['MSSIM_val']
    MSSIM_val = load_pickle(MSSIM_val)
    MSSIM_val = np.asarray(MSSIM_val); 
    if (MSSIM_val < 0).any(): MSSIM_val = (MSSIM_val * -1).tolist()
    avg_jacc = moving_average(MSSIM_val, n=avg_window_size).tolist()
    plt.figure(8); plt.plot(avg_jacc, alpha=0.6, label='MSSIM_val ' + name)
    
    
    


plt.figure(1); plt.ylabel('MAE'); plt.xlabel('Iterations');            
plt.legend(loc='lower right');    #plt.pause(0.05)

plt.figure(2); plt.ylabel('MAE'); plt.xlabel('Iterations');            
plt.legend(loc='lower right');    #plt.pause(0.05)

plt.figure(5); plt.ylabel('Loss'); plt.xlabel('Iterations');            
plt.legend(loc='upper right');    #plt.pause(0.05)
#plt.yscale('log')

plt.figure(6); plt.ylabel('Loss'); plt.xlabel('Iterations');            
plt.legend(loc='upper right');    #plt.pause(0.05)
#plt.yscale('log')
    
plt.figure(7); plt.ylabel('MSSIM'); plt.xlabel('Iterations');            
plt.legend(loc='upper right');    #plt.pause(0.05)
#plt.yscale('log')

plt.figure(8); plt.ylabel('MSSIM'); plt.xlabel('Iterations');            
plt.legend(loc='upper right');    #plt.pause(0.05)
#plt.yscale('log')
    