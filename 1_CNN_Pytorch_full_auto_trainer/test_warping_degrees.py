#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:48:39 2020

@author: user
"""


                 """ For trying out different warping functions 
                 
                         cell_idx in image timeseries 680 that are good:
                             - 600 ==> nice clean simple cell
                             
                 """

                 # test_permutations = 0
                 # if test_permutations:
                     
                   
                 #    plot_max(crop_im, ax=-1)
                 #    plot_max(crop_cur_seg, ax=-1)
                 #    plot_max(crop_next_input, ax=-1)
                 #    #plot_max(crop_next_seg, ax=-1)
                 #    crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
                 #    crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2              
                 #    plot_max(crop_next_seg_non_bin, ax=-1)               
                 #    plot_max(seg_train, ax=-1)
                  
                 #    p = 1 
                  
                 #    ### (1) try with different flips
                 #    #transforms = [RandomFlip(axes = 0, flip_probability = 1, p = p, seed = None)]; transform = Compose(transforms)
                    
                 #    ### (2) try with different blur
                 #    #transforms = [RandomBlur(std = (0, 4), p = p, seed=None)]; transforms = Compose(transforms)
                    
                    
                 #    ### (3) try with different warp (affine transformatins)
                 #    transforms = [RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
                 #                        default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
                 #                        p = p, seed=None)]; transforms = Compose(transforms)                    

                 #    ### (4) try with different warp (elastic transformations)
                 #    #transforms = [RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
                 #    #                                locked_borders = 2, image_interpolation = Interpolation.LINEAR,
                 #    #                                p = p, seed = None),]; transforms = Compose(transforms)

                 #    ### (5) try with different motion artifacts
                 #    #transforms = [RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = Interpolation.LINEAR,
                 #    #                    p = p, seed = None),]; transforms = Compose(transforms)


                 #    ### (6) try with different noise artifacts
                 #    #transforms = [RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None)]; transforms = Compose(transforms)

                     
                 #    ### transforms to apply to crop_im                     
                 #    inputs = crop_im
                 #    inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
                 #    #labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
                 #    labels = inputs
                
                 #    subject_a = Subject(
                 #            one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
                 #            a_segmentation=Image(None, torchio.LABEL, labels))
                      
                 #    subjects_list = [subject_a]
            
                 #    subjects_dataset = ImagesDataset(subjects_list, transform=transforms)
                 #    subject_sample = subjects_dataset[0]
                      
                      
                 #    """ MUST ALSO TRANSFORM THE SEED IF IS ELASTIC, rotational transformation!!!"""
                      
                 #    X = subject_sample['one_image']['data'].numpy()
                 #    Y = subject_sample['a_segmentation']['data'].numpy()
                     
                 #    if next_bool:
                 #        batch_x = np.zeros((4, ) + np.shape(crop_im))
                 #        batch_x[0,...] = X
                 #        batch_x[1,...] = crop_cur_seg
                 #        batch_x[2,...] = crop_next_input
                 #        batch_x[3,...] = crop_next_seg
                 #        batch_x = np.moveaxis(batch_x, -1, 1)
                 #        batch_x = np.expand_dims(batch_x, axis=0)
                
                 #    else:
                 #        batch_x = np.zeros((3, ) + np.shape(crop_im))
                 #        batch_x[0,...] = X
                 #        batch_x[1,...] = crop_cur_seg
                 #        batch_x[2,...] = crop_next_input
                 #        #batch_x[3,...] = crop_next_seg
                 #        batch_x = np.moveaxis(batch_x, -1, 1)
                 #        batch_x = np.expand_dims(batch_x, axis=0)
                
                    
                 #    ### NORMALIZE
                 #    batch_x = normalize(batch_x, mean_arr, std_arr)                 

                 #    ### Convert to Tensor
                 #    inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
    
                 #    # forward pass to check validation
                 #    output_val = unet(inputs_val)
    
                 #    """ Convert back to cpu """                                      
                 #    output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                 #    seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
    
                 #    plot_max(X[0], ax=-1)
                 #    plot_max(seg_train, ax=-1)
                     
                 