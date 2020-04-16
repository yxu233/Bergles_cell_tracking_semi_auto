# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:59:37 2020

@author: tiger
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
""" Turn on eager execution """
tf.compat.v1.enable_eager_execution()
import cProfile
tf.executing_eagerly()

""" Needed to initalize cuDNN"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

import IPython.display as display





import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)



content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')



#content_path = 'C:/Users/tiger/Documents/GitHub/Bergles-lab-CARE/style_transfer_training/content/Substack (10)_content.tif'
#style_path = 'C:/Users/tiger/Documents/GitHub/Bergles-lab-CARE/style_transfer_training/style/Substack (5).tif'

""" Reverse """
#style_path = 'C:/Users/tiger/Documents/GitHub/Bergles-lab-CARE/style_transfer_training/content/Substack (10)_content.tif'
#content_path = 'C:/Users/tiger/Documents/GitHub/Bergles-lab-CARE/style_transfer_training/style/Substack (5).tif'


""" Visualize the input """
# def load_img(path_to_img):
#   max_dim = 1024
#   img = tf.io.read_file(path_to_img)
#   img = tf.image.decode_image(img, channels=3)
#   img = tf.image.convert_image_dtype(img, tf.float32)

#   shape = tf.cast(tf.shape(img)[:-1], tf.float32)
#   long_dim = max(shape)
#   scale = max_dim / long_dim

#   new_shape = tf.cast(shape * scale, tf.int32)

#   img = tf.image.resize(img, new_shape)
#   img = img[tf.newaxis, :]
#   return img

from PIL import Image
def load_tiff(path_to_tiff):
  img = Image.open(path_to_tiff)
  img = np.asarray(img, dtype=np.uint8)
  max_dim = 512
  #img = tf.io.read_file(path_to_img)
  #img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
     
     

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
    
content_image = load_tiff(content_path)
style_image = load_tiff(style_path)

plt.subplot(1, 3, 1)
imshow(content_image, 'Content Image')
plt.pause(0.001)

plt.subplot(1, 3, 2)
imshow(style_image, 'Style Image')
plt.pause(0.001)


plt.figure(3);
imshow(content_image); plt.title('content_image');

plt.figure(4);
imshow(style_image); plt.title('style_image')



import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
im = tensor_to_image(stylized_image)






""" ***********NEED TO PIP INSTALL h5py"""
""" Build own model """

x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape


predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
  print(layer.name)
  
  
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
  
  
  
  
""" Build own for realsies """  
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()
  
  
  
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
  
  
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())

  
  
  
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


image = tf.Variable(content_image)
  
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)  


""" optimizers is only for tensorflow 2.0 """
#opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
opt = tf.train.AdamOptimizer(learning_rate=0.02, epsilon=1e-1)



style_weight=1e-2
content_weight=1e4


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss



@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))



train_step(image)
train_step(image)
train_step(image)
im_train = tensor_to_image(image)



import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  #display.clear_output(wait=True)
  #display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  
  plt.figure();
  output_im = tensor_to_image(image)
  plt.imshow(output_im); plt.pause(0.001)
  
  
  
  
end = time.time()
print("Total time: {:.1f}".format(end-start))

output_im = tensor_to_image(image)
plt.subplot(1, 3, 3)
plt.imshow(output_im); plt.pause(0.001)





