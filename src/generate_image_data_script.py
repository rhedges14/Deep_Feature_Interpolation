#!/usr/bin/env python
# coding: utf-8

# In[5]:


from keras.applications import VGG19
from keras import layers, models
from keras.models import Model
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd


def import_image(image,
                 image_directory,
                 resize_dims):
    
    """
    load image, change from bgr to rgb, resize dimensions, normalize pixel values
    input: 
        image: a jpg image
        resize_dims: dimensions to resize (downsample) image to
    output: 
        image processed for neural net
    """
    
    image_bgr = cv2.imread(image_directory+image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image_rgb, resize_dims)
    img = np.expand_dims(image_resize, 0)
    return img / 255.0






def load_images_to_h5(input_dims, image_directory, destination_file, n=None):
    """
    Process images from directory into h5py file
    
    Inputs:
        input_dims: dimensions of individual imagges
        image_list_csv: csv file with list of image file names as the index vector
        destination file: name of h5 file to be created
        n: # images to load in 
    """
    
    if os.path.exists(destination_file): os.remove(destination_file)
    
    if n == None:
        n = len(os.listdir(image_directory))
    
    with h5py.File(destination_file, "w") as images_h5py:
        img_file_dims = (n, input_dims[0], input_dims[1], input_dims[2])
        images_file = images_h5py.create_dataset("img_data",
                                               img_file_dims,
                                               dtype=np.float64)

        for i, image in enumerate(os.listdir(image_directory)[:n]):
            a = np.expand_dims(import_image(image,
                                            image_directory=image_directory+"/",
                                            resize_dims=(input_dims[1],input_dims[0])),
                               0)
            images_file[i] = a

    images_h5py.close()
    
    






def get_vgg_deep_features(vgg_input_dims,
                          h5py_file_dest):
    
    """
    Extract image arrays from h5 file. Import VGG model and construct new model using final
    3 convolutional layer outputs of VGG model as combined output for new model. Process
    image arrays through new model to generate deep feature vectors to represent images.
    Save to h5 file.
    
    Inputs:
        vgg_input_dims: expected input dimensions for vgg model
        h5py_file_dest: h5 file location and name
    """
    
    
    
    with h5py.File(h5py_file_dest, "a") as images_h5py:
        train_images = images_h5py["img_data"].value.astype(np.float64)
        vgg = VGG19(weights='imagenet',
                     include_top=False,
                     input_shape=vgg_input_dims)

        vgg_output_layers = ['block3_conv1','block4_conv1','block5_conv1']
        vgg_conv_output = [vgg.get_layer(layer).output for layer in vgg_output_layers]
        my_model = Model(input=vgg.input, output=vgg_conv_output)

        deep_features = my_model.predict(train_images)
        for i in np.arange(len(deep_features)):
            images_h5py.create_dataset("img_deep_features_%s" % str(i), data=deep_features[i])
    images_h5py.close()
    
    
def main():
    """Main entry point for script"""
    load_images_to_h5(input_dims=(220, 176, 3),
                      image_directory='data/final_images_train',
                      destination_file="data/images_train.h5",
                      n=None)
    get_vgg_deep_features(vgg_input_dims=(220, 176, 3),
                          h5py_file_dest="data/images_train.h5")
    
    
    
if __name__ == '__main__':
    sys.exit(main())


# In[ ]:




