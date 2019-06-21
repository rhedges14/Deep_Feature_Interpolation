#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from keras.applications import VGG19
from keras import layers, models
from keras.models import Model
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py
import pandas as pd
import time
from scipy.optimize import fmin_l_bfgs_b
import imageio





def import_image(image,
                 image_directory,
                 resize_dims=(176,220)):
    
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





def dfi(test_img_name,
          test_img_dir,
          attribute_vector_file,
          alpha=4,
          train_iterations=12,
          img_height=220,
          img_width=176,
          total_variation_weight=1e-2,
          tv_exp=1.25,
          h5_feature_file="data/images_train.h5",
          destination_file='data/bearded_images/'):
    
    """
    Takes an cleanly shaven face image and outputs the same face with a beard
    
    Inputs:
        test_img_name - file name of input image
        attribute_vector_file - destination of numpy attribute_vector used for interpolation
        alpha - controls magnitude of attribute vector; larger alpha produces more extreme interpolation
        train_iterations - number of gradient descent iterations
        img_height - height of input image
        img_width - width of input image
        total_variation_weight - weight given to total variation loss expression
        tv_exp - weighting factor for invidividual pixel discrepancies in TV loss expression
        test_img_dir - directory containing input image
        h5_feature_file - h5py file containing images and corresponding deep feature vectors
        destination_file - directory to save bearded image in
    
    """

    
    # Import VGG
    vgg = VGG19(weights='imagenet',
                include_top=False,
                input_shape=(img_height, img_width, 3))
    
    # Extract Conv layers 3, 4, and 5  and their outputs to reconstruct new model
    vgg_output_layers = ["block3_conv1", "block4_conv1", "block5_conv1"]
    vgg_conv_output = [vgg.get_layer(layer).output for layer in vgg_output_layers]
    model = Model(input=vgg.input, output=vgg_conv_output)
    


    test_img_dir = test_img_dir
    test_img = test_img_name

    try:
        poc_image = import_image(image=test_img,image_directory=test_img_dir+"/")
    except:
        print("Can't locate input image in directory")

 

    # LOAD PREVIOUSLY GENERATED ATTRIBUTE VECTOR
    w = np.load(attribute_vector_file)




    
    # get tensor representations of our poc_images as well as alpha and w
    original_image = K.variable(poc_image.copy())
    a = K.variable(alpha)
    w = K.variable(w)


    # create a placeholder for our mask --> this is what will be optimized
    beard_mask = K.placeholder((1, img_height, img_width, 3))


    # create input tensor, consisting of phi(x) and phi(x+r)
    input_tensor = K.concatenate([original_image,
                                  beard_mask + original_image],
                                 axis=0)





    def calculate_loss(phi_x, phi_x_r, a, w):
        delta = phi_x_r - (phi_x + w * a)
        loss = K.sum(K.square(delta))
        return loss
    
    # encourages spatial continuity in generated pixels; avoids overly pixelated results
    def total_variation_loss(x):
        a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] - 
            x[:, 1:, :img_width - 1, :])
        b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] - 
            x[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, tv_exp))
    
    total_variation_weight = total_variation_weight # implementation from style transfer uses 1e-4

    
    # get deep feature representation of inputs
    deep_features_train = model(input_tensor)
    
    # extract outputs at 3rd, 4th, and 5th conv layers, flatten, and then concatenate
    orig_image_features = [K.expand_dims(K.flatten(f[0, :, :, :]), 0) for f in deep_features_train]
    trained_image_features = [K.expand_dims(K.flatten(f[1, :, :, :]), 0) for f in deep_features_train]

    phi_x = K.concatenate(orig_image_features)
    phi_x_r = K.concatenate(trained_image_features)
    
    # calculate loss
    loss = K.variable(0.)
    loss = loss + calculate_loss(phi_x, phi_x_r, a, w)
    loss = loss + total_variation_weight * total_variation_loss(beard_mask)

    # get the gradients of the generated image wrt the loss
    gradients = K.gradients(loss, beard_mask)

    outputs = [loss]
    if type(gradients) in {list, tuple}:
        outputs += gradients
    else:
        outputs.append(gradients)

    propogate = K.function([beard_mask], outputs)

    def eval_loss_and_gradient(x):
        x = x.reshape((1, img_height, img_width, 3))
        status = propogate([x])
        current_loss = status[0]
        current_gradient = np.array(status[1]).flatten().astype('float64')
        return current_loss, current_gradient

    # Create Evaluator object (seen in Keras documentation)
    # this allows the scipy gradient descent algorithm to calculate loss and gradient in the same pass
    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.gradient = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, gradient = eval_loss_and_gradient(x)
            self.loss_value = loss_value
            self.gradient = gradient
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            gradient = np.copy(self.gradient)
            self.loss_value = None
            self.gradient = None
            return gradient

    evaluator = Evaluator()
    
    
    x = np.zeros((1, img_height, img_width, 3))  # initialize our mask with all zeros


    for i in range(train_iterations):
        print("Start of Iteration", i)
        start_time = time.time()

        x, loss, info = fmin_l_bfgs_b(evaluator.loss,
                                      x.flatten(),
                                      fprime=evaluator.grads,
                                      maxfun=20)
        end_time = time.time()
        print("Current loss:", loss,"..........Time:", (end_time - start_time))
        
    if not os.path.exists(destination_file):
        os.makedirs(destination_file)
            
    img = x.copy()
    img = img.reshape(img_height, img_width, 3)
    fname = destination_file+test_img_name[:-4]+' a'+str(alpha)+' '+str(i)+'iters.jpg'
    imageio.imwrite(fname, poc_image[0] + img)
    
    return



def main():
    """Main entry point for script"""
    return dfi(test_img_name='166654.jpg',
               test_img_dir="data/val/no_attribute",
               attribute_vector_file='data/beard_attribute_vec.npy',
               train_iterations=3,
               alpha=4,
               total_variation_weight=1e-4,
               tv_exp=1.25)


if __name__ == '__main__':
    sys.exit(main())

