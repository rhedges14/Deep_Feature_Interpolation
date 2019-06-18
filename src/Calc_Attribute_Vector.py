#!/usr/bin/env python
# coding: utf-8

# In[5]:


from keras.applications import VGG19
import h5py
import numpy as np
import pandas as pd

def calc_attribute_vector(dest_file,
                          img_height=220,
                          img_width=176,
                          attribute_reference_csv="data/attribute_reference_df.csv",
                          h5_feature_file="data/images_train.h5",
                          vgg19_weights_file="models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
                          vgg_output_layers=["block3_conv1", "block4_conv1", "block5_conv1"],
                          vgg_output_layer_weights=[1.0, 1.0, 1.0]):
    """
    Calculates the deep feature attribute vector for a given transformation
    
    Inputs:
        img_height - height of input image
        img_width - height of output image
        vgg_output_layers - list of vgg19 layers to be used for deep feature representation
        vgg_output_layer_weights - list of vgg19 layers to be used for deep feature representation
        h5_feature_file
        attribute_reference_csv
        
    """
    vgg = VGG19(weights='imagenet',
                include_top=False,
                input_shape=(img_height, img_width, 3))

    # Extract Conv layers and their outputs to reconstruct new model
    vgg_output_layers = vgg_output_layers

    with h5py.File(h5_feature_file, "a") as images_h5py:

        # extract indices for our groups; cap based on size of h5 file
        try:
            data_df = pd.read_csv(attribute_reference_csv)
        except:
            print("No Index Table Found for Attribute Groupings")

        data_train_df = data_df[data_df['Group'] == 0].iloc[:images_h5py['img_data'].shape[0]]

        yes_attribute_index = data_train_df[(data_train_df['Beard'] == 1) & # make this an input
                                      (data_train_df['5_o_Clock_Shadow'] == 0)].index
        no_attribute_index = data_train_df[data_train_df['Beard'] == 0].index


        # Get the deep features, flatten and concatenate
        Deep_vgg_yes_attribute = []
        for i, layer in enumerate(range(len(vgg_output_layers))): # change to length of layer
            images = images_h5py["img_deep_features_%s" % layer][yes_attribute_index] * vgg_output_layer_weights[i]
            flat_images = images.reshape(len(yes_attribute_index), -1)
            Deep_vgg_yes_attribute.append(flat_images)

        Deep_vgg_no_attribute = []
        for i, layer in enumerate(range(len(vgg_output_layers))):
            images = images_h5py["img_deep_features_%s" % layer][no_attribute_index] * vgg_output_layer_weights[i]
            flat_images = images.reshape(len(no_attribute_index), -1)
            Deep_vgg_no_attribute.append(flat_images)

        Deep_vgg_yes_attribute = np.concatenate(Deep_vgg_yes_attribute, 1)
        Deep_vgg_no_attribute = np.concatenate(Deep_vgg_no_attribute, 1)


    # Get the mean VGG feature for starting and target group
    Deep_yes_attribute_centroid = np.mean(Deep_vgg_yes_attribute, axis=0)
    Deep_no_attribute_centroid = np.mean(Deep_vgg_no_attribute, axis=0)


    # Compute w that goes from starting to target group
    w = Deep_yes_attribute_centroid - Deep_no_attribute_centroid
    # SAVE W VECTORS ONE FOR EACH COMBINATION OF GROUP
    print('Attribute Vector Loaded')

    np.save(dest_file, w)
    



def main():
    """Main entry point for script"""
    calc_attribute_vector(dest_file='data/beard_attribute_vec.npy')

    
    
    
if __name__ == '__main__':
    sys.exit(main())

