# Deep Feature Interpolation for Facial Hair Transformation

**Project Status: Completed**

### Project Objective
The objective of this project is to leverage Deep Feature Interpolation (DFI) techniques to apply facial hair transformations to facial images. Multiple implementations will be explored. An approach is first constructed that involves an individual training optimization phase for each image before extending the project to an enhanced implementation where a single neural network will be trained per transformation that has the ability to generalize to any input image, thus providing for a functionality where the transformation can render in a matter of seconds.

### Methods Used
- Machine Learning
- Deep Learning
- Machine Vision

### Technologies
- Python
- Jupyter Notebooks
- Tensorflow and Keras
- opencv

### Executive Summary
Deep Feature Interpolation (DFI) is a method that leverages feature representations to apply image transformations via an interpolation vector. 

For this project, a DFI algorithm is implemented to transform facial hair. In particular, the algorithm is trained to add and remove attributes such as beards and mustaches. 

Additionally, a Fast DFI algorithm is created based on work in Fast Style Transfer, where a network is trained with the ability to perform a single forward pass on an image to generate a facial transformation mask. Various nuances and additional enhancements to further improve the algorithm are also discussed.


# Instructions

### Prerequisites:
- download DLIB library API for access to **get_frontal_face_detector()**

### The following steps are meant to be run in the order that they appear

### Download Dataset:
- CelebA
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- place downloaded image directory into **data** subdirectory
- title the image directory: **celeba**
- download attribute file: "list_attr_celeba.txt" and place into **data** subdirectory
- download "list_eval_partition.txt" file and place into **data** subdirectory


### Rotate Images to Unified Template:
- Run **Facial Landmark Detection and Alignment** notebook steps through the **Process All Images** cell
- This creates a new directory called **celeba_aigned_cropped** within your data folder with the output images


### Run "Skin Tone Extraction" file
- This step can take a long time (a full day)
- Option to exclude this step, just will have to slightly edit code in **Image Filtering to Final Dataset** file to ignore this piece


### Run "Image Filtering to Final Dataset" file
- This file populates our image training folders based off of our preprocessing steps


### Run "generate_image_data_script" python script
- this loads all images into an h5py and extracts their deep feature representations

### Run "Calc_Attribute_Vector" python script
- this calculates and saves the deep feature attribute vector which will be used during training


### "DFI Script"
- This is the DFI script that trains and optimizes a mask for individual images
- Note that you should be prepared to specify in the argument the name and directory of the image you wish you transform
- Output appears in a "bearded" folder


### "Fast_DFI_script"
- This is the scipt that trains a full network to output a mask
- Note that models will be saved into **models** subdirectory
- Be prepared to specify the directory of training images. Note that the directory must contain two subdirectories for code to work. One of them can be empty. The **Image Filtering to Final Dataset** file should have set up correctly for this.


### "Predict_DFI_script"
- This is used to use trained Fast DFI networks to apply transformations to new images
- Be prepared to input the name and directory of the image you wish you transform
