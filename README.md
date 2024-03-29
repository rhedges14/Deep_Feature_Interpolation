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
- Note that DLIB's **68_face_landmarks_predictor** model is saved in the **models** subdirectory

### The following steps are meant to be run in the order that they appear
- **Note that if the only interest is to run the prediction algorithm, skip down to the "Prediction" files at the bottom**
    - "DFI_Script" (to run optimization algorithm for an individual image)
    - "Predict_DFI_mask" + "Post-Processing Mask" (to run pretrained Neural Net to apply transformation to image)


### Download Dataset:
- CelebA
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- place downloaded image directory into **data** subdirectory
- title the image directory: **celeba**
- download attribute file: "list_attr_celeba.txt" and place into **data** subdirectory (currently already in **data** subdirectory)
- download "list_eval_partition.txt" file and place into **data** subdirectory (currently already in **data** subdirectory)


### Rotate Images to Unified Template and create table of Facial Landmark Data:
- Run **Facial Landmark Detection and Alignment** notebook steps through the **Process All Images** cell
- This creates a new directory called **celeba_aigned_cropped** within your data folder with the output images


### Run "Skin Tone Extraction" file
- This step can take a long time (a full day)
- Option to exclude this step, just will have to slightly edit code in **Image Filtering to Final Dataset** file to ignore the table that is created from this file


### Run "Image Filtering to Final Dataset" file
- This file populates the image training folders based off of the preprocessing steps
- Run in jupyter notebook
- Note that if you did not run the "Skin Tone Extraction" file, you will have to make a minor tweak in code as a table will be referenced that doesn't exist
- Note that depending on what type of model you are building (clean to beard, or beard to clean), you will run 1 of the 2 final cells. Follow instructions in notebook


### Run "generate_image_data_script" python script
- This loads all images into an h5py and extracts their deep feature representations to be used during modeling

### Run "Calc_Attribute_Vector" python script
- This calculates and saves the deep feature attribute vector which will be used during training
- Be prepared to specify attribute that is to be transformed (added/removed)

### "DFI_Script"
- This is the DFI script that trains and optimizes a mask for individual images
- Note that you should be prepared to specify the name and directory of the image you wish you transform, the alpha, and the number of training iterations (12 is a good starting point for iterations)
- Output appears in a "bearded" folder; or feel free to modify destination folder


### "Fast_DFI_script"
- This is the scipt that trains a full network to output a mask
- Note that models will be saved into **models** subdirectory
- Key arguments to specify:
    - filepath for attribute vector
    - directory of training images. Note that the directory must contain two subdirectories for code to work. One of them should be empty.
    - Number of epochs
    - Whether to load weights and if so, the filepath
    - filepath to save the final model
    - number of epochs between each saving checkpoint
    - Early stopping patience
    
    The **Image Filtering to Final Dataset** file should have set up correctly for this.


### "Predict_DFI_mask"
- This is used to apply a trained Fast DFI network to "predict" and perform a transformation to a new image
- Note that output is a Mask, NOT the transformed image
- This is to be run in a Jupyter Notebook
- Key arguments to specify:
    - filepath for the model weights to be loaded in for prediction
    - image name and filepath
    - output directory to save mask

### "Post-Processing Mask" (to be run immediately after the "Predict_DFI_mask" script
- This file takes the output of the **Predict_DFI_mask** file and applies post-processing work before applying and saving a final image
- This to be run in a Jupyter Notebook
- Key arguments to specify:
    - filepath to the original image
    - filepath to the saved mask (output of **Predict_DFI_mask**)
    - Output directory
