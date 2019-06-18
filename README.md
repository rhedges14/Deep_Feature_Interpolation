# Instructions

### Prerequisites:
- download DLIB library API for access to **get_frontal_face_detector()**; only needed for preprocessing steps

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
