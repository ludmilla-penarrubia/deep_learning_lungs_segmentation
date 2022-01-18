# deep_learning_lungs_segmentation

![example](example.jpg)

## Pre-requisites and installations

* Make sure you have python3 installed  

* Clone this repository on your machine and go in it:  

    `cd deep_learning_lungs_segmentation/`  

* Create a virtual environments  

    `python3 -m venv env_seg`  

* Activate the virtual environment  

    `source env_seg/bin/activate`  

* Update pip3 repository and install dependencies listed in the requirements.txt  

    `pip3 install --upgrade pip`  
    `python3 -m pip install -r requirements.txt`  
 

## Use *our* trained model on *your* data

To predict lungs mask (on your image input_data.mhd) with the model (located at checkpoint_best.pt), in the working directory, run the following command:

    python3 trained_model_on_your_data.py -i input_data.mhd -o output_data.mhd -w checkpoint_best.pt
All paths are realtive to the working directory.
Predicted lungs mask as .mhd and .raw files will be located in : `results_showcase/`  
  


## Acknowledments

Thanks to the authors of this repository : https://github.com/milesial/Pytorch-UNet for providing an efficient implementation of U-net.  

Thanks to Olivier Bernard for getting us started with the project by providing examples of his codes.
