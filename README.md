# Same same but different FCD IIb vs. TSC – distinct histomorphological features revealed with the help of deep learning
## Introduction
The FCD_Tuber_Workflow.ipynb notebook in combination with the utils helps to reproduce our pipeline for the publication above. The code provided features part of our pipeline (preprocessing, training and visualization), but not the whole analysis and end-to-end described software solution.

## System requirements

We would suggest to create a new environment with the described dependencies. 

## Hardware requirements
Our experiments were carried out on 2x NVIDIA GeForce GTX 1080Ti using an AMD Ryzen Threadripper 1950X 16x 3.40GHz, 128Gb RAM, CUDA 10.0 and cuDNN 7.

## Using the code
### Filepaths
While running the code all filepaths need to be specified as described in the notebook. 

### Tiles
Input data should be around 2000x2000 pixels in .png in 20x magnification for histopathologic scientific questions.
