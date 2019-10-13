# Same same but different: a web-based deep learning application revealed classifying features for the histopathologic distinction of cortical malformations
## bioRxiv
## DOI: 
## Introduction
This repository contains the FCD_Tuber_Workflow.ipynb notebook in combination with the utils to help reproduce our pipeline for the publication above. The code provided features part of our pipeline (preprocessing, training and visualization), but not the whole analysis and end-to-end described software solution.

## System requirements
The notebook was developed and tested on an Ubuntu 18.04 LTS Server.
We would suggest to create a new environment with the python dependencies listed in the requirements.txt.

## Hardware requirements
Our experiments were carried out on 2x NVIDIA GeForce GTX 1080Ti using an AMD Ryzen Threadripper 1950X 16x 3.40GHz, 128Gb RAM, CUDA 10.0 and cuDNN 7.

## Using the code
### Filepaths
While running the code all filepaths need to be specified as described in the notebook. 

### Using your own data
Input data should be around 2000x2000 pixels in .png in 20x magnification for scientific histopathologic questions if you want to use our random-rotate-zoom preprocessing. 
For finer Guided GradCAMs use a model trained on smaller tiles or crops of tiles (300x300).
