# aerial_survey_ai
Code for object detection in aerial survey images

## Project Description
Our goal is to create an efficient workflow for analyzing images to detect wildlife, livestock, people, and other objects of interest from a large set of aerial survey images collected by the Tanzanian Wildlife Service, using AI (deep learning CNN models).  The images have some restrictions on use, and we aim to design a workflow that is open-source, transparent, robust, and can be easily re-used or adapted, while respecting those restrictions.

## Background (HF)
Background on the project (future paper intro).  This is but a small corner of the vast Frederick Empire...

## News/Updates
Just major developments 

## Code & documentation
This repository includes the following Jupyter Notebooks:

**[Server Setup](https://github.com/jcpayne/aerial_survey_ai/blob/master/server_setup.ipynb)**
- Setup of the Azure server we used, including deep learning libraries

**[Data preparation workflow](https://github.com/jcpayne/aerial_survey_ai/blob/master/data_preparation_workflow.ipynb)**
- Image types and naming conventions
- Migration of files from AWS to Azure
- Creation of UUIDs and database for image security
- Pre-processing of images to convert .NEF to .jpg and adjust contrast, resolution, size

**[Data labeling](https://github.com/jcpayne/aerial_survey_ai/blob/master/data_labeling.ipynb)**
- Labeling process in CVAT 
- A record of decisions regarding image classes, etc.

**[Object detection using AI]** #(https://github.com/jcpayne/aerial_survey_ai/blob/master)**
- Code for a "Faster R-CNN" model, using fastai2 for data loading and to control training
- Potentially, Detectron2 models either with fastai2 wrapper or directly in PyTorch

## Citations/Papers
As needed

## Installation
We could use nbdev to turn some of the notebooks into a Python package in one step, but much of the stuff won't be directly runnable.

## Contributors
John Payne & Howard Frederick (links, etc.)

## License
- **Code** is open-source (...)
- **Images**  Explain restrictions on images here