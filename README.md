![Elephant herd and aircraft shadow](_media/banner1.png)

# aerial_survey_ai
Code for object detection in aerial survey images

## Project Description
Our goal is to create an efficient workflow for analyzing images to detect wildlife, livestock, people, and other objects of interest from a large set of aerial survey images collected by the Tanzania Wildlife Research Institute, using AI (deep learning CNN models).  The images have some restrictions on use, and we aim to design a workflow that is open-source, transparent, robust, and can be easily re-used or adapted, while respecting those restrictions.

## Background (HF)
The Tanzania Wildlife Research Institute wants to develop new methods for aerial survey in Africa. One of the more promising developments is the use of digital photography from the same aerial platforms (Cessnas) already in use for wildlife counts - the problem is that _analysing_ the photographs takes far longer than a traditional survey. It is hoped that ML will allow us to more rapidly process the images and make photographic counts viable.

## News/Updates
Just major developments
* 8/12/2020 - Second batch of images uploaded and tiled (25,000 tiles total).
* 8/05/2020 - Training finally shows substantial progress after we switch to an NC24_v2 cluster (4 GPUs) and experience a 4.5X increase in speed; within 2 days we have >90% accuracy on foreground classes and 10% false negatives, training on about 6,000 images.  MAP is respectable for the classes that are better represented in the data. 
* 7/20/2020 - New image augmentation package (imgaug) and optimizer (AdaBound) integrated into TridentNet;
* 7/10/2020 - Initial runs of larger model (101-layer CNN backbone) on AML cluster (NC24rs_v1); howver, we ran into quota and software problems when upgrading to multiple-GPU machines.
* 6/20/2020 - Successful build of Dockerfile for cluster environment after many difficulties; new image sets uploaded and processed.
* 6/06/2020 - TridentNet model running on VM. However, training is very slow. Starting rebuild of code to run on a Azure Machine Learning Services compute cluster instead of a VM.
* 5/23/2020 - Added code for handling bbox fragments.  Have tiled everything that has been annotated so far (2240 tiles)
* 5/14/2020 - Image tiling package is working, using a customized fork of a ImageBboxSlicer package.
* 4/17/2020 - Data migrated from AWS to Azure (/TA25; annotation file TA25-RKE-20191128A)
* 4/22/2020 - Images from second annotation zipfile added (TA25-RKE-20191201)
* 4/24/2020 - fastai data loader working

## Code & documentation
Code and documentation are presented in Jupyter Notebooks:

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

**[Image tiling](https://github.com/jcpayne/aerial_survey_ai/blob/master/image_tiling_v3.ipynb)**
- A workflow for unzipping and sorting annotation files, and cutting large images and their bounding boxes into smaller tiles, padding the images as necessary, allowing overlap between tiles, and enabling the user to sample empty tiles.  It is a fork of the ImageBboxSlicer package.

**[Object detection using AI on AML cluster](https://github.com/jcpayne/aerial_survey_ai/blob/master/aml-pipeline-run.ipynb)**
- Code for running a TridentNet model, which is an "Faster R-CNN" model from Facebook Research (a fork of Detectron2) on Azure Machine Learning Services.  Customizations include dataloaders, image augmentation, and optimizer.  The notebook runs via the AML Python SDK, and it calls resources via `azureml`, including:
    - [a Dockerfile](https://github.com/jcpayne/aerial_survey_ai/blob/master/resources/trident_run.py) used to create the container environment that is run on the AML cluster.
    - [trident_run.py](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_run.py); the main script for running the model;
    - [development packages](https://github.com/jcpayne/aerial_survey_ai/blob/master/dev_packages) containing `TridentNet` (which is not available as a PYPI package), a lightly modified annotation file writer (`pascal-voc-writer`), and `ImageBboxTiler`, a heavily-modified fork of the `ImageBboxSlicer package`.
    
**[Object detection using AI on VM](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_tz_on_vm.ipynb)**
- For testing and building functionality.  Includes code for:
    - running TridentNet from the commandline; 
    - calculating the per-channel mean and standard deviation for a potentially enormous set of images using a non-overflowing incremental calculation;
    - doing Coco-style evaluation;
    - and hand-building a confusion matrix.  
Our VM runs CUDA 10.2, so this code is not 100% compatible wtih the version that is run on the AML cluster, which has CUDA 10.1.

### RProject
A `.RProj` file is present for the convenience of fiddling about in R with the various files. The `.gitingore` was also updated to ignore userdata / rdata stuff which would otherwise mess with the git structures (lots of invisible files) - what this means is that you should always re-run code to recreate your local environment if there have been changes (which is good practice anyway).

## References
As needed

## Installation
We could use nbdev to turn some of the notebooks into Python packages, but much of the code/docs won't be directly runnable.

## Contributors
John Payne & Howard Frederick (links, etc.)

## Citation
Please cite this work as follows:
...

## License
- **Code** is open-source (...)
- **Images**  Explain restrictions on image use here
