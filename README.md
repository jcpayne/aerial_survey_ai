![Elephant herd and aircraft shadow](_media/banner1.png)

# aerial_survey_ai
Code for object detection in aerial survey images

## Project Description
Our goal is to create an efficient workflow for analyzing images to detect wildlife, livestock, people, and other objects of interest from a large set of aerial survey images collected by the Tanzania Wildlife Research Institute, using AI (deep learning CNN models).  The images have some restrictions on use, and we aim to design a workflow that is open-source, transparent, robust, and can be easily re-used or adapted, while respecting those restrictions.  Please see the linked [blog](https://jcpayne.github.io/aerial-survey-blog) for more information.

## Background
The Tanzania Wildlife Research Institute wants to develop new methods for aerial survey in Africa. One of the more promising developments is the use of digital photography from the same aerial platforms (Cessnas) already in use for wildlife counts - the problem is that _analysing_ the photographs takes far longer than a traditional survey. It is hoped that ML will allow us to more rapidly process the images and make photographic counts viable.

## News/Updates
* 4/17/2020 - Data migrated from AWS to Azure (/TA25; annotation file TA25-RKE-20191128A)
* 4/22/2020 - Second batch of images and annotations added (TA25-RKE-20191201)
* 4/24/2020 - Simple model data loader working (*later abandoned)
* 5/14/2020 - Image tiling package is working, using a customized fork of a ImageBboxSlicer package.
* 5/23/2020 - Added code for handling bbox fragments when tiling.  Everything that has been annotated so far is tiled (2240 tiles).
* 6/06/2020 - TridentNet model running on VM. However, training is very slow. Starting rebuild of code to run on a Azure Machine Learning (AML) compute cluster.
* 6/20/2020 - Successful build of Dockerfile for cluster environment after many difficulties; new image sets uploaded and processed.
* 7/10/2020 - Initial runs of TridentNet model (101-layer CNN backbone) on AML cluster (NC24rs_v1); howver, we ran into quota issues and account problems.
* 7/20/2020 - New image augmentation package (imgaug) and optimizer (AdaBound) integrated into TridentNet; software problems when upgrading to multiple-GPU machines.
* 8/3/2020 - Detectron2 model working on multiple GPUs, but still slow.  Poor results; need more training data.
* 8/05/2020 - Training finally shows substantial progress after we switch to an NC24_v2 cluster (4 GPUs) and experience a 4.5X increase in speed; within 2 days we have >90% accuracy on foreground classes and 10% false negatives, training on about 6,000 images.  MAP is respectable for the classes that are better represented in the data. 
* 8/12/2020 - Second batch of images uploaded and tiled (25,000 tiles total).
* 9/6/2020 - Experimenting with doing production deployment in an Azure Containers.  Got a model running, but it was too slow.
* 11/29/2020 - Lots of work on code to do image tiling on the fly (i.e., to split a full-sized image into tiles, hold them in memory, and pass them to a dataloader without ever writing them to disk; then re-unite the individual tile results into a single annotation file for the original full-size image.  
* 12/1/2020 - Training set has finally come together after months of struggle
* 12/11/2020 - Initial training done
* 12/20/2020 - Add repeat sampling for rare classes. Working on speeding up inference.  Tiling adds complexity to the dataloader situation with multiple GPUs.
* 1/1/2021 - A full month of account problems.
* 2/11/2021 - Heavy training of the TridentNet model
* 3/10/2021 - Run batch of test images through TridentNet.  Results are good.  A competing model does not perform well.
* 3/26/2021 - First training of a model for Kazakhstan, using transfer learning from the Tanzania model
* 3/29/2021 - Benjamin Kellenberger added our trained Tridentnet model to his AIDE "Model Marketplace"
* 5/20/2021 - Built a multilabel classification model as a first filter of images, using the fastai library
* 5/27/2021 - Classification model trained (then ran into another full month of account problems.)
* 6/24/2021 - Did multiple installations of AIDE, but still no cigar.
* 7/19/2021 - Classification model does inference slowly until we discover the problem, then runs 150X faster.  Oh, joy.
* 7/20/2021 - Tiled 20,000 full-sized images from Cormon (written to disk, which took 12 hours).
* 7/21/2021 - 1.08 million tiles run through the classification model in about 2 hours.  Results are excellent.

## Project narrative
I've provided a narrative account of the development that helps to put the notebooks in perspective and to show why we made the choices we did in the linked blog https://jcpayne.github.io/aerial-survey-blog.  If the notebooks don't make sense, have a look at the blog.

## Caveat: this is an ongoing project!
This project is ongoing and unfinished.  The code is 100% Python and Linux commands, and all of the machine learning is based on Pytorch, but it is a mishmash of `Detectron2`, `fastai`, `azureml-sdk` and other libraries.  Most of my own code is in Jupyter notebooks but some is in .py files or other formats.  There is a stupid amount of file management in the notebooks due to an experimental workflow that has not yet been streamlined.  Much of my code itself is in the form of patches or functions that haven't yet been turned into proper object-oriented libraries or packaged. 

So these notebooks are not offered as a sterling example of how to do things right; they are just us sharing our ongoing journey.  I hope that you may avoid some of the pitfalls that we blundered into, and perhaps find inspiration in a few of the successes we had.

## Code & documentation
Code and documentation are presented in Jupyter Notebooks.  I used a package called [nbdev](https://nbdev.fast.ai) for some of it, which results in development notebooks that are also compiled as .py files.

**[Virtual machine setup](https://github.com/jcpayne/aerial_survey_ai/blob/master/01_dsvm_setup.ipynb.ipynb)**
- Setup of the Azure server we used, including deep learning libraries

**[Data pre-processing](https://github.com/jcpayne/aerial_survey_ai/blob/master/02_data_pre-processing.ipynb.ipynb)**
- Image types and naming conventions
- Creation of UUIDs and database for image security
- The labeling process in CVAT 
- A record of decisions regarding image classes, etc.
- Pre-processing of images to convert .NEF to .jpg and adjust contrast, resolution, size

**[Processing annotations](https://github.com/jcpayne/aerial_survey_ai/blob/master/03_process_annotations.ipynb)**
- How not to get sucked into a data management nightmare
- A counter-example of what happened to me

**[Image tiling](https://github.com/jcpayne/aerial_survey_ai/blob/master/04_tile_training_images.ipynb)**
- A workflow for cutting large images and their bounding boxes into smaller tiles, padding the images as necessary, allowing overlap between tiles, and enabling the user to sample empty tiles.
- The workflow is used to tile the training and validation sets

**[Object detection using AI on AML cluster](https://github.com/jcpayne/aerial_survey_ai/blob/master/05_aml-pipeline.ipynb)**
- Code for running a TridentNet model, which is an "Faster R-CNN" model from Facebook Research (a fork of Detectron2) on Azure Machine Learning Services.  Customizations include dataloaders, image augmentation, and optimizer, in-memory tiling, reassembly of tiled annotations, and more.  The notebook runs via the AML Python SDK, and it calls resources via `azureml`, including:
    - [a Dockerfile](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dockerfiles/Dockerfile_ub1804cuda101_v17) used to create the container environment that is run on the AML cluster.
    - [distrib_run_and_inference.py](https://github.com/jcpayne/aerial_survey_ai/blob/master/distrib_run_and_inference.py); the main Python script for running the Detectron2 model;
    - [development packages](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dev_packages) containing 
        - **[TridentNet](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dev_packages/TridentNet)** (which is not available as a PYPI package)
        - **[pascal-voc-writer](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dev_packages/pascal_voc_writer)** A lightly modified annotation file writer
        - **[ImageBboxTiler](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dev_packages/image_bbox_tiler)**, a heavily-modified fork of the `ImageBboxSlicer package`
        - **[trident_dev](https://github.com/jcpayne/aerial_survey_ai/blob/master/trident_project/dev_packages/trident_dev)**; the rest of the code I wrote for this project (also unpackaged)
    
**[Training a multi-label classification model](https://github.com/jcpayne/aerial_survey_ai/blob/master/06_fastai_model.ipynb)**
- Build a multi-label classification model (a ResNet50 from fastai) to act as a preliminary filter
- Train and re-train it (transfer learning)
- Do inference on a small batch
- Develop some functions for performance assessment

**[Inference with the multi-label classification model](https://github.com/jcpayne/aerial_survey_ai/blob/master/07_fastai_inference.ipynb)**
- Run a batch of 1 million tiles through the classification model
- Assess the results

### RProject
A `.RProj` file is present for the convenience of fiddling about in R with the various files. The `.gitingore` was also updated to ignore userdata / rdata stuff which would otherwise mess with the git structures (lots of invisible files) - what this means is that you should always re-run code to recreate your local environment if there have been changes (which is good practice anyway).

## References (BibTex)
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}

## Installation
Our code is not yet installable...but I'm working on it.  

## Contributors
John Payne wrote the notebooks, notes, and other code in this repository that are focused on AI.
Howard Frederick runs the larger project.  He ran the survey teams that collected the data used here and set up and directed a photographic analysis lab team in Tanzania who created the annotations used.  He participated in all of the strategic decisions and model performance assessments, and we shared our Microsoft Azure resources for the project.

## Citation
Please cite this work as follows:
Payne, J.C. and H.L. Frederick 2021.  Object detection in aerial surveys.  URL: https://github.com/jcpayne/aerial_survey_ai

## License
- **Code** is open-source.  
- **Images**  Explain restrictions on image use here
