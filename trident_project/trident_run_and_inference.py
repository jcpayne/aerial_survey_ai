#TESTED AND WORKING
#import sys
#sys.path.insert(1, '.')
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
print('Testing if I can load imports')
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import argparse
import numpy as np
import cv2
import random
import logging
import json
from PIL import Image
from collections import OrderedDict
import copy
import itertools
from time import time
from collections import Counter

#from google.colab.patches import cv2_imshow
from pathlib import Path
from fvcore.common.file_io import PathManager
from torch import randperm,load,nn
import torch.nn.functional as F
from torchvision import transforms as transforms
import PIL
#from sklearn import metrics
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
from adabound import AdaBound
from typing import List, Optional, Union

# Import Detectron2 structures and functions
from detectron2.config import get_cfg, configurable
from detectron2.structures import Boxes, BoxMode, pairwise_iou 
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper,build_detection_test_loader
from detectron2.data import DatasetFromList, MapDataset, detection_utils as utils, get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader, trivial_batch_collator
from detectron2.data.transforms import TransformGen
from fvcore.transforms.transform import (BlendTransform,NoOpTransform,Transform)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset,DatasetEvaluator,DatasetEvaluators,COCOEvaluator
from detectron2.evaluation import inference_context
from detectron2.modeling.matcher import Matcher
from detectron2.solver.build import *
from detectron2.structures import Instances, Boxes
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

print("Trying tridentnet import")
from dev_packages.TridentNet.tridentnet import add_tridentnet_config
from dev_packages.pascal_voc_writer.pascal_voc_writer import Writer

from azureml.core.run import Run
print("All imports passed")

def get_annotation_files(annotationpath):
    """Returns full filepaths for .xml files in the annotationpath."""
    annotation_files = [str(x) for x in annotationpath.iterdir() if x.suffix == '.xml']
    return (annotation_files)

def count_classes(annotation_files):
    """Count object classes in annotation files"""
    counts = {}
    for f in annotation_files:
        tree = ET.parse(f)
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in counts:
                counts[cls] += 1
            else:
                counts[cls] = 1
    return(list(counts.items()))

def load_voc_instances(rootdir: str, split: str,CLASS_NAMES):
    """
    Load VOC annotations to Detectron2 format.  It takes filenames from train.txt or valid.txt and joins them to a root dir
    to get filenames for annotation and image files.  It does not use the filename or path from the XML annotations.
    From the XML annotation, the only things it uses are height, width, and objects (class, plus bndbox incl. coordinates).
    Any classname not in CLASS_NAMES is renamed to "other_animal" inside the model.
    Note: train.txt and valid.txt (in rootdir) contain image filenames (without extension or path).
    Args:
        rootdir: Root directory, under which are "tiled_annotations" and "tiled_images" subdirectories
        split (str): one of "train", "valid"
        CLASS_NAMES: a set of valid classnames
    """
    #Open the train.txt or valid.txt files
    with PathManager.open(os.path.join(rootdir, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    #Randomly shuffle the order of the training files on each run (leave validation file order unchanged)
    if (split=='train'):
        random.shuffle(fileids)

    dicts = []
    #Note: fileid is the filename stem (no path or suffix) from train.txt or valid.txt.
    #anno_file and jpeg_file just join that stem to a root directory and give it a suffix.
    for fileid in fileids:
        anno_file = os.path.join(rootdir, "tiled_annotations", fileid + ".xml")
        jpeg_file = os.path.join(rootdir, "tiled_images", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            #Set all classnames that aren't in the classnames list to "other" 
            if not cls in CLASS_NAMES:
                cls = "other_animal"
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

#Split filenames into train and validate sets (copied from fastai2)
def split_by_random(o,valid_pct):
    rand_idx = [int(i) for i in randperm(len(o))]
    cut = int(valid_pct * len(o))
    return rand_idx[cut:],rand_idx[:cut]

def write_train_validate_files(rootdir,annotation_files,train_idxs,val_idxs):
    """Writes two text files to the root directory that define training and validation data sets.
       They contain filenames without path or suffix.
       """
    trfile = open(str(rootdir/'train.txt'),'w')
    for idx in train_idxs:
        fname = Path(annotation_files[idx]).stem
        trfile.write(fname)
        trfile.write('\n')
    trfile.close()
    
    valfile = open(str(rootdir/'valid.txt'),'w')
    for idx in val_idxs:
        fname = Path(annotation_files[idx]).stem
        valfile.write(fname)
        valfile.write('\n')
    valfile.close()
    
def register_datasets(rootdir, CLASS_NAMES):
    #I modified this to pop a name from the _REGISTERED dict if it already exists    
    for d in ["train", "valid"]:
        dsetname = "survey_" + d
        if dsetname in DatasetCatalog._REGISTERED.keys():
            DatasetCatalog._REGISTERED.pop(dsetname)
        DatasetCatalog.register("survey_" + d, lambda d=d: load_voc_instances(rootdir, d,CLASS_NAMES))
        MetadataCatalog.get("survey_" + d).set(thing_classes=CLASS_NAMES)
    survey_metadata = MetadataCatalog.get("survey_train")

# def setup(args):
#     """
#     Create configs and perform basic setups 
#     """
#     cfg = get_cfg()
#     add_tridentnet_config(cfg) #Function is from tridentnet.config.py
#     #cfg.merge_from_file(args.config_file) #optional
#     #cfg.merge_from_list(args.opts) #optional
#     cfg.freeze() #
    
#     return cfg
    
#WARNING: if you mis-format a value, it will be silently ignored (i.e., use 'key:value' instead of 'key=value')
def setup_model_configuration(rootdir, output_dir, CLASS_NAMES):
    cfg = get_cfg()
    add_tridentnet_config(cfg) #Loads a few values that distinguish TridentNet from Detectron2
    
    #This didn't work
    #config_file = '/home/egdod/detectron2/projects/TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml'
    #config_file = '/home/egdod/detectron2/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml'
    #config_file = './dev_packages/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml'
    #cfg.merge_from_file(config_file)
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts) #this might work -- not tried

    #OPTIONAL [WARNING!]  Specify model weights
    #cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl" #original weights
    cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth") #This is in blobstore/temp directory
    
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default is 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    
    #KEEP. Default Solver specs. (Note; 3x seems to require 3 times as many steps as 1x)
    #Base-TridentNet-Fast-C4.yaml 60,000-80,000 steps and 90,000 max_iter.
    #tridentnet_fast_R_50_C4_1x.yaml SAME
    #tridentnet_fast_R_50_C4_3x.yaml 210,000 - 250,000 steps and 270,000 max_iter.
    #tridentnet_fast_R_101_C4_3x.yaml SAME      
    
    #On a V2 machine, it takes about 9 hours to do 25,000 steps with a 500x500 tile.
    #BUT on a v3 machine it takes 1.3 hours to do 1,000 steps, i.e. 9 hours = 13,500 steps.  Ugh.
    #cfg.SOLVER.STEPS = (60000, 80000) #for base model
    #cfg.SOLVER.MAX_ITER = 90000  
    #cfg.LR_SCHEDULER='WarmupCosineLR'
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    #You idiot!  The steps are the places where the default "WarmupMultiStepLR" scheduler drops the learning rate by gamma=0.1.
    cfg.SOLVER.STEPS = (18000,)#If one value, then trailing comma is required (must be iterable) #(210000, 250000) for trident
    cfg.SOLVER.MAX_ITER = 20000 #25000 #  270000 for trident (that's 270,000!)
    cfg.SOLVER.WARMUP_ITERS = 2000 #1000 is default
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000 #2500
    
    #Learning Rates
    #From Goyal et al. 2017: Linear Scaling Rule: When the minibatch size is multiplied by k, 
    #multiply the learning rate by k.  i.e., as you increase the batch size because of using 
    #additional GPUs, increase the learning rate too.  Works up to very large batches (8,000 images)
    #See auto_scale_workers() in Detectron2 (very important!)
    cfg.SOLVER.BASE_LR = 0.0003 #Is .001 in defaults.py but .02 in tridentnet, but they trained on 8 GPUs
    
    #Pixel means are from 19261 500x500 tiles on Aug 15 2020 (train_dict)
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [143.078, 140.690, 120.606] #first 5K batch was [137.473, 139.769, 106.912] #default was [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [34.139, 29.849, 31.695]#first 5k batch was [26.428, 22.083, 26.153]
        
    #Auugmentation. Add corruption to images with probability p
    cfg.INPUT.corruption = 0.1
    
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800) #My version for 500x500 tiles:(400,420,440,460,480,500) 
    cfg.INPUT.MIN_SIZE_TEST = 800
    
    cfg.DATASETS.TRAIN = ("survey_train",) #Why the comma???  Bcs you can have multiple training datasets
    cfg.DATASETS.TEST = ("survey_valid",)

    cfg.DATALOADER.NUM_WORKERS = 24 #Set to equal the number of CPUs.

    # if True, the dataloader will filter out images that have no associated
    # annotations at train time.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = output_dir
      
    #FOR DOING INFERENCE ON GPU/CPU
    cfg.MODEL.DEVICE='cuda' #'cuda' or 'cpu'
    
    # Specify the test branch index TridentNet Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    cfg.MODEL.TRIDENT.TEST_BRANCH_IDX = 1

    #WARNING. #I think freeze() makes the config immutable; hence it must come last
    cfg.freeze() 

    return cfg

def register_test_dataset(testdict,CLASSNAMES):
    #I modified this to pop a name from the _REGISTERED dict if it already exists    
    if "test" in DatasetCatalog._REGISTERED.keys():
        DatasetCatalog._REGISTERED.pop("test")
    #Register a name with a function to retrieve the dataset
    DatasetCatalog.register("test", lambda: testdict)
    MetadataCatalog.get("test").set(thing_classes=CLASS_NAMES)

def calc_padding_for_fpad(img_size,tile_size,tile_overlap):
    """
    Calculates the padding needed to make an image an even multiple of tile size, given overlap 
    between tiles.  This version of the function is designed for input to torch F.pad().
    NOTE: With overlap, final dimensions are hard to visualize. The correct formula for width is 
        (((img_width // stride_w)-1) * stride_w) + tile_w
    Height is analogous.  In other words, you sum the stride for all tiles except the last tile,
    for which you add the full tile width.

    Parameters
    ---------------
    img_size: tuple(height, width) in pixels
    tile_size: tuple (height, width) in pixels
    tile_overlap: int. Tile overlap between consecutive strides in pixels (the same number is used for height and width).
    Returns:
    ---------------
    padding.  A tuple (left, right, top, bottom) of padding in pixels for the sides of an image.  
                Padding is added to right and bottom only.
    """
    #Extract params
    img_height,img_width = img_size
    tile_height,tile_width = tile_size

    #Tile overlap in pixels (same for H and W)
    tile_w_overlap = tile_overlap
    tile_h_overlap = tile_overlap

    #Calculate stride in pixels
    tile_w_stride = tile_width - tile_w_overlap
    tile_h_stride = tile_height - tile_h_overlap

    #Calculate amount to add (in pixels)
    delta_w = tile_width - (img_width % tile_w_stride) 
    delta_h = tile_height - (img_height % tile_h_stride)

    #Adjust if deltas are > tile size
    if delta_w >= tile_width: 
        delta_w = delta_w - tile_width
    if delta_h >= tile_height:
        delta_h = delta_h - tile_height

    #Padding (left, right, top, bottom) in pixels.
    padding = (0,delta_w,0,delta_h)
    return padding

def get_tiles(full_image, tile_size, tile_overlap,print_padding=False):
    """
    Cuts a full-size image into tiles, potentially overlapping.  It first
    pads the image with black on the right and bottom edges as necessary to
    get a set of equal-sized tiles.
    Parameters:
        full_image: (Numpy array [H, W, C] where C is RGB).
        tile_size: (int, int) (H, W) in pixels
        tile_overlap: int in pixels
    Returns a Pytorch array of tiles.  Dimensions: torch.Size([1, rows, cols, channels, tile_h, tile_w]).
    """
    tile_height, tile_width = tile_size
    tile_overlap = tile_overlap

    step_h = tile_height - tile_overlap #aka "stride"
    step_w = tile_width - tile_overlap  

    img_size = full_image.shape[1:] # H, W

    #Get padding
    padding = calc_padding_for_fpad (img_size,tile_size,tile_overlap)
    padded_img = F.pad(full_image,padding) # (L,R,top,bottom)
    if (print_padding):
        print(full_image.shape, padding,padded_img.shape)

    #Unfold parameters are (dimension, kernel, stride)
    #This splits the image into patches, based on kernels and strides
    tiles = padded_img.unfold(0,3,3).unfold(1, tile_height, step_h).unfold(2, tile_width, step_w)
    return tiles

def get_tile_offsets(trow,tcol,tile_size,overlap):
    """Calulates X and Y offsets in pixels for a particular tile.  Used for reassembling tile output.
        Parameters:
        trow, tcol: int, int.  Row and column number for the tile
        tile_size: tuple. (height, width) in pixels
        overlap: int.  Tile overlap in pixels
    """
    tile_h, tile_w = tile_size
    row_offset = trow * (tile_h - overlap)
    col_offset = tcol * (tile_w - overlap)
    return (row_offset,col_offset)

def combine_tile_annotations(tile_results,tile_size,overlap,fullimage_size):
    """Reassemble annotations for individual tiles into an annotation for the original file.
       Duplication of boxes due to tile overlap is ignored (all boxes are returned, regardless of overlap).
       Parameters:
       tile_results: OrderedDict.  Results returned by the model (an OrderedDict with one element called 'Predictions')
       tile_size: int.  tile size in pixels
       overlap: int. tile overlap in pixels
       fullimage_size: tuple.  (height, width) of original image
       
       Returns: a dict with the combined Instances objects (boxes, scores, predicted classes)
    """
    print("Reassembling tile annotations")
    tile_predictions = tile_results['Predictions']
    firsttile = True
    for tres in tile_predictions:
        trow = tres["trow"]
        tcol = tres["tcol"]
        row_offset,col_offset = get_tile_offsets(trow,tcol,tile_size,overlap) #use regex         
        tinst = tres['instances']
        tboxes = tinst.pred_boxes.tensor
        tscores = tinst.scores
        tclasses = tinst.pred_classes
        #Adjust boxes by tile offset
        N = tboxes.shape[0]
        for r in range(N):
            tboxes[r,0] += col_offset
            tboxes[r,2] += col_offset
            tboxes[r,1] += row_offset
            tboxes[r,3] += row_offset
        if firsttile:
            master_boxes = tboxes
            master_scores = tscores
            master_classes = tclasses
            firsttile = False
        else:
            master_boxes = torch.cat((master_boxes, tboxes), 0)
            master_scores = torch.cat((master_scores, tscores), 0)
            master_classes = torch.cat((master_classes, tclasses), 0)
    master_instances = Instances(fullimage_size) #fullimage_size is a tuple (ht,width)
    master_instances.pred_boxes = Boxes(master_boxes.cpu())
    master_instances.scores = master_scores.cpu()
    master_instances.pred_classes = master_classes.cpu()
    return {"instances":master_instances}

def write_results_to_xml(result_dict,outdir,CLASS_NAMES): 
    """
    Writes a dict containing annotations in Detectron2 format to PASCAL-VOC xml-style annotation files, 
    one xml file per image file, using the pascal_voc_writer package.  
    Also writes a summary csv file with number of instances of each class detected per image.
    Parameters:
    resultdict: dict in Detectron2 annotation format
    outdir: str Output directory (files will be written here)
    """
    nfiles = len(result_dict)
    #Set up a dict for tracking class counts (we'll turn into a dataframe) 
    class_counts = {"file_name":np.zeros(nfiles).tolist()}
    for cn in range(len(CLASS_NAMES)):
        class_counts[CLASS_NAMES[cn]] = np.zeros(nfiles).tolist()

    n_written = 0
    for i in range(nfiles):
        #Get contents of the dict
        d = result_dict[i]
        filename = d["file_name"]
        filestem = Path(filename).stem
        height,width = d["height"], d["width"]
        instances = d["instances"]
        boxes = instances.pred_boxes.tensor
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        
        #Count the number of occurrences of each class in an incrementing list (for a summary)
        valcount = Counter(classes)
        class_counts["file_name"][i] = filename
        for k,v in valcount.items():
            class_counts[CLASS_NAMES[k]][i] += v
        
        writer = Writer(filename, width, height) # Writer(image_filepath, image_width, image_height)
        #writer.changePath(relpath)
        #writer.changeFolder(folder)

        assert ((boxes.shape[0]==len(scores)) & (len(scores)==len(classes))), 'Lengths of boxes, scores and classes dont match'
        for inst in range(len(scores)):
            xmin, ymin, xmax, ymax = boxes[inst].tolist()
            name = CLASS_NAMES[classes[inst]] #convert integer back to text for classname
            score = scores[inst]
            writer.addObject(name, xmin, ymin, xmax, ymax, score=score, pose='Unspecified', truncated=0, difficult=0)
        outfile = str((Path(outdir)/filestem).with_suffix('.xml'))
        writer.save(outfile)
        n_written +=1
    #Convert to dataframe and write csv file
    ccounts = pd.DataFrame.from_dict(class_counts,'columns')
    ccounts.to_csv(str(Path(outdir)/'class_counts.csv'), sep=',', encoding='utf-8',index=False)
    print("   Created " + str(n_written) + " new annotation files; ")
    return(class_counts)


#Overriding the default method (in detectron2/data/detection_utils.py) to add image corruption as an augmentation.
def build_augmentation(cfg, is_train):
    #logger = logging.getLogger(__name__)

    result = utils.build_augmentation(cfg, is_train) #Returns list[Augmentation(s)] containing default behavior

    if is_train:
        random_corruption = RandomCorruption(cfg.INPUT.corruption) #prob of adding corruption ([0,1])
        result.append(random_corruption)
        print("Random corruption augmentation used in training")

        #logger.info("Random corruption augmentation used in training: " + str(random_corruption))
        print(result)
    return result

class RandomCorruption(TransformGen):
    """
    Randomly transforms image corruption using the 'imgaug' package 
    (which is only guaranteed to work for uint8 images).  
    Returns an Numpy ndarray.
    """

    def __init__(self, p):
        """
        Args: 
            p probability of applying corruption (p is on [0,1]) 
        """
        super().__init__()
        self._init(locals())
        self.p = p

    def get_transform(self, img):
        r = random.random()
        if(r <= self.p):
            #A selection of effects from imgaug
            #ia.seed(None)
            severity = random.randint(1,5)
            augmenter_list = [
                iaa.BlendAlphaSimplexNoise(
                    foreground=iaa.EdgeDetect(alpha=(0.5, 0.9)),
                    background=iaa.LinearContrast((0.5, 0.2)),
                    per_channel=0.5),
                iaa.CoarseDropout(p=0.25, size_px=8),
                iaa.imgcorruptlike.GaussianBlur(severity),
                iaa.imgcorruptlike.SpeckleNoise(severity),
                iaa.Cutout(fill_mode="gaussian", fill_per_channel=True,nb_iterations=(1, 5), size=0.2, squared=False),
                iaa.imgcorruptlike.Spatter(severity)]
            #Blend noise with the source image
            augmenter = random.choice(augmenter_list)
            blended_img = augmenter.augment_image(img)
            return BlendTransform(src_image=blended_img, src_weight=1, dst_weight=0)
        else:
            return(NoOpTransform())

class TridentDatasetMapper(DatasetMapper):
    """
    A customized version of DatasetMapper.  
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and maps it into a format used by the model.

    """        
    #The only change I made is to switch build_augmentation for utils.build_augmentation
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret
    
class TileDatasetMapper(DatasetMapper):
    """
    Just like DatasetMapper except instead of opening the file, it is passed a dict containing already-opened file.
    See "configurable": the __init__ class is decorated with @configurable so you can pass
    a cfgNode object and it will use the from_config() method for initiation.
    
    1. Accept an opened image as a Pytorch array
    2. Potentially applies cropping/geometric transforms to the image and annotations
    """
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False
    ):
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = False):
        augs = utils.build_augmentation(cfg, is_train) #returns T.ResizeShortestEdge, plus optionally others
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         # USER: Write your own image loading if it's not from a file
#         # We just use the image that is passed in, instead of opening the file and converting
#         #The returned dict has 2 things in it: file_name and image (which contains a tensor).  That's all.
#         image = dataset_dict["image"]
        
#         image_shape = image.shape[1:]  # h, w

        if not self.is_train:
            return dataset_dict

def build_inference_loader(cfg, dataset_name, dataset_dicts, mapper=None):
    """
    A modified version of the Detectron2 function. Takes a list of dicts as input 
    instead of creating it from a list of files.
    Arguments:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        dataset_dicts (list): A list of dicts (one per input image) in the Detectron2 format.
        mapper (callable): a callable which takes a sample (dict) from dataset
            and returns the format to be consumed by the model.
    Returns:
        DataLoader: a torch DataLoader.
    """
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False) #For ONE image: open file, permute, do transform, 
    else:
        mapper = mapper(cfg, False) #should be TileDatasetMapper
    print("Mapper:", type(mapper))
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Allow a batch size of > 1 for speed (since we don't care about papers!)
#    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False) #original -- 1 image per worker

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

class JTFMEvaluator(DatasetEvaluator):
    """
    A dead-simple, Just The Facts, Ma'am evaluator.  Just compiles and returns results.
    """

    def __init__(self,  distributed, output_dir=None):
        """
        Arguments:
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process. Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset [not implemented at present]:
        """
        self._tasks = ("bbox",)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        print("Evaluator: JTFMEvaluator")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            #Copy some input values to the output
            prediction = {"image_id":input["image_id"], "trow":input["trow"], "tcol":input["tcol"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions)) #Necessary.  I think it joins a set of lists into one

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}
        return {"Predictions":predictions}
    
    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        pass
    
class Trainer(DefaultTrainer):
    """Customized version of DefaultTrainer, which enables augmentation
        to be added via a custom DatasetMapper
    """
    #For testing, we don't use augmentation (but see detectron2/tools/train_net.py to add test-time augmentation)
#     @classmethod
#     def build_inference_loader(cls, cfg, dataset_name):
#         return build_test_loader(cfg, dataset_name, mapper=TileDatasetMapper(cfg,False))  
   
    #For training we add image augmentation (=corruption of several kinds)
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=TridentDatasetMapper(cfg, True))
    
    #Add this method to use a custom optimizer (but how does it interact with momentum, weight decay?)
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.  This is built on`detectron2.solver.build_optimizer`,
        but it returns an AdaBound optimizer instead of SGD.
        """
        norm_module_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        #optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1) #orig
        #print('LR before optimizer: ',cfg.SOLVER.BASE_LR)
        optimizer = AdaBound(params, lr=cfg.SOLVER.BASE_LR, final_lr=0.05)
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        #print('LR after optimizer: ',cfg.SOLVER.BASE_LR)
        return optimizer
    
def main(args):
    #torch.cuda.empty_cache()

    #You need this line to get the 'run' object so you can log to it
    run = Run.get_context()
    run.log('Data folder', args.data_folder,)
    run.log('Output folder', args.output_dir,'')
    #run.log('test', 100, 'test logging a single value') #Log a value
    #run.log_list('test_list', [1,2,3], description='test logging a list') #Log a list
    
    #Set paths
    rootdir = Path(str(args.data_folder))
    imagepath = Path(rootdir/'tiled_images')
    annotationpath = Path(rootdir/'tiled_annotations')
    inferencepath = Path(rootdir/'images_to_process')
    output_dir = args.output_dir
    run.log("rootdir",rootdir,)
    run.log("imagepath",imagepath,)
    run.log("annotationpath",annotationpath,)
    run.log("inferencepath",inferencepath,)
    run.log("output_dir",output_dir)
    
    #Get classes
    annotation_files = get_annotation_files(annotationpath)
    classcounts = count_classes(annotation_files) #Get a list of tuples (class, count)
    orig_class_names = [tup[0] for tup in classcounts] #Get a list of classes
    run.log_list('Original class names',orig_class_names,'Class names found in annotation files.')
    
    #Split data into train and validation sets (not used here currently because of difficulty writing the file)
#     train_idxs,val_idxs = split_by_random(annotation_files,0.2)
#     run.log('Length of training dataset',len(train_idxs),)
#     run.log('Length of validation dataset',len(val_idxs),)

#     #Write the text files to the root data directory (FAILED -- it's s read-only filesystem)
#     write_train_validate_files(rootdir,annotation_files,train_idxs,val_idxs)
    
    #Set list of permitted class names.  
    #Some of the rarer names are lumped into "other_animal" & the list is alphabetical
    CLASS_NAMES = [
     'boma',
     'buffalo',
     'building',
     'charcoal mound',
     'charcoal sack',
     'cow',
     'donkey',
     'eland',
     'elephant',
     'gazelle',
     'giraffe',
     'hartebeest',
     'human',
     'impala',
     'kudu',
     'oryx',
     'other_animal',
     'shoats',
     'warthog',
     'wildebeest',
     'zebra']

    #Load datasets and register them with Detectron
    register_datasets(rootdir, CLASS_NAMES)
    sval = str(MetadataCatalog.get("survey_valid"))
    run.log('survey_valid',sval,)
 
    #Create the config (cfg) object.  This is my function (above)
    cfg = setup_model_configuration(rootdir, output_dir, CLASS_NAMES)
    
    # default_setup performs some basic common setups at the beginning of a job, including:
    #      1. Set up the detectron2 logger
    #      2. Log basic information about environment, cmdline arguments, and config
    #      3. Backup the config to the output directory
    default_setup(cfg, args) ##in detectron2/engine/defaults.py.  
    
    #If you are doing inference (=evaluation), build the model first
    if args.eval_only:
        print("Building model")
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #Run the evaluation (build a test loader, an evaluator, and run)
        print("Building data loader and evaluator")
        image_files = list(inferencepath.glob('*.jpg'))
        #Build evaluator (doesn't need to be in the for loop).  distributed=True if total number of GPUs is > 1.
        evaluators = DatasetEvaluators([JTFMEvaluator(distributed=False, output_dir=cfg.OUTPUT_DIR)])
        results = []
        for f in image_files:
            file_results = []
            pil_img = Image.open(f)
            pil_img = np.asarray(pil_img)
            pil_img = copy.deepcopy(pil_img) #necessary step
            #Convert to tensor
            img = torch.from_numpy(pil_img)
            img = img.permute(2,0,1)
            fullimage_size = img.shape[1:]

            #Get tiles
            tile_size = (800, 800) #H, W in pixels
            tile_overlap = 100
            tiles = get_tiles(img,tile_size,tile_overlap)

            #Now put in a list and make sure the tiles are contiguous (they weren't)
            nrows, ncols = tiles.shape[1],tiles.shape[2]
            tilelist = []
            tilelist = [tiles[0][ir][ic].contiguous() for ir in range(nrows) for ic in range(ncols) ] 

            #Create a set of tile positions (note: order must match that of tilelist!)
            tile_positions = [{"trow":ir,"tcol":ic} for ir in range(nrows) for ic in range(ncols)]

            #Create a list of dicts (one per image) for Detectron2 input
            datasetdicts = [{"image_id":f, "trow":tile_pos["trow"],"tcol":tile_pos["tcol"],"image":tile,\
                             "width":tile_size[1],"height":tile_size[0]} for tile_pos,tile \
                            in zip(tile_positions,tilelist)]

            #Create testloader 
            testloader = build_inference_loader(cfg, "test", datasetdicts,TileDatasetMapper)

            print(f'Running {f}: {len(datasetdicts)} tiles./n')
            print('Note: Detectron count will be misprinted if batch size > 1 (multiply \
            by batch size to get actual count.)')
            tic = time()
            tile_results = inference_on_dataset(model, testloader, evaluators)
            print("Inference on full-size image done in {:.2f}s".format(time() - tic))
            
            #Recombine tile annotations into a full-image annotation 
            print("Reassembling tile annotations")
            all_instances = combine_tile_annotations(tile_results,tile_size,tile_overlap,fullimage_size)
            
            #Create the dict for this file
            file_results = {'file_name':f}
            file_results['height'] = fullimage_size[0]
            file_results['width'] = fullimage_size[1]
            file_results['instances'] = all_instances['instances']
            results.append(file_results)
            
        #Write the results dict to disk: one xml annotation file per image plus a summary (class_counts.csv)
        write_results_to_xml(results,output_dir,CLASS_NAMES)
        return results

    #Otherwise, train (build the Trainer and return it)
    print("Creating or loading a trainer")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print("Training")
    return trainer.train()

#     #Run the evaluation (build a test loader, an evaluator, and run)
#     print("Building data loader and evaluator")
#     val_loader = build_detection_test_loader(cfg, "survey_valid")
#     evaluators = DatasetEvaluators([COCOEvaluator("survey_valid", cfg, distributed=True, output_dir=cfg.OUTPUT_DIR)])
#     #Note: 'results' holds the summary table values.  Predictions are written to "instances_predictions.pth"
#     print("Doing inference")
#     results = inference_on_dataset(model, val_loader, evaluators)
#     return results


if __name__ == '__main__':
    print("Testing if I can parse arguments")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type=str,dest='data_folder', help='Root folder for input data')
    
    parser.add_argument('--eval_only', dest='eval_only', action='store_true')
    parser.set_defaults(eval_only=False)

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    
    #Note: Yuxin Wu says --resume must come  before --output_dir (https://github.com/facebookresearch/detectron2/issues/148)
    parser.add_argument('--output_dir',type=str,dest='output_dir',help='Root folder for output')
    
    parser.add_argument('--num_gpus_per_machine', type=int,dest='num_gpus_per_machine')
    parser.set_defaults(num_gpus_per_machine=1)

    parser.add_argument('--num_machines',type=int, dest='num_machines')
    parser.set_defaults(num_machines=1)

    parser.add_argument('--machine_rank', type=int,dest='machine_rank')
    parser.set_defaults(machine_rank=0)

    parser.add_argument('--dist_url', type=str, dest='dist_url')
    parser.set_defaults(dist_url="auto")
    
    #Call the parser
    args = parser.parse_args()
    print("Command Line Args:", args)
    #main(args)
    launch(
        main,
        args.num_gpus_per_machine,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )











