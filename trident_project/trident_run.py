import sys
sys.path.insert(1, './dev_packages/TridentNet')
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
#from google.colab.patches import cv2_imshow
from pathlib import Path
from fvcore.common.file_io import PathManager
from torch import randperm,load,nn
from IPython import display
import PIL
from sklearn import metrics
from pathlib import Path
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
from adabound import AdaBound

# Import Detectron2 structures and functions
from detectron2.config import get_cfg
from detectron2.structures import Boxes, BoxMode, pairwise_iou 
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper,build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data.transforms import TransformGen
from fvcore.transforms.transform import (BlendTransform,NoOpTransform,Transform)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators,COCOEvaluator, inference_on_dataset
from detectron2.modeling.matcher import Matcher
from detectron2.solver.build import *

print("Trying tridentnet import")
from tridentnet import add_tridentnet_config
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
    #cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth") #This is in blobstore/temp directory
    cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth")
    
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default is 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # 
    
    #KEEP. Default Solver specs; (Note 3x seems to require 3 times as many steps as 1x)
    #Base-TridentNet-Fast-C4.yaml: 60,000-80,000 steps and 90,000 max_iter.
    #tridentnet_fast_R_50_C4_1x.yaml: SAME
    #tridentnet_fast_R_50_C4_3x.yaml: 210,000 - 250,000 steps and 270,000 max_iter.
    #tridentnet_fast_R_101_C4_3x.yaml: SAME      
    
    #cfg.SOLVER.STEPS = (60000, 80000) #for base model
    #cfg.SOLVER.MAX_ITER = 90000  
    cfg.LR_SCHEDULER='WarmupCosineLR'
    cfg.SOLVER.STEPS = (7000, 8000) #(210000, 250000) for trident, also 16K,18K for 20K
    cfg.SOLVER.MAX_ITER = 10000 #  270000 for trident
    cfg.SOLVER.WARMUP_ITERS = 1000 #1000 is default
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.01 #Is .001 in defaults.py.  It overflowed when I tried 0.02
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    
    #Pixel means are from 19261 500x500 tiles on Aug 15 2020 (train_dict)
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [143.078, 140.690, 120.606] #first 5K batch was [137.473, 139.769, 106.912] #default was [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [34.139, 29.849, 31.695]#first 5k batch was [26.428, 22.083, 26.153]
        
    #Auugmentation. Add corruption to images with probability p
    cfg.INPUT.corruption = 0.1
    
    cfg.INPUT.MIN_SIZE_TRAIN = (400,420,440,460,480,500) #See next cell for how this was calculated
    cfg.INPUT.MIN_SIZE_TEST = 500
    
    cfg.DATASETS.TRAIN = ("survey_train",) #Why the comma???  Bcs you can have multiple training datasets
    cfg.DATASETS.TEST = ("survey_valid",)
    cfg.DATALOADER.NUM_WORKERS = 24 #Set to equal the number of CPUs.

    # if True, the dataloader will filter out images that have no associated
    # annotations at train time.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = output_dir
      
    #WARNING; #freeze() makes the config immutable; hence it must come last
    cfg.freeze() 
    return cfg

#Overriding the method (which is in detectron2/data/detection_utils.py)
def build_augmentation(cfg, is_train):
    #logger = logging.getLogger(__name__)

    result = utils.build_augmentation(cfg, is_train) #probably contains default behavior

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

class Trainer(DefaultTrainer):
    """Customized version of DefaultTrainer, which enables augmentation
        to be added via a custom DatasetMapper
    """
    #For testing, we don't use augmentation (but see detectron2/tools/train_net.py to add test-time augmentation)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg,False))

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
        optimizer = AdaBound(params, lr=cfg.SOLVER.BASE_LR, final_lr=0.1)
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
def main(args):
    
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
    output_dir = args.output_dir
    run.log("rootdir",rootdir,)
    run.log("imagepath",imagepath,)
    run.log("annotationpath",annotationpath,)
    run.log("output_dir",output_dir)
    
    #Get classes
    annotation_files = get_annotation_files(annotationpath)
    class_counts = count_classes(annotation_files) #Get a list of tuples (class, count)
    orig_class_names = [tup[0] for tup in class_counts] #Get a list of classes
    run.log_list('Original class names',orig_class_names,'Class names found in annotation files.')
    
    #Split data into train and validation sets (not used here currently because of difficulty writing the file)
#     train_idxs,val_idxs = split_by_random(annotation_files,0.2)
#     run.log('Length of training dataset',len(train_idxs),)
#     run.log('Length of validation dataset',len(val_idxs),)

#     #Write the text files to the root data directory (FAILED -- it's s read-only filesystem)
#     write_train_validate_files(rootdir,annotation_files,train_idxs,val_idxs)
    
    #Set list of permitted class names.  
    #Some of the rarer names are lumped into "other_animal" & the list is alphabetical
    CLASS_NAMES = ['boma',
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
 
    #Configure the model
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
        val_loader = build_detection_test_loader(cfg, "survey_valid")
        evaluators = DatasetEvaluators([COCOEvaluator("survey_valid", cfg, distributed=True, output_dir=cfg.OUTPUT_DIR)])
        #Note: 'results' holds the summary table values.  Predictions are written to "instances_predictions.pth"
        print("Doing inference")
        results = inference_on_dataset(model, val_loader, evaluators)
        return results

    #Otherwise, train (build the Trainer and return it)
    print("Creating or loading a trainer")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print("Training")
    return trainer.train()

if __name__ == '__main__':
    print("Testing if I can parse arguments")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type=str,dest='data_folder', help='Root folder for input data')
    
    parser.add_argument('--output_dir',type=str,dest='output_dir',help='Root folder for output')

    parser.add_argument('--eval_only', dest='eval_only', action='store_true')
    parser.set_defaults(eval_only=False)

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    
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











