
import sys
sys.path.insert(1, '/') #The container root

print('Loading imports')
import argparse
import os
import time
import cv2
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np 
import copy 
from PIL import Image
import random
import logging 
from pathlib import Path

import pandas as pd

import datetime
import logging
from collections import OrderedDict,Counter
from contextlib import contextmanager

#Detectron2 imports
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import inference_context

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import build_detection_test_loader, trivial_batch_collator
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup, launch 
from detectron2.evaluation import inference_on_dataset,DatasetEvaluators 
from detectron2.evaluation import DatasetEvaluator,COCOEvaluator
from detectron2.structures import Instances, Boxes
import detectron2.utils.comm as comm
from detectron2.utils.comm import get_world_size, get_rank, get_local_rank, is_main_process

#My code
from dev_packages.TridentNet.tridentnet import add_tridentnet_config
from trident_dev.detectron2_utilities import *
from trident_dev.tiling import *
from trident_dev.model import *
from trident_dev.utilities import *

#Azureml SDK
from azureml.core.run import Run
print("All imports passed")
print("Rank:",comm.get_rank())
logger = logging.getLogger("detectron2")
    
#WARNING: if you mis-format a value, it will be silently ignored (i.e., use 'key:value' instead of 'key=value')
def setup_model_configuration(rootdir, output_dir, CLASS_NAMES):
    cfg = get_cfg()
    add_tridentnet_config(cfg) #Loads a few values that distinguish TridentNet from Detectron2
    
    #MODEL
    cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth") #In blobstore
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.BACKBONE.FREEZE_AT = 0 #There are 4 conv stages.  0 means unfreeze.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default is 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.DEVICE='cuda' #THIS IS FOR INFERENCE! Options: 'cuda' or 'cpu'
    
    # Specify the test branch index TridentNet Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    cfg.MODEL.TRIDENT.TEST_BRANCH_IDX = 1

    #SOLVER
    #KEEP. Default Solver specs. (Note; 3x seems to require 3 times as many steps as 1x)
    #Base-TridentNet-Fast-C4.yaml 60,000-80,000 steps and 90,000 max_iter.
    #tridentnet_fast_R_50_C4_1x.yaml SAME
    #tridentnet_fast_R_50_C4_3x.yaml 210,000 - 250,000 steps and 270,000 max_iter.
    #tridentnet_fast_R_101_C4_3x.yaml SAME      
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR" #Options: 'WarmupCosineLR'
    #You idiot!  The steps are the places where the default "WarmupMultiStepLR" scheduler drops the learning rate by gamma=0.1.
    #V3 TRAINING: A 4-GPU machine takes 1.41 s/img = ~25 mins/1000 tiles or 4 hours/10,000 tiles
    #V3 INFERENCE a 4-GPU machine takes 0.05s/img x 54 tiles/img = ~47 minutes/1000 images.
    cfg.SOLVER.MAX_ITER = 12000 #  270000 for trident by default
    cfg.SOLVER.STEPS = (2000,3000,4000,5000,10000)#If a single value, then trailing comma is required (must be iterable) #(210000, 250000) for trident
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.WARMUP_ITERS = 1000 #1000 is default
    cfg.SOLVER.IMS_PER_BATCH = 3 #16 if using 4 GPUs
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000 #How often the model is written
    
    #Learning Rates
    #From Goyal et al. 2017: Linear Scaling Rule: When the minibatch size is multiplied by k, 
    #multiply the learning rate by k.  i.e., as you increase the batch size because of using 
    #additional GPUs, increase the learning rate too.  Works up to very large batches (8,000 images)
    #See auto_scale_workers() in Detectron2 (very important!)
    cfg.SOLVER.BASE_LR = 0.002  #LR is .001 in defaults.py but .02 in tridentnet, but they trained on 8 GPUs
    
    cfg.SOLVER.OPTIMIZER_TYPE = "Default" # 'Default' or 'Adabound'
    
    #INPUT
    #Pixel means are from 19261 500x500 tiles on Aug 15 2020
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [143.078, 140.690, 120.606] 
    cfg.MODEL.PIXEL_STD = [34.139, 29.849, 31.695]
        
    #Auugmentation. Add corruption to images with probability p
    cfg.INPUT.corruption = 0.1
    
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800) 
    cfg.INPUT.MIN_SIZE_TEST = 800
    
    #DATASETS
    cfg.DATASETS.TRAIN = ("survey_train",) #Why the comma?  Bcs you can have multiple training datasets
    cfg.DATASETS.TEST = ("survey_valid",)

    #DATALOADER
    cfg.DATALOADER.SAMPLER_TRAIN = "JP_RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.02 #Goal is 400 images per category given 20,000 images total.
    cfg.DATALOADER.NUM_WORKERS = 4 #Set to equal the number of GPUs.

    # if True, the dataloader will filter out images that have no associated
    # annotations at train time.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = output_dir
      
    #WARNING. #freeze() makes the config immutable; hence it must come last
    cfg.freeze() 

    return cfg


def jp_inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            print("rank",comm.get_rank(),"is processing batch",idx)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            
            outputs = model(inputs) #RUN THE MODEL!!!!!!!!!

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

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
    inferencepath = Path(rootdir/'inferencetest')
    output_dir = args.output_dir
    run.log("rootdir",rootdir,)
    run.log("imagepath",imagepath,)
    run.log("annotationpath",annotationpath,)
    run.log("inferencepath",inferencepath,)
    run.log("output_dir",output_dir)
    
    #Define allowed file types
    imagefile_extensions = ['.jpg','.JPG']
    annotationfile_extensions = ['.xml']
    
    #Get annotation files
    annotation_files = get_files(annotationpath,annotationfile_extensions,recursive=False)
        
    #Log a count of the original classes    
    classcounts = count_classes(annotation_files) #Returns a list of tuples (class, count)
    orig_class_names = [tup[0] for tup in classcounts] #Get a list of classes
    run.log_list('Original class names',orig_class_names,'Class names found in annotation files.')

    #Set list of permitted class names.  
    CLASS_NAMES = [
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

    omit_classes = ['boma'] #Classes to be dropped
    rename_classes = {} #The format of this dict is oldname:newname if required.
    
    #Load datasets and register them with Detectron
    trainfile = rootdir/'train.txt'
    validfile = rootdir/'valid.txt'
    register_dataset_from_file(imagepath,annotationpath,trainfile,True, 'survey_train', CLASS_NAMES, rename_classes, omit_classes)
    register_dataset_from_file(imagepath,annotationpath,validfile,False, 'survey_valid', CLASS_NAMES,rename_classes,omit_classes)
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
        #NOTE: we are NOT creating a new Trainer here; we are only using its class method 'build_model'
        #Therefore it doesn't do some of the other tricks like wrapping the model and calling build_train_loader
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #Wrap the model in DistributedDataParallel module
        distributed = comm.get_world_size() > 1
        if distributed:
            world_size = comm.get_world_size()
            rank = comm.get_local_rank()
            model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
        print("This is rank",rank)
   
        #Run the evaluation (build one evaluator; then for each image file, rebuild the test loader with new tiles.)
        print("Building evaluator")
        image_files = get_files(inferencepath,imagefile_extensions,recursive=False)
            
        #Build evaluator (doesn't need to be in the for loop).  Note: distributed MUST BE True if total number of GPUs is > 1.
        evaluators = DatasetEvaluators([JTFMEvaluator(distributed=True, output_dir=cfg.OUTPUT_DIR)])
        mapper = TileDatasetMapper(cfg, False)
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

            #Create a list of dicts (one per image) for Detectron2 input (note: list comprehension zips names and positions)
            datasetdicts = [{"image_id":str(f), "trow":tile_pos["trow"], "tcol":tile_pos["tcol"],\
                             "image":tile, "width":tile_size[1],"height":tile_size[0]} \
                            for tile_pos,tile in zip(tile_positions,tilelist)]

            inference_dataset = DatasetFromList(datasetdicts)
            
            inferencesampler = torch.utils.data.distributed.DistributedSampler(
                inference_dataset,
                num_replicas=cfg.DATALOADER.NUM_WORKERS,
                rank=rank
                )

            testloader = torch.utils.data.DataLoader(
                dataset=inference_dataset,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                shuffle=False,            
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                pin_memory=True,
                collate_fn=trivial_batch_collator,
                sampler=inferencesampler)     

            print(f'Running {f}: {len(datasetdicts)} tiles./n')
            if comm.is_main_process():
                tic = time.perf_counter()
            tile_results = []
            tile_results = jp_inference_on_dataset(model, testloader, evaluators)
            
            if comm.is_main_process():
                print("Inference on full-size image done in {:.2f}s".format(time.perf_counter() - tic))

                #Recombine tile annotations into a full-image annotation 
                print("Reassembling tile annotations")
                if 'Predictions' in tile_results:
                    all_instances = combine_tile_annotations(tile_results,tile_size,tile_overlap,fullimage_size)
                    print("all_instances after combine_tile_annotations",all_instances)

                    #Create the dict for this file
                    file_results = {'file_name':f}
                    file_results['height'] = fullimage_size[0]
                    file_results['width'] = fullimage_size[1]
                    file_results['instances'] = all_instances['instances']
                    results.append(file_results)
                else:
                    next
        if comm.is_main_process():
            #Write the results dict to disk: one xml annotation file per image plus a summary (class_counts.csv)
            write_results_to_xml(results,output_dir,CLASS_NAMES)
            if args.write_annotated_images==True:
                print("n_image_files",len(image_files))
                print("n_results",len(results))
                print(CLASS_NAMES)
                save_annotated_images(image_files,results,CLASS_NAMES,output_dir,'ann')
            return results #if doing evaluation
        print("rank",comm.get_local_rank(),"returning")
        return 

    #Else train (build the Trainer and return it)
    else:
        print("Creating or loading a trainer")
        #Note: __init__ calls self.build_train_loader(cfg) and wraps the model in DistributedDataParallel
        trainer = Trainer(cfg) 
        trainer.resume_or_load(resume=args.resume)
        print("Training")
        if args.eval_after_training==True:
            trainer.train()
            #Run the evaluation (build a test loader, an evaluator, and run)
            print("Building data loader and evaluator")
            val_loader = build_detection_test_loader(cfg, "survey_valid")
            evaluators = DatasetEvaluators([COCOEvaluator("survey_valid", cfg, distributed=True, output_dir=cfg.OUTPUT_DIR)])
            #Note: 'results' holds the summary table values.  Predictions are written to "instances_predictions.pth"
            print("Doing inference")
            results = inference_on_dataset(trainer.model, val_loader, evaluators)
            return results
        else:
            return trainer.train()

if __name__ == '__main__':

#    __spec__ = None #fixes an error that pops up when debugging with pdb from commandline
    
    print("Testing if I can parse arguments")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type=str,dest='data_folder', help='Root folder for input data')
    
    parser.add_argument('--eval_only', dest='eval_only', action='store_true')
    parser.set_defaults(eval_only=False)

    parser.add_argument('--eval_after_training', dest='eval_after_training', action='store_true')
    parser.set_defaults(eval_only=False)

    parser.add_argument('--write_annotated_images', dest='write_annotated_images', action='store_true')
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
