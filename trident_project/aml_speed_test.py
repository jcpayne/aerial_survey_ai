# import some common libraries
import os, argparse
import torch,torchvision
import numpy as np
import logging
from time import time
from pathlib import Path
import PIL
import copy
import torch.nn.functional as F

import detectron2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, configurable
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.engine import DefaultPredictor #,DefaultTrainer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from dev_packages.TridentNet.tridentnet import add_tridentnet_config
setup_logger()

def setup_model_configuration(rootdir, output_dir, CLASS_NAMES):
    cfg = get_cfg()
    add_tridentnet_config(cfg) #Loads a few values that distinguish TridentNet from Detectron2
    
    cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth") 
    
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) #
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (20000, 22000) #(210000, 250000) for trident, also 16K,18K for 20K
    cfg.SOLVER.MAX_ITER = 25000 #  270000 for trident
    cfg.SOLVER.WARMUP_ITERS = 1000 #1000 is default
    cfg.SOLVER.IMS_PER_BATCH = 1
        
    #Learning Rates
    cfg.SOLVER.BASE_LR = 0.0005 
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.PIXEL_MEAN = [143.078, 140.690, 120.606] 
    cfg.MODEL.PIXEL_STD = [34.139, 29.849, 31.695]
        
    cfg.INPUT.MIN_SIZE_TRAIN = (400,420,440,460,480,500) 
    cfg.INPUT.MIN_SIZE_TEST = 500
    
    cfg.DATASETS.TRAIN = ("survey_train",) 
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 6 #Set to equal the number of CPUs.

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = output_dir
    
    #FOR DOING INFERENCE ON GPU/CPU
    cfg.MODEL.DEVICE='cuda' #'cuda' or 'cpu'
    
    #WARNING. freeze() makes the config immutable; hence it must come last
    cfg.freeze() 

    return cfg
    #Set list of permitted class names.  

def timing_run(model,inputs):
    model.eval()
    outputs = []
    starttime = time()
    with torch.no_grad():
        for i in range(len(inputs)):
            tic = time()
            output = model([inputs[i]])
            print((time() - tic))
            outputs.append(output)
    print("Inference on {} images done in {:.2f}s".format(len(inputs), (time() - starttime)))
  #return outputs
    
def main(args):
    rootdir = Path(str(args.data_folder))
    
    #Create dummy class names for 21 classes to match model output dimension
    CLASS_NAMES =   ['class' + str(i+1) for i in range(21)]
    
    # Initialize model with random weights
    cfg = setup_model_configuration(rootdir, None, CLASS_NAMES)
    model = build_model(cfg)   
    
    #Make some fake image dicts
    inputs = []
    for i in range(40):
        image_dict = {}
        image_dict['image_id'] = 'image_number_' + str(i)
        image_dict['image'] = torch.randint(low=0, high=256,size=(3,800,800), dtype=torch.uint8)
        image_dict['height'] = 800
        image_dict['width'] = 800
        inputs.append(image_dict)
        
    #Run the model with random weights
    print("Running with random weights:")
    timing_run(model,inputs)
    
    print("Loading weights")
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) 
    
    print("Running with trained weights")
    timing_run(model,inputs)

    print("torch.cuda.get_device_name(0):",torch.cuda.get_device_name(0))
    myCmd1 = os.popen('nvcc -V').read()
    print("CUDA version: ",myCmd1)
    myCmd2 = os.popen('whereis cudnn.h | xargs cat| grep CUDNN_MAJOR -A 2').read()
    print("CUDNN version: ",myCmd2)

if __name__ == '__main__':
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
    launch(
    main,
    args.num_gpus_per_machine,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args,),
    )