import sys
sys.path.insert(1, '/dev_packages')

import torch,torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, io, json, cv2, random
from IPython import display
import PIL

# import some detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from TridentNet.tridentnet import add_tridentnet_config
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def setup_model_configuration(model_path, output_dir, CLASS_NAMES):
    cfg = get_cfg()
    add_tridentnet_config(cfg) #Loads a few values that distinguish TridentNet from Detectron2
    
    #OPTIONAL [WARNING!]  Specify model weights
    #cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl" #original weights
    cfg.MODEL.WEIGHTS = model_path #This is where model weights are found
    
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    #You idiot!  The steps are the places where the default "WarmupMultiStepLR" scheduler drops the learning rate by gamma=0.1.
    cfg.SOLVER.STEPS = (20000, 22000) #(210000, 250000) for trident, also 16K,18K for 20K
    cfg.SOLVER.MAX_ITER = 25000 #  270000 for trident
    cfg.SOLVER.WARMUP_ITERS = 1000 #1000 is default
    cfg.SOLVER.IMS_PER_BATCH = 16
        
    #Learning Rates
    #From Goyal et al. 2017: Linear Scaling Rule: When the minibatch size is multiplied by k, 
    #multiply the learning rate by k.  i.e., as you increase the batch size because of using 
    #additional GPUs, increase the learning rate too.  Works up to very large batches (8,000 images)
    #See auto_scale_workers() in Detectron2 (very important!)
    cfg.SOLVER.BASE_LR = 0.0005 #Is .001 in defaults.py but .02 in tridentnet, but they trained on 8 GPUs
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
    
    #FOR DOING INFERENCE ON CPU-ONLY
    cfg.MODEL.DEVICE='cpu'
      
    #WARNING. #freeze() makes the config immutable; hence it must come last
    cfg.freeze() 
    return cfg

def init():
    global model
    global test_metadata
    print(os.environ)
#    os.environ['timeout'] = 120000 #run_invocation_timeout

    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_final.pth')

    # If your model were stored in the same directory as your score.py, you could also use the following:
    # model_path = os.path.abspath(os.path.join(os.path.dirname(__file_), 'sklearn_mnist_model.pkl')
    
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
    
#     sample_dict = [
#         {'file_name': '/cdata/tanzania/temp/tiled_images/SL25-2013-10-12_AM_TNP_R_0054[3][2].jpg',
#           'image_id': 'SL25-2013-10-12_AM_TNP_R_0054[3][2]',
#           'height': 500,
#           'width': 500,
#           'annotations': [{'category_id': 8,
#             'bbox': [458.0, 112.0, 496.0, 139.0]},
#            {'category_id': 8,
#             'bbox': [423.0, 110.0, 452.0, 131.0]}]},
#         {'file_name': '/cdata/tanzania/temp/tiled_images/SL25-2013-10-05_PM_5HCFA_R_0235[3][13].jpg',
#           'image_id': 'SL25-2013-10-05_PM_5HCFA_R_0235[3][13]',
#           'height': 500,
#           'width': 500,
#           'annotations': [{'category_id': 0,
#           'bbox': [262.0, 169.0, 460.0, 342.0]}]}]

    #We define a 'test' MetadataCatalog object because the Visualizer uses it to find class names
#     DatasetCatalog.register("test", sample_dict)
    MetadataCatalog.get("test").set(thing_classes=CLASS_NAMES)
    test_metadata = MetadataCatalog.get("test")
    
    #global predictor 
    cfg = setup_model_configuration(model_path, "dummy_output_dir", CLASS_NAMES)
    model = DefaultPredictor(cfg)

# @input_schema('data', NumpyParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))

# def run(binary_data):
#     try: 
#         #data = np.array(json.loads(raw_data)['data'],dtype='uint8') #load data from the bytes object into an ndarray
#         tempfile = open("./test_image.jpg", "wb")
#         tempfile.write(binary_data)
#         tempfile.close()
#         data = cv2.imread("./test_image.jpg")
#         result = model(data)

#         #print(result)
#         if (len(result['instances']) > 0):
#             visualizer = Visualizer(data[:, :, ::-1], metadata=test_metadata, scale=1.0)
#             vis = visualizer.draw_instance_predictions(result['instances']) #.to("cpu"))
#             # You can return any data type, as long as it is JSON serializable.
#             # But the prediction itself is not serializable because it contains classes such as Instances and Boxes.
#             annotated_img = vis.get_image().tolist()
#             jd = json.dumps({"annotated_img": annotated_img})
#             return jd #bytes(jd, encoding='utf8')
#         else:
#             return raw_data
#     except Exception as e:
#         error = str(e)
#         return json.dumps({"error": error}) #TODO: Comment out this line for production!

@rawhttp    
def run(request):
    try:
        if request.method == 'POST':
            reqBody = request.get_data(False)
            image = cv2.imread(BytesIO(reqBody))
            result = model(image)
            if (len(result['instances']) > 0):
                visualizer = Visualizer(data[:, :, ::-1], metadata=test_metadata, scale=1.0)
                vis = visualizer.draw_instance_predictions(result['instances']) #.to("cpu"))
                # You can return any data type, as long as it is JSON serializable.
                # But the prediction itself is not serializable because it contains classes such as Instances and Boxes.
                annotated_img = vis.get_image().tolist()
                return {'prediction':result.tolist(),'annotated_img':annotated_img}
            else:
                return {'prediction':result.tolist(),'annotated_img':image.tolist()}
        else:
            return AMLResponse("bad request, use POST", 500)
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error}) #TODO: Comment out this line for production!
