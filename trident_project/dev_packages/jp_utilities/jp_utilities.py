import cv2
import PIL
from PIL import Image
import numpy as np
import detectron2
from detectron2.evaluation import inference_context
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes
from detectron2.data import DatasetCatalog, MetadataCatalog

import contextlib
import io
import sys
import torch #,torchvision

#Suppress output from a function.  Usage:
#  with nostdout():
#    func()
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))
    
def evaluate_dataloader_item(dataloader,model,indx):
    """Test-evaluate one particular item from a dataloader.  The item is expected to be a Pytorch tensor.
    """
    etl = enumerate(dataloader)
    with inference_context(model), torch.no_grad():
        found = False
        for idx, inputs in etl:
            if idx==indx:
                outputs = model(inputs)
                return outputs
        if found == False:
            return "Index not found"
        
def get_one_batch(dataloader):
    iter_dl = iter(dataloader) #Necessary.  You have to tell the fucking thing it's iterable.  Why?
    batch = next(iter_dl) 
    return(batch)

def get_dataloader_item(dataloader,indx):
    """Examine a particular input from a dataloader
    """
    etl = enumerate(dataloader)
    found = False
    for idx, inputs in etl:
        if idx==indx:
            return inputs
    if found == False:
        return "Index not found"
    
def evaluate_dicts(model,items):
    """Evaluate a list of 1 or more dicts.  Expects one dict per image, in Detectron format ({'image':tensor, 'height':int, etc.})
    """
    with inference_context(model), torch.no_grad():
        inputs = items
        if isinstance(items,dict):
            items = [items] #make a single dict into a list
        outputs = model(items)
        return outputs

def display_tensor(pttensor):
    """Convert a Pytorch tensor to a Numpy array and display with cv2.
    """
    nparray = pttensor.permute(1,2,0).numpy()
    cv2_nparray = nparray[:,:,::-1] #switch to BGR for cv2
    cv2_imshow(cv2_nparray)
    
def show_annotated_image(image, annotations):
    """Show an annotated image.
       Parameters:
       image: Numpy array.
       annotations: output of a Detectron2 model.
    """
    test_metadata = MetadataCatalog.get("test")
    visualizer = Visualizer(image, metadata=test_metadata, scale=0.5)
    vis = visualizer.draw_instance_predictions(annotations["instances"])
    cv2_imshow(vis.get_image()[:, :, ::-1])
    
def dummy_detectron2_result():
    dummy_result = Instances((500,500))
    dummyboxes = Boxes([[1,2,3,4],[5,6,7,8]])
    dummy_result.pred_boxes = dummyboxes
    dummy_result.pred_classes = [0,13]
    dummy_result.scores = [0.65,1.877]
    return dummy_result