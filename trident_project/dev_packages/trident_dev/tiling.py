# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_tiling.ipynb (unless otherwise specified).

__all__ = ['calc_padding_for_fpad', 'get_tiles', 'tile_images', 'get_tile_offsets', 'combine_tile_annotations']

# Cell
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

#These are for PascalVOCWriter and PascalVOCReader
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

from collections import Counter
from collections import OrderedDict #tile_results is an OrderedDict

# Cell
def calc_padding_for_fpad(img_size:tuple,tile_size:tuple,tile_overlap:int)->tuple:
    """
    Calculates the padding needed to make an image an even multiple of tile size,
    given overlap between tiles.  Returns padding: tuple (left, right, top, bottom)
    in pixels for the sides of an image. It is assumed that padding is added to right
    side and bottom only.
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

# Cell
def get_tiles(full_image, tile_size, tile_overlap,print_padding=False):
    """
    Cuts a full-size image into tiles, potentially overlapping.  It first
    pads the image with black on the right and bottom edges as necessary to
    get a set of equal-sized tiles.  Returns a Pytorch array of tiles of
    `torch.Size([1, rows, cols, channels, tile_h, tile_w])`.

    **Arguments**:
    - `full_image`: (Numpy array [H, W, C] where C is in RGB order).
    - `tile_size`: (int, int) (H, W) in pixels
    - `tile_overlap`: int in pixels
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

# Cell
def tile_images(image_files,tile_size,tile_overlap, tiledir):
    """Breaks images into tiles and saves them to disk."""
    n_images = len(image_files)
    n_saved = 0
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
        ddict = [{"image_id":str(f), "trow":tile_pos["trow"], "tcol":tile_pos["tcol"],\
                         "image":tile, "width":tile_size[1],"height":tile_size[0]} \
                        for tile_pos,tile in zip(tile_positions,tilelist)]

        for i in range(len(ddict)):
            row = ddict[i]['trow']
            col = ddict[i]['tcol']
            fname = Path(ddict[i]['image_id']).stem
            file_type = 'jpg'
            tile_id_str = '{}[{}][{}]'.format(fname,row,col) #To match with images, extension is omitted
            new_im = transforms.ToPILImage()(ddict[i]['image']).convert("RGB")
            new_im.save('{}/{}.{}'.format(tiledir, tile_id_str, file_type))
            n_saved += 1
            if n_saved % 50000 == 0:
                print("working:",n_saved,"tiles created...")
    print("Saved",n_saved,"tiles to ",tiledir,"from ",n_images,"full-sized images.")


# Cell
def get_tile_offsets(trow,tcol,tile_size,overlap):
    """Calulates X and Y offsets in pixels for a particular tile.  Used for reassembling tile output.

        **Arguments**:
        - `trow`, `tcol`: int, int.  Row and column number for the tile
        - `tile_size`: tuple. (height, width) in pixels
        - `overlap`: int.  Tile overlap in pixels
    """
    tile_h, tile_w = tile_size
    row_offset = trow * (tile_h - overlap)
    col_offset = tcol * (tile_w - overlap)
    return (row_offset,col_offset)

# Cell
def combine_tile_annotations(tile_results,tile_size,overlap,fullimage_size):
    """Reassemble Detectron2-style annotations for individual tiles into an annotation for the original file.
       Duplication of boxes due to tile overlap is ignored (all boxes are returned, regardless of overlap).

       **Arguments**:
       - `tile_results`: OrderedDict.  Results returned by the model (an OrderedDict with one element called 'Predictions')
       - `tile_size`: int.  tile size in pixels
       - `overlap`: int. tile overlap in pixels
       - `fullimage_size`: tuple.  (height, width) of original image

       Returns: a dict with the combined Instances objects (boxes, scores, predicted classes)
    """
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