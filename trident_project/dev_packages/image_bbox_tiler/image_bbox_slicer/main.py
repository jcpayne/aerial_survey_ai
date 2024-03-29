"""
Main functionality of ``image_bbox_slicer``.
"""
import os
import csv
import glob
import random
from PIL import Image
#Note: called from the same envt as the notebook is using
from dev_packages.pascal_voc_writer.pascal_voc_writer import Writer
from torchvision.transforms.functional import pad as tvpad #Used for padding
#from pathlib import Path  #I put this in helpers instead
from random import randrange
from .helpers import *


class Slicer(object):
    """
    Slicer class.

    Attributes
    ----------
    IMG_SRC : str
        /path/to/images/source/directory.
        Default value is the current working directory. 
    IMG_DST : str
        /path/to/images/destination/directory.
        Default value is the `/sliced_images` in the current working directory.
    ANN_SRC : str
        /path/to/annotations/source/directory.
        Default value is the current working directory.
    ANN_DST : str
        /path/to/annotations/source/directory.
        Default value is the `/sliced_annotations` in the current working directory.
    keep_partial_labels : bool
        A boolean flag to denote if the slicer should keep partial labels or not.
        Partial labels are the labels that are partly in a tile post slicing.
        Default value is `False`.
    save_before_after_map : bool
        A boolean flag to denote if mapping between 
        original file names and post-slicing file names in a csv or not. 
        Default value is `False`.
    ignore_empty_tiles : bool
        A boolean flag to denote if tiles with no labels post-slicing
        should be ignored or not.
        Default value is `True`.
    empty_sample : float (0 to 1)
        Proportion of the 'empty' tiles (tiles without bounding boxes) to sample.  Default is 0.
    """

    def __init__(self):
        """
        Constructor. 

        Assigns default values to path attributes and other preference attributes. 

        Parameters
        ----------
        None
        """
        self.IMG_SRC = os.getcwd()
        self.IMG_DST = os.path.join(os.getcwd(), 'sliced_images')
        self.ANN_SRC = os.getcwd()
        self.ANN_DST = os.path.join(os.getcwd(), 'sliced_annotations')
        self.keep_partial_labels = False
        self.save_before_after_map = False
        self.ignore_empty_tiles = True
        self.exclude_fragments = True
        self._ignored_files = [] #Files without objects of interest to be ignored (if ignore_empty_tiles=TRUE)
        self._mapper = {} #A dict of file+tile names.
        self._just_image_call = True
        self._tilematrix_dim = None
        self._tile_size = None
        self._tile_overlap = None
        self.badlist = None
        
    def config_dirs(self, img_src, ann_src,
                    img_dst=os.path.join(os.getcwd(), 'sliced_images'),
                    ann_dst=os.path.join(os.getcwd(), 'sliced_annotations'),
                    badlist=[]):
        """Configures paths to source and destination directories after validating them. 

        Parameters
        ----------
        img_src : str
            /path/to/image/source/directory
        ann_src : str
            /path/to/annotation/source/directory
        img_dst : str, optional
            /path/to/image/destination/directory
            Default value is `/sliced_images`.
        ann_dst : str, optional
            /path/to/annotation/destination/directory
            Default value is `/sliced_annotations`.
        badlist : A list of xml files (full paths) to exclude

        Returns
        ----------
        None
        """
        validate_dir(img_src)
        validate_dir(ann_src)
        validate_dir(img_dst, src=False)
        validate_dir(ann_dst, src=False)
        validate_file_names(img_src, ann_src)
        self.IMG_SRC = img_src
        self.IMG_DST = img_dst
        self.ANN_SRC = ann_src
        self.ANN_DST = ann_dst
        self.badlist = badlist
    
    def config_tilesize(self, tile_size,tile_overlap):
        """
        Set tile size in the Slicer object
        tile_size: tuple (width, height) in pixels
        tile_overlap: float. Tile overlap between consecutive strides as proportion of tile size
        """
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
    
    def pad_image(self,img,tile_size,tile_overlap):
        """
        Returns an image that is padded to an even multiple of the tile size, taking overlap 
        into account.  The padding (black by default) is added to the right and bottom edges of 
        the original image.
        Parameters
        ---------------
        img: a PIL image
        tile_size: tuple (width, height) in pixels
        tile_overlap: float. Tile overlap between consecutive strides as proportion of tile size 
        (the same proportion is used for height and width).
        """
        #Extract params
        im = Image.open(img)
        img_size = im.size
        
        #Calculate padding
        padding = calc_padding(img_size,tile_size,tile_overlap)
        
        #Pad image, using pad function from torchvision.transforms.functional
        padded_img = tvpad(im, padding) #By default, fill=0 (black), padding_mode='constant'
        return padded_img

    def __get_tiles(self,img_size, tile_size, tile_overlap):
        """Generates a list coordinates of all the tiles after validating the values. 
        Private Method.
        Parameters
        ----------
        img_size : tuple
            Size of the original image in pixels, as a 2-tuple: (width, height).
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float 
            Tile overlap between two consecutive strides as percentage of tile size.
        Returns
        ----------
        list
            A list of tuples.
            Each holding coordinates of possible tiles 
            in the format - `(xmin, ymin, xmax, ymax)` 
        """
        validate_tile_size(tile_size, img_size)
        tiles = []
        img_w, img_h = img_size
        tile_w, tile_h = tile_size
        #Convert overlap to pixels
        tile_w_overlap = int(tile_w * tile_overlap) 
        tile_h_overlap = int(tile_h * tile_overlap)
        #Calculate stride
        stride_w = tile_w - tile_w_overlap
        stride_h = tile_h - tile_h_overlap
        #Calculate number of rows and cols (and index)
        rows = range(0, img_h-tile_h+1, stride_h)
        nrows = len(rows)
        cols = range(0, img_w-tile_w+1, stride_w)
        ncols = len(cols)
        #Make list of tile coordinates
        for y in rows:
            for x in cols:
                x2 = x + tile_w
                y2 = y + tile_h
                tiles.append((x, y, x2, y2))
        self._tilematrix_dim = (nrows,ncols) 
        #breakpoint()
        return tiles
    
    def slice_by_size(self, tile_size, tile_overlap=0.0,empty_sample=0.0):
        """Slices both images and box annotations in source directories by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size (width, height) of each tile.
        tile_overlap: float, optional  
            Percentage of tile overlap between two consecutive strides.
            Default value is `0`.

        Returns
        ----------
        None
        """
        self._just_image_call = False
        self.slice_bboxes_by_size(tile_size, tile_overlap,empty_sample)
        self.slice_images_by_size(tile_size, tile_overlap)
        #Reset params
        self._ignored_files = []
        self._just_image_call = True

    def slice_by_number(self, number_tiles):
        """Slices both images and box annotations in source directories into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.

        Returns
        ----------
        None
        """
        self._just_image_call = False
        self.slice_bboxes_by_number(number_tiles)
        self.slice_images_by_number(number_tiles)
        self._ignored_files = []
        self._just_image_call = True

    def slice_images_by_size(self, tile_size, tile_overlap=0.0):
        """Slices each image in the source directory by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float, optional  
            Percentage of tile overlap between two consecutive strides.
            Default value is `0`.

        Returns
        ----------
        None
        """
        validate_tile_size(tile_size)
        validate_overlap(tile_overlap)
        if self._just_image_call:
            self.ignore_empty_tiles = []
        mapper = self.__slice_images(tile_size, tile_overlap, number_tiles=-1)
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.IMG_SRC)
        self._mapper = {} #reset mapper

    def slice_images_by_number(self, number_tiles):
        """Slices each image in the source directory into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.

        Returns
        ----------
        None
        """
        validate_number_tiles(number_tiles)
        if self._just_image_call:
            self.ignore_empty_tiles = []
        mapper = self.__slice_images(None, None, number_tiles=number_tiles)
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.IMG_SRC)
        self._mapper = {} #reset mapper
        
    def __slice_images(self, tile_size, tile_overlap, number_tiles):
        """
        Private Method
        If a self._mapper dict has been created by slice_bboxes(), we use it to determine which tiles to save.
        Otherwise we follow our own logic.
        """
        mapper = {} #A dict
        img_no = 1
        self._tile_size = tile_size #set these in self for plotting
        self._tile_overlap = tile_overlap
            
        #for file in sorted(glob.glob(self.IMG_SRC + "/*")):
        image_files = [str(x) for x in sorted(list(Path(self.IMG_SRC).rglob('*')))]
        for file in image_files:
            file_fullname = file
            file_name = file.split('/')[-1].split('.')[0]
            file_type = file.split('/')[-1].split('.')[-1].lower()
            if file_type.lower() not in IMG_FORMAT_LIST:
                continue
            #Pad image (black added to right and bottom) so you don't lose edges
            im = self.pad_image(file,tile_size,tile_overlap) 

            if number_tiles > 0:
                n_cols, n_rows = calc_columns_rows(number_tiles)
                tile_w, tile_h = int(floor(im.size[0] / n_cols)), int(floor(im.size[1] / n_rows))
                tile_size = (tile_w, tile_h)
                tile_overlap = 0.0

            #Get a list of tile coordinates
            tiles = self.__get_tiles(im.size, tile_size, tile_overlap)
            
            #Note: the top-level parent function, slice_by_size(), calls two child functions: 
            # 1) slice_bboxes_by_size() and 2) slice_images_by_size().  
            #In the parent, slice_bboxes_by_size is called *before* slice_images_by_size,
            #therefore, the 'ignore_tiles' list has already been modified in __slice_bboxes() 
            #when it is passed to this function, and mapper.csv has already been written.
            new_ids = []
            for tile in tiles:
                row,col = self.__get_rowcol_indexes(tiles,tile)
                tile_id_str = '{}{}{}'.format(file_name,row,col) #To match with images, extension is omitted
                #tile_id_str = str('{:06d}'.format(img_no))
                
                if self._mapper:
                    if tile_id_str in self._mapper[file_name]:
                        new_im = im.crop(tile) 
                        new_im.save('{}/{}.{}'.format(self.IMG_DST, tile_id_str, file_type))
                        new_ids.append(tile_id_str)
                        img_no += 1
                else:
                    #Skip files if they are in the ignore list
                    if len(self._ignored_files) != 0:
                        if tile_id_str in self._ignored_files:
                            #pop the name once it has been skipped so you don't keep finding it
                            self._ignored_files.remove(tile_id_str) 
                            continue
                    new_im = im.crop(tile) #moved down to avoid wasting the cropping operation on skipped files
                    new_im.save('{}/{}.{}'.format(self.IMG_DST, tile_id_str, file_type))
                    new_ids.append(tile_id_str)
                    img_no += 1
            mapper[file_fullname] = new_ids #Add the tiles to the dict (key=file_name, item = saved tiles (new_ids))    
        print('Obtained {} image slices!'.format(img_no-1))
        return mapper

    def slice_bboxes_by_size(self, tile_size, tile_overlap,empty_sample):
        """Slices each box annotation in the source directory by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float, optional  
            Proportion of tile overlap between two consecutive strides.
            Default value is `0`.
        empty_sample: float [0-1]
            Proportion of tiles that don't include bounding boxes to sample

        Returns
        ----------
        None
        """
        validate_tile_size(tile_size)
        validate_overlap(tile_overlap)
        self._ignored_files = []
        mapper = self.__slice_bboxes(tile_size, tile_overlap,number_tiles=-1,empty_sample=empty_sample)
        self._mapper = mapper
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.ANN_SRC)

    def slice_bboxes_by_number(self, number_tiles,empty_sample):
        """Slices each box annotation in source directories into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.
        
        Returns
        ----------
        None
        """
        validate_number_tiles(number_tiles)
        self._ignored_files = []
        mapper = self.__slice_bboxes(None, None, number_tiles=number_tiles,empty_sample=empty_sample)
        self._mapper = mapper
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.ANN_SRC)

    def __slice_bboxes(self, tile_size, tile_overlap, number_tiles, empty_sample):
        """
        Private Method.  Determines whether tiles contain bounding boxes, then saves tiles as requested.
        Writes a mapper (a dict of filenames/tiles) to self that may later be read by __slice_images()
        """        
        img_no = 1
        mapper = {}
        empty_count = 0

        for xml_file in sorted(glob.glob(self.ANN_SRC + '/*.xml')):
            if(xml_file in self.badlist):
                next
            root, objects = extract_from_xml(xml_file)
            #Get size of original image
            orig_w, orig_h = int(root.find('size')[0].text), int(
                root.find('size')[1].text)
            #Get original image filename
            im_filename = str(Path(root.find('filename').text).stem)
            if root.find('path') is not None:
                im_filepath = str(Path(root.find('path').text).with_suffix(''))
            else:
                im_filepath = str(Path(self.IMG_SRC)/im_filename)
#            
#            im_filename = str(Path(im_filepath).stem)
            #Get size of padded image
            padding = calc_padding((orig_w,orig_h),tile_size,tile_overlap)
            im_size = (orig_w + padding[2],orig_h + padding[3])
            im_w,im_h = im_size
            #If called from __slice_by_number, then number_tiles will be > 0
            if number_tiles > 0:
                n_cols, n_rows = calc_columns_rows(number_tiles)
                tile_w = int(floor(im_w / n_cols))
                tile_h = int(floor(im_h / n_rows))
                tile_size = (tile_w, tile_h)
                tile_overlap = 0.0
            #Else was called by __slice_by_size
            else:
                tile_w, tile_h = tile_size
                tile_w_overlap = int(tile_w * tile_overlap) #convert overlap to pixels
                tile_h_overlap = int(tile_h * tile_overlap)
            tiles = self.__get_tiles(im_size, tile_size, tile_overlap)
            tile_ids = []

            for tile in tiles:
                #Get tile row and column
                row,col = self.__get_rowcol_indexes(tiles,tile)
                #breakpoint()
                tile_name = '{}{}{}'.format(im_filename,row,col) #produces a tile name like imgsourcefile[3][6]
                tilepath = '{}{}{}'.format(im_filepath,row,col) #same but with a full path for the tile
                #initialize a new annotation writer for this tile
                #Initialize the writer with the full path so it correctly fills out the <filename>,<folder>,<path> elements.
                voc_writer = Writer('{}'.format(tilepath), tile_w, tile_h)
                voc_writer.changePath(tilepath) #override the default choice, which is wrong here
                #Loop through all objects (bboxes) in the image to check if each falls in this tile
                empty_count = 0 #The number of bboxes that don't fall in the tile
                for obj in objects:
                    obj_lbl = obj[-4:]
                    points_info = which_points_lie(obj_lbl, tile)

                    if points_info == Points.NONE:
                        empty_count += 1 
                        continue

                    elif points_info == Points.ALL:       # All points lie inside the tile
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    elif not self.keep_partial_labels:    # Ignore partial labels based on configuration
                        empty_count += 1
                        continue

                    elif points_info == Points.P1:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   tile_w, tile_h)

                    elif points_info == Points.P2:
                        new_lbl = (0, obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], tile_h)

                    elif points_info == Points.P3:
                        new_lbl = (obj_lbl[0] - tile[0], 0,
                                   tile_w, obj_lbl[3] - tile[1])

                    elif points_info == Points.P4:
                        new_lbl = (0, 0, obj_lbl[2] - tile[0],
                                   obj_lbl[3] - tile[1])

                    elif points_info == Points.P1_P2:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], tile_h)

                    elif points_info == Points.P1_P3:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   tile_w, obj_lbl[3] - tile[1])

                    elif points_info == Points.P2_P4:
                        new_lbl = (0, obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    elif points_info == Points.P3_P4:
                        new_lbl = (obj_lbl[0] - tile[0], 0,
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    if self.exclude_fragments:   
                        #Dimensions of the new box
                        box_w = abs(new_lbl[2] - new_lbl[0])
                        box_h = abs(new_lbl[3] - new_lbl[1])

                        #This set of conditions excludes new boxes that are smaller than the tile overlap
                        #in the appropriate dimension.
                        #If only one corner of the box is in the tile:
                        if points_info in (Points.P1, Points.P2, Points.P3, Points.P4):
                            if (box_w < tile_w_overlap) or (box_h < tile_h_overlap):
                                empty_count += 1 
                                continue
                        #If two points of the box are in the tile and the box is on a vertical edge (i.e. side) of the tile:
                        elif points_info in (Points.P1_P3,Points.P2_P4):
                            if box_w < tile_w_overlap:
                                empty_count += 1 
                                continue
                        #If two points of the box are in the tile and the box is on the top or bottom of the tile:
                        elif points_info in (Points.P1_P2,Points.P3_P4):
                            if box_h < tile_h_overlap:
                                empty_count += 1 
                                continue                    
                    
                    #addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0)
                    voc_writer.addObject(obj[0], new_lbl[0], new_lbl[1], new_lbl[2], new_lbl[3],
                                         obj[1], obj[2], obj[3])
                #Add tile name to the "ignore" list if none of the bbox objects fall in it
                if self.ignore_empty_tiles and (empty_count == len(objects)):
                    self._ignored_files.append(tile_name)
                    #However we may still sample it
                    rd = random.random()
                    if rd < empty_sample:
                        #Save the tile (it's empty but we sample it)
                        voc_writer.save('{}/{}.xml'.format(self.ANN_DST, tile_name))
                        tile_ids.append(tile_name)
                        img_no += 1
                else:
                    #Save the tile (it contains objects of interest)
                    voc_writer.save('{}/{}.xml'.format(self.ANN_DST, tile_name))
                    tile_ids.append(tile_name)
                    img_no += 1
            mapper[im_filename] = tile_ids #Add new item to mapper dict (key=filename,value=tile_ids)

        print('Obtained {} annotation slices!'.format(img_no-1))
        return mapper

    def resize_by_size(self, new_size, resample=0):
        """Resizes both images and box annotations in source directories to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        self.resize_images_by_size(new_size, resample)
        self.resize_bboxes_by_size(new_size)

    def resize_images_by_size(self, new_size, resample=0):
        """Resizes images in the image source directory to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_new_size(new_size)
        self.__resize_images(new_size, resample, None)

    def resize_by_factor(self, resize_factor, resample=0):
        """Resizes both images and annotations in the source directories by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the images and the annotations should be scaled.
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.resize_images_by_factor(resize_factor, resample)
        self.resize_bboxes_by_factor(resize_factor)

    def resize_images_by_factor(self, resize_factor, resample=0):
        """Resizes images in the image source directory by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the images should be scaled.
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.__resize_images(None, resample, resize_factor)

    def __resize_images(self, new_size, resample, resize_factor):
        """Private Method
        """
        #for file in sorted(glob.glob(self.IMG_SRC + "/*")):
        image_files = [str(x) for x in sorted(list(Path(self.IMG_SRC).rglob('*')))]
        for file in image_files:
            file_name = file.split('/')[-1].split('.')[0]
            file_type = file.split('/')[-1].split('.')[-1].lower()
            if file_type not in IMG_FORMAT_LIST:
                continue
            im = Image.open(file)
            if resize_factor is not None:
                new_size = [0, 0]
                new_size[0] = int(im.size[0] * resize_factor)
                new_size[1] = int(im.size[1] * resize_factor)
                new_size = tuple(new_size)

            new_im = im.resize(size=new_size, resample=resample)
            new_im.save('{}/{}.{}'.format(self.IMG_DST, file_name, file_type))

    def resize_bboxes_by_size(self, new_size):
        """Resizes bounding box annotations in the source directory to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)

        Returns
        ----------
        None
        """
        validate_new_size(new_size)
        self.__resize_bboxes(new_size, None)

    def resize_bboxes_by_factor(self, resize_factor):
        """Resizes bounding box annotations in the source directory by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the bounding box annotations should be scaled.

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.__resize_bboxes(None, resize_factor)

    def __resize_bboxes(self, new_size, resize_factor):
        """Private Method
        """
        for xml_file in sorted(glob.glob(self.ANN_SRC + '/*.xml')):
            root, objects = extract_from_xml(xml_file)
            im_w, im_h = int(root.find('size')[0].text), int(
                root.find('size')[1].text)
            im_filename = root.find('filename').text.split('.')[0]
            an_filename = xml_file.split('/')[-1].split('.')[0]
            if resize_factor is None:
                w_scale, h_scale = new_size[0]/im_w, new_size[1]/im_h
            else:
                w_scale, h_scale = resize_factor, resize_factor
                new_size = [0, 0]
                new_size[0], new_size[1] = int(
                    im_w * w_scale), int(im_h * h_scale)
                new_size = tuple(new_size)

            voc_writer = Writer(
                '{}'.format(im_filename), new_size[0], new_size[1])

            for obj in objects:
                obj_lbl = list(obj[-4:])
                obj_lbl[0] = int(obj_lbl[0] * w_scale)
                obj_lbl[1] = int(obj_lbl[1] * h_scale)
                obj_lbl[2] = int(obj_lbl[2] * w_scale)
                obj_lbl[3] = int(obj_lbl[3] * h_scale)

                voc_writer.addObject(obj[0], obj_lbl[0], obj_lbl[1], obj_lbl[2], obj_lbl[3],
                                     obj[1], obj[2], obj[3])
            voc_writer.save('{}/{}.xml'.format(self.ANN_DST, an_filename))


    #Choose a random line from a file (return the string but strip the line return)
    def random_line(self, afile):
        lines = afile.readlines()
        if len(lines)>1:
            rline = random.choice(lines[1:]) #exclude header
        else:
            raise SystemExit("mapper.csv file has no data (only header)")
        return rline.strip()    
    
    def visualize_sliced_random(self, im_src, an_src, im_dst, an_dst):
        """Picks an image randomly and visualizes unsliced and sliced images using `matplotlib`.

        Parameters:
        ----------
        map_dir : str, optional. /path/to/mapper/directory.
            By default, looks for `mapper.csv` in original annotation folder (an_src). 
        im_src: str. Directory of full-size images
        an_src: str. Directory of full-size annotations
        im_dst: str. Directory of tiled images
        an_dst: str. Directory of tiled annotations.
        Returns:
        ----------
        None
            However, displays the final plots.
        """
        map_path = an_src + '/mapper.csv'
        
        map_path = an_src + '/mapper.csv'        
        with open(map_path) as f:
            line = self.random_line(f)
            mapping = line.split(',')
            maybe_fname = str(Path(mapping[0]).with_suffix('.jpg'))
            src_fullpath = find_filename_case_insensitive(im_src, maybe_fname) #to deal with .jpg vs .JPG
            src_name = mapping[0] #just the filename without extension
            tile_files = mapping[1:]
            print(src_name,tile_files)
        
            if len(tile_files) > 0:
                tsize = self._tile_size
                toverlap = self._tile_overlap
                #Plot the original image, then the tiles
                self.plot_image_boxes(im_src, an_src, src_name)
                self.plot_tile_boxes(im_src, im_dst, an_dst, src_fullpath, tile_files,tsize,toverlap)
            

    def visualize_resized_random(self):
        """Picks an image randomly and visualizes original and resized images using `matplotlib`

        Parameters:
          None 

        Returns: None
        Side effect: Displays randomly-selected original image and resized versions of it.
        """
        #im_file = random.choice(list(glob.glob('{}/*'.format(self.IMG_SRC))))
        image_files = [str(x) for x in list(Path(self.IMG_SRC).rglob('*'))]
        im_file = random.choice(image_files)
        file_name = im_file.split('/')[-1].split('.')[0]

        self.plot_image_boxes(self.IMG_SRC, self.ANN_SRC, file_name)
        self.plot_image_boxes(self.IMG_DST, self.ANN_DST, file_name)

    def __get_rowcol_indexes(self,tiles,tile):
        """
        Private method.
        Finds the row and column index for a given tile by searching for the tile's
        coordinates in a list of all tile coordinates.  

        Parameters
        ----------
        tiles : a list of tuples (xmin,ymin,xmax,ymax) that define a tile
        tile : a tuple for one particular tile
        Returns:
        ----------
        (rownum,colnum): tuple.  The row and column index of the tile passed in

        """
        #Get list of unique row and col coordinates
        x1 = sorted(set([i[0] for i in tiles]))
        x2 = sorted(set([i[1] for i in tiles]))
        x3 = sorted(set([i[2] for i in tiles]))
        x4 = sorted(set([i[3] for i in tiles]))
        cols = list(zip(x1,x3)) #enclose iterable zip in list() to make reusable
        rows = list(zip(x2,x4))

        #Find a particular tile's coordinates in the list
        colnum  = [n for (n,tpl) in enumerate(cols) if (tile[0],tile[2]) == tpl]
        rownum = [n for (n,tpl) in enumerate(rows) if (tile[1],tile[3]) == tpl]
        return (rownum,colnum)

    def plot_image_boxes(self,im_src, an_src, src_name):
        """Plots bounding boxes on original images with original annotation using `matplotlib`.
        Parameters
          im_src: str. Directory of full-size images
          an_src: str. Directory of full-size annotations
          src_name: str image name without extension
          
        Returns: None
        Side effect: Plots original image with original annotations
        """    
        #Plot original image
        #Find the original xml file with bbox annotations for this image:
        tree = ET.parse(an_src + '/' + src_name + '.xml')
        root = tree.getroot()
        #Find the image and convert to Numpy array
        maybe_fname = src_name + '.jpg'
        src_fullpath = find_filename_case_insensitive(im_src, maybe_fname) #to deal with .jpg vs .JPG
        im = Image.open(src_fullpath)
        im = np.array(im, dtype=np.uint8)

        #list the original (un-tiled) bounding boxes.  Text has to be converted to float, then int, then put into a tuple
        rois = []
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            rois.append((int(float(bbx[0].text)), int(float(bbx[1].text)),int(float(bbx[2].text)), int(float(bbx[3].text))))

        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 10))

        # Display the image
        ax.imshow(im)

        #Display the bounding boxes on top of it
        for roi in rois:
            # Create a Rectangle patch
            rect = patches.Rectangle((roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
                                     linewidth=3, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

    def plot_tile_boxes(self,im_src,im_dst, an_dst, src_fullpath,tile_names,tile_size,tile_overlap):
        """
        Plots a matrix of tiles from a tiled image in correct row/column locations.
        Parameters
        ----------
          im_src: str. Directory of full-size images
          im_dst: str. Directory of tiled images
          an_dst: str. Directory of tiled annotations.
          src_fullpath: full path to original (un-tiled) file
          src_name: filename only of original (un-tiled) file, without extension
          tile_names: list of filenames for the tiles corresponding to the src_img_name file
          tile_size: tuple(int,int).  Tile size in pixels.  Set in __slice_images().
          tile_overlap: float.  Proportion of overlap between consecutive tiles.  
        
        Returns: None
        Side-effect: Plots a matrix of tiles from a tiled image.
        """
        #Get tile matrix dimensions for this particular image
        #Image must be padded and then tiles calculated.
        orig_size = Image.open(src_fullpath).size
        tile_size = tile_size
        tile_overlap = tile_overlap
        padding = calc_padding(orig_size,tile_size,tile_overlap)
        img_size = (orig_size[0] + padding[2],orig_size[1] + padding[3])
        #We don't care about the tiles, just the side-effect of setting tilematrix_dim
        _ = self.__get_tiles(img_size, tile_size, tile_overlap)
        rows,cols = self._tilematrix_dim
        #Create a matrix of empty subplots (n_rows x n_cols)
        pos = []
        for i in range(0, rows):
            for j in range(0, cols):
                pos.append((i, j))
        fig, ax = plt.subplots(rows, cols, sharex='col',
                               sharey='row', figsize=(10, 7))
        for tile in tile_names:
            #Get the tile annotation
            tree = ET.parse(an_dst + '/' + tile + '.xml')
            root = tree.getroot()
            #Get the tile image
            maybe_tilepath = tile + '.jpg'
            tilepath = find_filename_case_insensitive(im_dst, maybe_tilepath) #to deal with .jpg vs .JPG
            im = Image.open(tilepath)
            #Extract the tile row & col coordinates from the name        
            clist = re.findall(r"\[([0-9]+)\]", tile)
            coords = tuple([int(x) for x in clist])
            #Convert the image to an Numpy array
            im = np.array(im, dtype=np.uint8)

            #Make a list of bboxes
            rois = []
            for member in root.findall('object'):
                rois.append((int(float(member[4][0].text)), int(float(member[4][1].text)),
                             int(float(member[4][2].text)), int(float(member[4][3].text))))

            # Display the tile at the right position
            ax[coords[0], coords[1]].imshow(im)
            #ax[pos[idx][0], pos[idx][1]].imshow(im)

            #Show the bounding boxes on the tile
            for roi in rois:
                # Create a Rectangle patch
                rect = patches.Rectangle((roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
                                         linewidth=3, edgecolor='b', facecolor='none')
                # Add the patch to the Axes
                ax[coords[0], coords[1]].add_patch(rect)
        plt.show()

class Points(Enum):
    """An Enum to hold info of points of a bounding box or a tile.
    Used by the method `which_points_lie` and a private method in `Slicer` class. 
    See `which_points_lie` method for more details.

    Example
    ----------
    A box and its points
    P1- - - - - - -P2
    |               |
    |               |
    |               |
    |               |
    P3- - - - - - -P4
    """

    P1, P2, P3, P4 = 1, 2, 3, 4
    P1_P2 = 5
    P1_P3 = 6
    P2_P4 = 7
    P3_P4 = 8
    ALL, NONE = 9, 10


def which_points_lie(label, tile):
    """Method to check if/which points of a label lie inside/on the tile.

    Parameters
    ----------
    label: tuple
        A tuple with label coordinates in `(xmin, ymin, xmax, ymax)` format.
    tile: tuple
        A tuple with tile coordinates in `(xmin, ymin, xmax, ymax)` format.  

    Note
    ----------
    Ignoring the cases where either all 4 points of the `label` or none of them lie on the `tile`, 
    at most only 2 points can lie on the `tile`. 

    Returns
    ----------
    Point (Enum)
        Specifies which point(s) of the `label` lie on the `tile`.
    """
    # 0,1 -- 2,1
    # |        |
    # 0,3 -- 2,3
    points = [False, False, False, False]

    if (tile[0] <= label[0] and tile[2] >= label[0]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[0] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[2] = True

    if (tile[0] <= label[2] and tile[2] >= label[2]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[1] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[3] = True

    if sum(points) == 0:
        return Points.NONE
    elif sum(points) == 4:
        return Points.ALL

    elif points[0]:
        if points[1]:
            return Points.P1_P2
        elif points[2]:
            return Points.P1_P3
        else:
            return Points.P1

    elif points[1]:
        if points[3]:
            return Points.P2_P4
        else:
            return Points.P2

    elif points[2]:
        if points[3]:
            return Points.P3_P4
        else:
            return Points.P3

    else:
        return Points.P4
