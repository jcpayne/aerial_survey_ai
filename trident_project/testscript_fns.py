def setup_model_configuration(rootdir, output_dir, CLASS_NAMES):
    cfg = get_cfg()
    add_tridentnet_config(cfg) #Loads a few values that distinguish TridentNet from Detectron2
    
    #OPTIONAL [WARNING!]  Specify model weights
    #cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl" #original weights
    cfg.MODEL.WEIGHTS = str(Path(rootdir)/"model_final.pth") #This is in blobstore/temp directory
    
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
    cfg.SOLVER.IMS_PER_BATCH = 1
        
    #Learning Rates
    #From Goyal et al. 2017: Linear Scaling Rule: When the minibatch size is multiplied by k, 
    #multiply the learning rate by k.  i.e., as you increase the batch size because of using 
    #additional GPUs, increase the learning rate too.  Works up to very large batches (8,000 images)
    #See auto_scale_workers() in Detectron2 (very important!)
    cfg.SOLVER.BASE_LR = 0.0005 #Is .001 in defaults.py but .02 in tridentnet, but they trained on 8 GPUs
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    
    #Pixel means are from 19261 500x500 tiles on Aug 15 2020 (train_dict)
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.PIXEL_MEAN = [143.078, 140.690, 120.606] #first 5K batch was [137.473, 139.769, 106.912] #default was [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [34.139, 29.849, 31.695]#first 5k batch was [26.428, 22.083, 26.153]
        
    #Auugmentation. Add corruption to images with probability p
    cfg.INPUT.corruption = 0.1
    
    cfg.INPUT.MIN_SIZE_TRAIN = (400,420,440,460,480,500) #See next cell for how this was calculated
    cfg.INPUT.MIN_SIZE_TEST = 500
    
    cfg.DATASETS.TRAIN = ("survey_train",) #Why the comma???  Bcs you can have multiple training datasets
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 24 #Set to equal the number of CPUs.

    # if True, the dataloader will filter out images that have no associated
    # annotations at train time.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = output_dir
    
    #FOR DOING INFERENCE ON GPU/CPU
    cfg.MODEL.DEVICE='cuda' #'cuda' or 'cpu'
    
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
#        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        # We just use the image that is passed in, instead of opening the file and converting
        #The returned dict has 2 things in it: file_name and image (which contains a tensor).  That's all.
#        image = dataset_dict["image"]
        
#        image_shape = image.shape[1:]  # h, w

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
    # batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)
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

#What does itertools.chain do?
#Make an iterator that returns elements from the first iterable until it is exhausted, then 
#proceeds to the next iterable, until all of the iterables are exhausted. Used for treating 
#consecutive sequences as a single sequence. Roughly equivalent to:

# def chain(*iterables):
#     # chain('ABC', 'DEF') --> A B C D E F
#     for it in iterables:
#         for element in it:
#             yield element
            
            
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

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
    @classmethod
    def build_inference_loader(cls, cfg, dataset_name):
        return build_test_loader(cfg, dataset_name, mapper=TileDatasetMapper(cfg,False))  
    
    @classmethod
    def test(cls, cfg, model, data_loader, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results["test"] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
            logger.info("Evaluation results for {} in csv format:".format("test"))
            #print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
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

    n_annfiles = 0
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
        n_annfiles +=1
    #Convert to dataframe and write csv file
    ccounts = pd.DataFrame.from_dict(class_counts,'columns')
    ccounts.to_csv(str(Path(outdir)/'class_counts.csv'), sep=',', encoding='utf-8',index=False)
    print("   Created " + str(n_annfiles) + " new annotation files; ")
    return(class_counts)

def init(rootdir, output_dir=None):
    """
    Initialize the model and prepare for run.
    """
    global cfg
    global model
    global test_metadata
    
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

    print("Building model")
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename 
    #   of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_final.pth')

    
#     modeldir = os.getenv('AZUREML_MODEL_DIR') #This is where the model should be
#     cfg = setup_model_configuration(modeldir, None, CLASS_NAMES)
    cfg = setup_model_configuration(rootdir, None, CLASS_NAMES)
    model = Trainer.build_model(cfg)
    #Load model weights.  Note that it checks cfg for the path, which is based on rootdir
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS) 
    return model   

@rawhttp    
def run(request):
    """Expects a JSON request, which when loaded is a dict containing one item: 
       {"file_list", : [a list of filenames to be processed]}.
    """
    try:
# We are not using actual http requests in testing
#        if request.method == 'POST':
#        reqBody = request.get_data(False)
#        image_files = json.loads(reqBody)['file_list']
        jl = json.loads(request)
        image_files = jl['file_list']
        imagedir = Path('/cdata/tanzania/temp/images_to_process')
        outdir = Path('/cdata/tanzania/temp/outputs')
        results = []
        for f in image_files:
            fpath = str(imagedir/f)
            pil_img = Image.open(fpath)
            pil_img = np.asarray(pil_img)
            pil_img = copy.deepcopy(pil_img) #necessary step
            #Convert to tensor
            img = torch.from_numpy(pil_img)
            img = img.permute(2,0,1)
            fullimage_size = img.shape[1:]

            #Get tiles
            tile_size = (500, 500) #H, W in pixels
            tile_overlap = 100
            tiles = get_tiles(img,tile_size,tile_overlap)

            #Now put in a list and make sure the tiles are contiguous (they weren't)
            nrows, ncols = tiles.shape[1],tiles.shape[2]
            tilelist = []
            tilelist = [tiles[0][ir][ic].contiguous() for ir in range(nrows) for ic in range(ncols) ] 

            #Create a set of tile positions (note: order must match that of tilelist!)
            tile_positions = [{"trow":ir,"tcol":ic} for ir in range(nrows) for ic in range(ncols)]

            #Create a list of dicts (one per image) for Detectron2 input
            datasetdicts = [{"image_id":f, "trow":tile_pos["trow"],"tcol":tile_pos["tcol"],"image":tile,"width":tile_size[1],"height":tile_size[0]} for tile_pos,tile in zip(tile_positions,tilelist)]

            #Create testloader and evaluator
            testloader = build_inference_loader(cfg, "test", datasetdicts,TileDatasetMapper)
            evaluators = DatasetEvaluators([JTFMEvaluator(distributed=True, output_dir=cfg.OUTPUT_DIR)])

            print(f'Running {f}: {len(datasetdicts)} tiles./n')
            print('Note: Detectron count will be misprinted if batch size > 1 (multiply by batch size to get actual count.)')
            tile_results = inference_on_dataset(model, testloader, evaluators)

            #Recombine tile annotations into a full-image annotation 
            print("Reassembling tile annotations")
 
            all_instances = combine_tile_annotations(tile_results,tile_size,tile_overlap,fullimage_size)
            #Create the dict for this file
            file_results = {'file_name':f}
            file_results['height'] = fullimage_size[0]
            file_results['width'] = fullimage_size[1]
            file_results['instances'] = all_instances['instances']
            results.append(file_results)
            
        write_results_to_xml(results,str(outdir),CLASS_NAMES)
        pickled_results = jsonpickle.encode(results) #produces a JSON string 
        print('Done')        
        return pickled_results
#         else:
#             return AMLResponse("bad request, use POST", 500)
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error}) #TODO: Comment out this line for production!