import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((800, 800))
   ]))
# use this dataloader instead of the default
If you use DefaultTrainer, you can overwrite its build_{train,test}_loader method to use your own dataloader. See the densepose dataloader for an example.

THIS IS MY OWN FUCKING FILE; it's in testing_aml_workflow/trident_project/dev_packages/TridentNet/tridentnet/data

